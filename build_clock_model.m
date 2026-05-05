function net = build_clock_model(numHours, numMinutes)
%BUILD_CLOCK_MODEL Construct a two-head ResNet-18 for clock reading.
%   net = build_clock_model(numHours, numMinutes)
%
%   Loads ResNet-18 pretrained on ImageNet, removes the original
%   classification head, and adds two parallel branches:
%     - Hour head:   FC(256)->BN->ReLU->Drop(0.4)->FC(12)->Softmax
%     - Minute head: FC(256)->BN->ReLU->Drop(0.4)->FC(60)->Softmax
%
%   The backbone is FROZEN by default (learn rate = 0).
%
%   Inputs:
%     numHours   - number of hour classes   (default: 12)
%     numMinutes - number of minute classes  (default: 60)
%
%   Output:
%     net - initialized dlnetwork with two output heads

    if nargin < 1, numHours   = 12; end
    if nargin < 2, numMinutes = 60; end

    fprintf('Building two-head ResNet-18 model...\n');
    fprintf('  Hour classes:   %d\n', numHours);
    fprintf('  Minute classes: %d\n', numMinutes);

    % =====================================================================
    % 1. Load pretrained ResNet-18
    % =====================================================================
    baseNet = resnet18;
    lgraph = layerGraph(baseNet);

    % Display original final layers for reference
    fprintf('\nOriginal final layers:\n');
    lastLayers = lgraph.Layers(end-3:end);
    for i = 1:numel(lastLayers)
        fprintf('  %s (%s)\n', lastLayers(i).Name, class(lastLayers(i)));
    end

    % =====================================================================
    % 2. Remove original classification layers
    % =====================================================================
    % ResNet-18 ends with: pool5 -> fc1000 -> prob -> ClassificationLayer
    % We keep pool5 (global average pooling) and remove everything after it
    layersToRemove = {'fc1000', 'prob', 'ClassificationLayer_predictions'};

    % Check which layers actually exist (names may vary by MATLAB version)
    existingLayers = {lgraph.Layers.Name};
    layersToRemove = layersToRemove(ismember(layersToRemove, existingLayers));

    lgraph = removeLayers(lgraph, layersToRemove);
    fprintf('\nRemoved %d original classification layers.\n', numel(layersToRemove));

    % =====================================================================
    % 3. Find the connection point (global average pooling layer)
    % =====================================================================
    % The connection point is 'pool5' in standard ResNet-18
    connectFrom = 'pool5';
    if ~ismember(connectFrom, {lgraph.Layers.Name})
        % Fallback: find the global average pooling layer
        for i = numel(lgraph.Layers):-1:1
            if isa(lgraph.Layers(i), 'nnet.cnn.layer.GlobalAveragePooling2DLayer')
                connectFrom = lgraph.Layers(i).Name;
                break;
            end
        end
    end
    fprintf('Branching from layer: %s\n', connectFrom);

    % =====================================================================
    % 4. Create HOUR branch
    % =====================================================================
    hourLayers = [
        fullyConnectedLayer(256, 'Name', 'hour_fc1')
        batchNormalizationLayer('Name', 'hour_bn')
        reluLayer('Name', 'hour_relu')
        dropoutLayer(0.4, 'Name', 'hour_drop')
        fullyConnectedLayer(numHours, 'Name', 'hour_fc2')
        softmaxLayer('Name', 'hour_softmax')
    ];

    lgraph = addLayers(lgraph, hourLayers);
    lgraph = connectLayers(lgraph, connectFrom, 'hour_fc1');

    % =====================================================================
    % 5. Create MINUTE branch
    % =====================================================================
    minuteLayers = [
        fullyConnectedLayer(256, 'Name', 'min_fc1')
        batchNormalizationLayer('Name', 'min_bn')
        reluLayer('Name', 'min_relu')
        dropoutLayer(0.4, 'Name', 'min_drop')
        fullyConnectedLayer(numMinutes, 'Name', 'min_fc2')
        softmaxLayer('Name', 'min_softmax')
    ];

    lgraph = addLayers(lgraph, minuteLayers);
    lgraph = connectLayers(lgraph, connectFrom, 'min_fc1');

    % =====================================================================
    % 6. Convert to dlnetwork
    % =====================================================================
    net = dlnetwork(lgraph);

    fprintf('\nNetwork created with outputs: %s\n', strjoin(net.OutputNames, ', '));
    fprintf('Total learnable parameters: %d\n', sum(cellfun(@numel, net.Learnables.Value)));

    % =====================================================================
    % 7. Freeze backbone layers (everything except hour_* and min_*)
    % =====================================================================
    allLayerNames = string({lgraph.Layers.Name});
    headLayerNames = allLayerNames(startsWith(allLayerNames, "hour_") | ...
                                   startsWith(allLayerNames, "min_"));
    backboneLayerNames = setdiff(allLayerNames, headLayerNames);

    % Only freeze layers that have learnable parameters
    learnableLayerNames = unique(string(net.Learnables.Layer));
    backboneToFreeze = intersect(backboneLayerNames, learnableLayerNames);

    addpath('utils');  % Ensure freeze_layers is accessible
    net = freeze_layers(net, backboneToFreeze);

    fprintf('\nModel architecture summary:\n');
    fprintf('  Backbone layers (frozen): %d\n', numel(backboneToFreeze));
    fprintf('  Hour head layers:         %d\n', sum(startsWith(allLayerNames, "hour_")));
    fprintf('  Minute head layers:       %d\n', sum(startsWith(allLayerNames, "min_")));

    % =====================================================================
    % 8. Verify with a test forward pass
    % =====================================================================
    fprintf('\nVerifying forward pass...\n');
    testInput = dlarray(randn(224, 224, 3, 1, 'single'), 'SSCB');
    [yHour, yMin] = forward(net, testInput, 'Outputs', net.OutputNames);
    fprintf('  Hour output size:   [%s]\n', strjoin(string(size(yHour)), ' x '));
    fprintf('  Minute output size: [%s]\n', strjoin(string(size(yMin)), ' x '));
    fprintf('Model verification PASSED.\n\n');
end
