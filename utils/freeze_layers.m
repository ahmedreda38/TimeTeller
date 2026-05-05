function net = freeze_layers(net, layerNames)
%FREEZE_LAYERS Set learn rate factor to 0 for specified layers.
%   net = freeze_layers(net, layerNames)
%
%   This prevents weights/biases in the given layers from being updated
%   during training. Used to freeze the pretrained ResNet-18 backbone.
%
%   Inputs:
%     net        - a dlnetwork object
%     layerNames - string array or cell array of layer names to freeze
%
%   Output:
%     net - the modified dlnetwork with frozen layers

    learnables = net.Learnables;

    for i = 1:height(learnables)
        lName = string(learnables.Layer(i));
        pName = string(learnables.Parameter(i));

        if ismember(lName, string(layerNames))
            net = setLearnRateFactor(net, lName, pName, 0);
        end
    end

    fprintf('Froze %d layers (%d parameters set to LR=0).\n', ...
        numel(layerNames), height(learnables(ismember(string(learnables.Layer), string(layerNames)), :)));
end
