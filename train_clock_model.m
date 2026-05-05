function [net, trainingLog] = train_clock_model(net, dsTrain, dsVal, opts)
%TRAIN_CLOCK_MODEL Train the two-head clock model with a custom loop.
%   [net, trainingLog] = train_clock_model(net, dsTrain, dsVal, opts)
%
%   Uses Adam optimizer with step-decay learning rate and early stopping.
%   Monitors validation loss to save the best model checkpoint.
%
%   Inputs:
%     net     - dlnetwork from build_clock_model()
%     dsTrain - training minibatchqueue from create_datastores()
%     dsVal   - validation minibatchqueue from create_datastores()
%     opts    - (optional) struct with training hyperparameters:
%       .maxEpochs     - max training epochs      (default: 30)
%       .initialLR     - starting learning rate    (default: 1e-4)
%       .lrDropFactor  - LR multiplier at drop     (default: 0.5)
%       .lrDropPeriod  - epochs between LR drops   (default: 10)
%       .patience      - early stopping patience   (default: 5)
%       .savePath      - path to save best model   (default: 'results/trained_model.mat')
%       .hourWeight    - loss weight for hour head  (default: 1.0)
%       .minuteWeight  - loss weight for min head   (default: 1.0)
%
%   Outputs:
%     net         - best trained dlnetwork (lowest val loss)
%     trainingLog - struct with per-epoch metrics for plotting

    % =====================================================================
    % Default options
    % =====================================================================
    if nargin < 4, opts = struct(); end
    if ~isfield(opts, 'maxEpochs'),    opts.maxEpochs    = 30;    end
    if ~isfield(opts, 'initialLR'),    opts.initialLR    = 1e-4;  end
    if ~isfield(opts, 'lrDropFactor'), opts.lrDropFactor = 0.5;   end
    if ~isfield(opts, 'lrDropPeriod'), opts.lrDropPeriod = 10;    end
    if ~isfield(opts, 'patience'),     opts.patience     = 5;     end
    if ~isfield(opts, 'savePath'),     opts.savePath     = fullfile('results', 'trained_model.mat'); end
    if ~isfield(opts, 'hourWeight'),   opts.hourWeight   = 1.0;   end
    if ~isfield(opts, 'minuteWeight'), opts.minuteWeight = 1.0;   end

    fprintf('=== Training Configuration ===\n');
    fprintf('  Max epochs:      %d\n', opts.maxEpochs);
    fprintf('  Initial LR:      %.2e\n', opts.initialLR);
    fprintf('  LR drop:         x%.2f every %d epochs\n', opts.lrDropFactor, opts.lrDropPeriod);
    fprintf('  Early stopping:  patience = %d\n', opts.patience);
    fprintf('  GPU available:   %s\n', string(canUseGPU()));
    fprintf('  Save path:       %s\n', opts.savePath);
    fprintf('==============================\n\n');

    % =====================================================================
    % Initialize optimizer state (Adam)
    % =====================================================================
    avgGrad  = [];
    avgSqGrad = [];
    lr = opts.initialLR;
    iteration = 0;

    % =====================================================================
    % Initialize logging
    % =====================================================================
    trainingLog.epochTrainLoss   = [];
    trainingLog.epochValLoss     = [];
    trainingLog.epochTrainHourAcc  = [];
    trainingLog.epochTrainMinAcc   = [];
    trainingLog.epochValHourAcc    = [];
    trainingLog.epochValMinAcc     = [];
    trainingLog.learningRate     = [];

    bestValLoss = Inf;
    bestNet     = net;
    waitCount   = 0;

    % =====================================================================
    % Create live training plot
    % =====================================================================
    fig = figure('Name', 'Training Progress', 'Position', [100 100 1000 400]);

    ax1 = subplot(1, 2, 1);
    title(ax1, 'Loss');
    xlabel(ax1, 'Epoch');
    ylabel(ax1, 'Loss');
    hold(ax1, 'on');
    grid(ax1, 'on');

    ax2 = subplot(1, 2, 2);
    title(ax2, 'Accuracy');
    xlabel(ax2, 'Epoch');
    ylabel(ax2, 'Accuracy (%)');
    hold(ax2, 'on');
    grid(ax2, 'on');

    % =====================================================================
    % Training loop
    % =====================================================================
    totalStart = tic;

    for epoch = 1:opts.maxEpochs
        epochStart = tic;

        % --- Learning rate step decay ---
        lr = opts.initialLR * opts.lrDropFactor^(floor((epoch - 1) / opts.lrDropPeriod));

        % ---- TRAINING PHASE ----
        reset(dsTrain);
        epochLoss = 0;
        epochHourCorrect = 0;
        epochMinCorrect  = 0;
        epochSamples     = 0;
        batchCount       = 0;

        while hasdata(dsTrain)
            iteration = iteration + 1;

            % Read mini-batch
            [X, hourIdx, minIdx] = next(dsTrain);
            batchSize = size(X, 4);

            % One-hot encode targets
            THour = oneHotEncode(hourIdx, 12);
            TMin  = oneHotEncode(minIdx,  60);

            % Move to GPU if available
            if canUseGPU()
                X     = gpuArray(X);
                THour = gpuArray(THour);
                TMin  = gpuArray(TMin);
            end

            % Forward + loss + gradients
            [loss, gradients, hourLoss, minLoss] = dlfeval(@modelLoss, ...
                net, X, THour, TMin, opts.hourWeight, opts.minuteWeight);

            % Update weights (Adam)
            [net, avgGrad, avgSqGrad] = adamupdate(net, gradients, ...
                avgGrad, avgSqGrad, iteration, lr);

            % Accumulate metrics
            epochLoss = epochLoss + double(extractdata(loss)) * batchSize;

            % Compute accuracy for this batch
            [predH, predM] = predictBatch(net, X);
            trueH = extractIndices(hourIdx);
            trueM = extractIndices(minIdx);
            epochHourCorrect = epochHourCorrect + sum(predH == trueH);
            epochMinCorrect  = epochMinCorrect  + sum(predM == trueM);
            epochSamples     = epochSamples + batchSize;
            batchCount       = batchCount + 1;
        end

        trainLoss    = epochLoss / epochSamples;
        trainHourAcc = epochHourCorrect / epochSamples * 100;
        trainMinAcc  = epochMinCorrect  / epochSamples * 100;

        % ---- VALIDATION PHASE ----
        [valLoss, valHourAcc, valMinAcc] = evaluateOnSet(net, dsVal, ...
            opts.hourWeight, opts.minuteWeight);

        epochTime = toc(epochStart);

        % ---- Logging ----
        trainingLog.epochTrainLoss(epoch)    = trainLoss;
        trainingLog.epochValLoss(epoch)      = valLoss;
        trainingLog.epochTrainHourAcc(epoch) = trainHourAcc;
        trainingLog.epochTrainMinAcc(epoch)  = trainMinAcc;
        trainingLog.epochValHourAcc(epoch)   = valHourAcc;
        trainingLog.epochValMinAcc(epoch)    = valMinAcc;
        trainingLog.learningRate(epoch)      = lr;

        % ---- Print progress ----
        fprintf('Epoch %2d/%d | %.0fs | LR=%.2e | Train Loss=%.4f | Val Loss=%.4f | ', ...
            epoch, opts.maxEpochs, epochTime, lr, trainLoss, valLoss);
        fprintf('Train H=%.1f%% M=%.1f%% | Val H=%.1f%% M=%.1f%%\n', ...
            trainHourAcc, trainMinAcc, valHourAcc, valMinAcc);

        % ---- Update live plot ----
        updatePlots(ax1, ax2, trainingLog, epoch);

        % ---- Early stopping + checkpointing ----
        if valLoss < bestValLoss
            bestValLoss = valLoss;
            bestNet = net;
            waitCount = 0;

            % Save best model
            saveDir = fileparts(opts.savePath);
            if ~isfolder(saveDir), mkdir(saveDir); end
            save(opts.savePath, 'net', 'trainingLog', 'opts', '-v7.3');
            fprintf('  >> Best model saved (val loss = %.4f)\n', valLoss);
        else
            waitCount = waitCount + 1;
            if waitCount >= opts.patience
                fprintf('\nEarly stopping triggered after %d epochs without improvement.\n', opts.patience);
                break;
            end
            fprintf('  >> No improvement (%d/%d patience)\n', waitCount, opts.patience);
        end
    end

    totalTime = toc(totalStart);
    fprintf('\n=== Training complete in %.1f minutes ===\n', totalTime / 60);
    fprintf('Best validation loss: %.4f\n', bestValLoss);

    % Return best model
    net = bestNet;

    % Save final training log
    logPath = fullfile('results', 'training_log.mat');
    save(logPath, 'trainingLog', 'opts', '-v7.3');
    fprintf('Training log saved to: %s\n', logPath);

    % Save training plots
    saveas(fig, fullfile('results', 'figures', 'training_curves.png'));
    saveas(fig, fullfile('results', 'figures', 'training_curves.fig'));
    fprintf('Training plots saved.\n');
end


% =========================================================================
% HELPER FUNCTIONS
% =========================================================================

function [loss, gradients, hourLoss, minLoss] = modelLoss(net, X, THour, TMin, wH, wM)
%MODELLOSS Combined loss for both heads.
    [YHour, YMin] = forward(net, X, 'Outputs', net.OutputNames);

    hourLoss = crossentropy(YHour, THour);
    minLoss  = crossentropy(YMin,  TMin);

    loss = wH * hourLoss + wM * minLoss;

    gradients = dlgradient(loss, net.Learnables);
end


function T = oneHotEncode(indices, numClasses)
%ONEHOTENCODE Convert numeric indices to one-hot dlarray.
    indices = double(extractdata(gather(indices)));
    indices = indices(:)';  % ensure row vector
    B = numel(indices);
    T = zeros(numClasses, B, 'single');
    linearIdx = sub2ind([numClasses, B], indices, 1:B);
    T(linearIdx) = 1;
    T = dlarray(T, 'CB');
end


function [predH, predM] = predictBatch(net, X)
%PREDICTBATCH Get predicted hour and minute indices for a batch.
    [YHour, YMin] = predict(net, X, 'Outputs', net.OutputNames);
    [~, predH] = max(extractdata(YHour), [], 1);
    [~, predM] = max(extractdata(YMin),  [], 1);
end


function idx = extractIndices(indices)
%EXTRACTINDICES Convert minibatchqueue label output to double vector.
    if isa(indices, 'dlarray')
        idx = double(extractdata(gather(indices)));
    else
        idx = double(gather(indices));
    end
    idx = idx(:)';
end


function [totalLoss, hourAcc, minAcc] = evaluateOnSet(net, ds, wH, wM)
%EVALUATEONSET Compute loss and accuracy over an entire dataset (no gradients).
    reset(ds);
    totalLoss    = 0;
    hourCorrect  = 0;
    minCorrect   = 0;
    totalSamples = 0;

    while hasdata(ds)
        [X, hourIdx, minIdx] = next(ds);
        batchSize = size(X, 4);

        THour = oneHotEncode(hourIdx, 12);
        TMin  = oneHotEncode(minIdx,  60);

        if canUseGPU()
            X     = gpuArray(X);
            THour = gpuArray(THour);
            TMin  = gpuArray(TMin);
        end

        % Forward pass (no gradient tracking)
        [YHour, YMin] = predict(net, X, 'Outputs', net.OutputNames);

        hourLoss = crossentropy(YHour, THour);
        minLoss  = crossentropy(YMin,  TMin);
        loss = wH * hourLoss + wM * minLoss;

        totalLoss = totalLoss + double(extractdata(loss)) * batchSize;

        [~, predH] = max(extractdata(YHour), [], 1);
        [~, predM] = max(extractdata(YMin),  [], 1);
        trueH = extractIndices(hourIdx);
        trueM = extractIndices(minIdx);

        hourCorrect  = hourCorrect  + sum(predH == trueH);
        minCorrect   = minCorrect   + sum(predM == trueM);
        totalSamples = totalSamples + batchSize;
    end

    totalLoss = totalLoss / totalSamples;
    hourAcc   = hourCorrect / totalSamples * 100;
    minAcc    = minCorrect  / totalSamples * 100;
end


function updatePlots(ax1, ax2, log, epoch)
%UPDATEPLOTS Refresh the live training plots.
    epochs = 1:epoch;

    % Loss plot
    cla(ax1);
    plot(ax1, epochs, log.epochTrainLoss(epochs), 'b-o', 'LineWidth', 1.5, 'MarkerSize', 4);
    plot(ax1, epochs, log.epochValLoss(epochs),   'r-s', 'LineWidth', 1.5, 'MarkerSize', 4);
    legend(ax1, 'Train', 'Validation', 'Location', 'northeast');

    % Accuracy plot
    cla(ax2);
    plot(ax2, epochs, log.epochTrainHourAcc(epochs), 'b-o', 'LineWidth', 1.5, 'MarkerSize', 4);
    plot(ax2, epochs, log.epochValHourAcc(epochs),   'r-s', 'LineWidth', 1.5, 'MarkerSize', 4);
    plot(ax2, epochs, log.epochTrainMinAcc(epochs),  'b--^', 'LineWidth', 1.2, 'MarkerSize', 4);
    plot(ax2, epochs, log.epochValMinAcc(epochs),    'r--v', 'LineWidth', 1.2, 'MarkerSize', 4);
    legend(ax2, 'Train Hour', 'Val Hour', 'Train Minute', 'Val Minute', 'Location', 'southeast');

    drawnow;
end
