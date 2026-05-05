function metrics = evaluate_model(net, dsTest, testTable)
%EVALUATE_MODEL Evaluate the trained clock model on the test set.
%   metrics = evaluate_model(net, dsTest, testTable)
%
%   Computes all target metrics and generates visualization plots.
%
%   Inputs:
%     net       - trained dlnetwork
%     dsTest    - test minibatchqueue
%     testTable - test annotation table (for ground truth metadata)
%
%   Output:
%     metrics - struct with all evaluation metrics:
%       .hourAccuracy     - % correct hours
%       .minuteAccuracy   - % correct minutes
%       .MATE             - Mean Absolute Time Error (minutes)
%       .medianATE        - Median Absolute Time Error
%       .within5min       - % within 5 minutes
%       .within15min      - % within 15 minutes
%       .predHours        - all predicted hours
%       .predMinutes      - all predicted minutes

    fprintf('=== Evaluating on test set (%d samples) ===\n\n', height(testTable));
    addpath('utils');

    % =====================================================================
    % 1. Collect all predictions
    % =====================================================================
    allPredH = [];
    allPredM = [];
    allTrueH = [];
    allTrueM = [];

    reset(dsTest);
    batchNum = 0;

    while hasdata(dsTest)
        [X, hourIdx, minIdx] = next(dsTest);

        if canUseGPU()
            X = gpuArray(X);
        end

        [YHour, YMin] = predict(net, X, 'Outputs', net.OutputNames);

        [~, predH] = max(extractdata(YHour), [], 1);
        [~, predM] = max(extractdata(YMin),  [], 1);

        % Convert indices back to actual values
        % Hour: index 1..12 -> hour 1..12 (no offset)
        % Minute: index 1..60 -> minute 0..59 (subtract 1)
        predH = double(predH(:)');
        predM = double(predM(:)') - 1;   % offset correction

        if isa(hourIdx, 'dlarray')
            trueH = double(extractdata(gather(hourIdx(:)')));
            trueM = double(extractdata(gather(minIdx(:)')))  - 1;
        else
            trueH = double(hourIdx(:)');
            trueM = double(minIdx(:)')  - 1;
        end

        allPredH = [allPredH, predH]; %#ok<AGROW>
        allPredM = [allPredM, predM];
        allTrueH = [allTrueH, trueH];
        allTrueM = [allTrueM, trueM];

        batchNum = batchNum + 1;
        if mod(batchNum, 50) == 0
            fprintf('  Processed %d batches...\n', batchNum);
        end
    end

    N = numel(allPredH);
    fprintf('  Total predictions: %d\n\n', N);

    % =====================================================================
    % 2. Compute metrics
    % =====================================================================

    % Hour accuracy
    hourAcc = mean(allPredH == allTrueH) * 100;

    % Minute accuracy (exact match)
    minAcc = mean(allPredM == allTrueM) * 100;

    % Circular time error (MATE)
    timeErrors = circular_time_error(allPredH, allPredM, allTrueH, allTrueM);
    MATE = mean(timeErrors);
    medianATE = median(timeErrors);

    % Within-N-minute accuracy
    within5  = mean(timeErrors <= 5)  * 100;
    within15 = mean(timeErrors <= 15) * 100;
    within30 = mean(timeErrors <= 30) * 100;

    % =====================================================================
    % 3. Print results
    % =====================================================================
    fprintf('========================================\n');
    fprintf('       TEST SET EVALUATION RESULTS      \n');
    fprintf('========================================\n');
    fprintf('  Hour accuracy:        %6.2f%%\n', hourAcc);
    fprintf('  Minute accuracy:      %6.2f%%\n', minAcc);
    fprintf('  MATE:                 %6.2f min\n', MATE);
    fprintf('  Median ATE:           %6.2f min\n', medianATE);
    fprintf('  Within  5 min:        %6.2f%%\n', within5);
    fprintf('  Within 15 min:        %6.2f%%\n', within15);
    fprintf('  Within 30 min:        %6.2f%%\n', within30);
    fprintf('========================================\n\n');

    % =====================================================================
    % 4. Store metrics
    % =====================================================================
    metrics.hourAccuracy   = hourAcc;
    metrics.minuteAccuracy = minAcc;
    metrics.MATE           = MATE;
    metrics.medianATE      = medianATE;
    metrics.within5min     = within5;
    metrics.within15min    = within15;
    metrics.within30min    = within30;
    metrics.predHours      = allPredH;
    metrics.predMinutes    = allPredM;
    metrics.trueHours      = allTrueH;
    metrics.trueMinutes    = allTrueM;
    metrics.timeErrors     = timeErrors;

    % =====================================================================
    % 5. Generate plots
    % =====================================================================
    figDir = fullfile('results', 'figures');
    if ~isfolder(figDir), mkdir(figDir); end

    % --- 5a. Hour confusion matrix ---
    fig1 = figure('Name', 'Hour Confusion Matrix', 'Position', [100 100 600 500]);
    hourLabels = 1:12;
    cmHour = confusionmat(allTrueH, allPredH, 'Order', hourLabels);
    confusionchart(fig1, cmHour, string(hourLabels), ...
        'Title', 'Hour Prediction Confusion Matrix', ...
        'RowSummary', 'row-normalized', ...
        'ColumnSummary', 'column-normalized');
    saveas(fig1, fullfile(figDir, 'confusion_hour.png'));

    % --- 5b. Minute confusion (binned into 5-min groups) ---
    fig2 = figure('Name', 'Minute Confusion (5-min bins)', 'Position', [120 120 700 600]);
    binEdges = 0:5:60;
    predMBin = discretize(allPredM, binEdges);
    trueMBin = discretize(allTrueM, binEdges);
    binLabels = compose("%02d-%02d", binEdges(1:end-1)', binEdges(2:end)'-1);
    cmMinBin = confusionmat(trueMBin, predMBin, 'Order', 1:12);
    confusionchart(fig2, cmMinBin, binLabels, ...
        'Title', 'Minute Prediction Confusion (5-min bins)', ...
        'RowSummary', 'row-normalized');
    saveas(fig2, fullfile(figDir, 'confusion_minute_binned.png'));

    % --- 5c. Error distribution histogram ---
    fig3 = figure('Name', 'Time Error Distribution', 'Position', [140 140 700 400]);
    histogram(timeErrors, 'BinWidth', 2, 'FaceColor', [0.2 0.6 0.9], ...
        'EdgeColor', 'white', 'FaceAlpha', 0.85);
    hold on;
    xline(MATE, 'r--', sprintf('MATE = %.1f min', MATE), 'LineWidth', 2, ...
        'LabelOrientation', 'horizontal', 'FontSize', 12);
    xline(medianATE, 'g--', sprintf('Median = %.1f min', medianATE), 'LineWidth', 2, ...
        'LabelOrientation', 'horizontal', 'FontSize', 12);
    xlabel('Absolute Time Error (minutes)');
    ylabel('Number of Samples');
    title('Distribution of Time Prediction Errors');
    grid on;
    saveas(fig3, fullfile(figDir, 'error_distribution.png'));

    % --- 5d. Per-hour MATE breakdown ---
    fig4 = figure('Name', 'Per-Hour MATE', 'Position', [160 160 700 400]);
    hourMATE = zeros(1, 12);
    for h = 1:12
        mask = allTrueH == h;
        if sum(mask) > 0
            hourMATE(h) = mean(timeErrors(mask));
        end
    end
    bar(1:12, hourMATE, 'FaceColor', [0.3 0.7 0.5], 'EdgeColor', 'white');
    xlabel('True Hour');
    ylabel('MATE (minutes)');
    title('Mean Absolute Time Error by Hour');
    xticks(1:12);
    grid on;
    saveas(fig4, fullfile(figDir, 'mate_per_hour.png'));

    fprintf('All evaluation figures saved to: %s\n', figDir);

    % Save metrics
    save(fullfile('results', 'test_metrics.mat'), 'metrics', '-v7.3');
    fprintf('Metrics saved to: results/test_metrics.mat\n');
end
