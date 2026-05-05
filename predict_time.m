function [timeStr, confidence] = predict_time(net, imagePath, inputSize)
%PREDICT_TIME Run inference on a single clock image and display results.
%   [timeStr, confidence] = predict_time(net, imagePath, inputSize)
%
%   Reads an image, preprocesses it, runs it through the two-head model,
%   and displays the result with confidence visualization.
%
%   Inputs:
%     net       - trained dlnetwork
%     imagePath - path to a clock image
%     inputSize - [height width] for preprocessing (default: [224 224])
%
%   Outputs:
%     timeStr    - predicted time as "hh:mm" string
%     confidence - struct with .hour and .minute probability vectors

    if nargin < 3, inputSize = [224 224]; end
    addpath('utils');

    % =====================================================================
    % 1. Read and preprocess the image
    % =====================================================================
    if ~isfile(imagePath)
        error('Image not found: %s', imagePath);
    end

    originalImg = imread(imagePath);
    img = preprocess_clock_image(imagePath, inputSize);

    % Create dlarray (SSCB format, batch of 1)
    X = dlarray(img, 'SSCB');

    if canUseGPU()
        X = gpuArray(X);
    end

    % =====================================================================
    % 2. Forward pass
    % =====================================================================
    [YHour, YMin] = predict(net, X, 'Outputs', net.OutputNames);

    hourProbs = double(extractdata(YHour));
    minProbs  = double(extractdata(YMin));

    % =====================================================================
    % 3. Decode predictions
    % =====================================================================
    [hourConf, hourIdx] = max(hourProbs);
    [minConf,  minIdx]  = max(minProbs);

    predHour   = hourIdx;          % 1..12
    predMinute = minIdx - 1;       % 1..60 -> 0..59

    timeStr = sprintf('%02d:%02d', predHour, predMinute);

    confidence.hour        = hourProbs;
    confidence.minute      = minProbs;
    confidence.hourConf    = hourConf;
    confidence.minuteConf  = minConf;
    confidence.predHour    = predHour;
    confidence.predMinute  = predMinute;

    % =====================================================================
    % 4. Display results
    % =====================================================================
    fig = figure('Name', 'Clock Time Prediction', 'Position', [100 100 1200 500]);

    % --- Original image with prediction overlay ---
    subplot(1, 3, 1);
    imshow(originalImg);
    title(sprintf('Predicted: %s', timeStr), 'FontSize', 16, 'FontWeight', 'bold');
    xlabel(sprintf('Hour conf: %.1f%%  |  Minute conf: %.1f%%', ...
        hourConf * 100, minConf * 100), 'FontSize', 11);

    % --- Hour confidence bar chart ---
    subplot(1, 3, 2);
    barH = bar(1:12, hourProbs, 'FaceColor', 'flat');
    colors = repmat([0.6 0.6 0.6], 12, 1);       % gray for all
    colors(hourIdx, :) = [0.2 0.7 0.3];           % green for predicted
    barH.CData = colors;
    xlabel('Hour');
    ylabel('Probability');
    title(sprintf('Hour Prediction: %d (%.1f%%)', predHour, hourConf*100), ...
        'FontSize', 12);
    xticks(1:12);
    xlim([0.5 12.5]);
    ylim([0 1]);
    grid on;

    % --- Minute confidence bar chart (grouped by 5-min for readability) ---
    subplot(1, 3, 3);
    barM = bar(0:59, minProbs, 'FaceColor', 'flat');
    mColors = repmat([0.6 0.6 0.6], 60, 1);
    mColors(minIdx, :) = [0.3 0.5 0.9];           % blue for predicted
    barM.CData = mColors;
    xlabel('Minute');
    ylabel('Probability');
    title(sprintf('Minute Prediction: %d (%.1f%%)', predMinute, minConf*100), ...
        'FontSize', 12);
    xticks(0:5:55);
    xlim([-0.5 59.5]);
    ylim([0 max(minProbs)*1.2]);
    grid on;

    % Print result to command window
    fprintf('\n');
    fprintf('╔════════════════════════════════════╗\n');
    fprintf('║   PREDICTED TIME:  %s            ║\n', timeStr);
    fprintf('║   Hour confidence:  %5.1f%%        ║\n', hourConf * 100);
    fprintf('║   Minute confidence: %5.1f%%       ║\n', minConf * 100);
    fprintf('╚════════════════════════════════════╝\n');
    fprintf('\n');

    % Top-3 hour predictions
    [sortedH, idxH] = sort(hourProbs, 'descend');
    fprintf('Top-3 hour predictions:\n');
    for k = 1:min(3, numel(sortedH))
        fprintf('  %2d:00  (%.1f%%)\n', idxH(k), sortedH(k)*100);
    end

    % Top-3 minute predictions
    [sortedM, idxM] = sort(minProbs, 'descend');
    fprintf('Top-3 minute predictions:\n');
    for k = 1:min(3, numel(sortedM))
        fprintf('  :%02d    (%.1f%%)\n', idxM(k)-1, sortedM(k)*100);
    end
end
