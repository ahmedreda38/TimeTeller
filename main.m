%% ========================================================================
%  ANALOG CLOCK READER — Main Pipeline
%  ========================================================================
%  Two-head ResNet-18 trained on TickTockVQA dataset.
%  Predicts time from analog clock images as hh:mm.
%
%  Run each section (Ctrl+Enter) cell-by-cell, or run the entire script.
%  ========================================================================

%% SETUP — Add paths and configure
clc; close all;
fprintf('=== Analog Clock Reader Pipeline ===\n\n');

% Add utility functions to path
addpath('utils');

% Configuration
config.dataDir    = fullfile(pwd, 'data', 'dataset');
config.inputSize  = [224 224];
config.batchSize  = 32;
config.valRatio   = 0.15;
config.maxEpochs  = 30;
config.initialLR  = 1e-4;
config.savePath   = fullfile('results', 'trained_model.mat');

% Check GPU availability
if canUseGPU()
    fprintf('GPU detected — training will use GPU acceleration.\n');
    gpuInfo = gpuDevice();
    fprintf('  GPU: %s (%.1f GB memory)\n', gpuInfo.Name, gpuInfo.TotalMemory/1e9);
else
    fprintf('No CUDA GPU detected — training will run on CPU.\n');
    fprintf('NOTE: MATLAB requires NVIDIA GPUs with CUDA support.\n');
    fprintf('      AMD GPUs (like RX 580) are not supported.\n');
    fprintf('      CPU training may take 6-12 hours for 30 epochs.\n');
end
fprintf('\n');

%% STEP 1 — Download Dataset
fprintf('=== STEP 1: Download Dataset ===\n\n');

download_dataset(config.dataDir);

fprintf('\nStep 1 complete.\n\n');

%% STEP 2 — Load Annotations
fprintf('=== STEP 2: Load Annotations ===\n\n');

jsonPath = fullfile(config.dataDir, 'annotations.json');
T = load_annotations(jsonPath, config.dataDir);

% Quick data exploration
fprintf('\nClock type distribution:\n');
typeCounts = groupcounts(T, 'clockType');
disp(typeCounts);

fprintf('Source distribution:\n');
srcCounts = groupcounts(T, 'source');
disp(srcCounts);

fprintf('Step 2 complete.\n\n');

%% STEP 3 — Create Datastores
fprintf('=== STEP 3: Create Datastores ===\n\n');

[dsTrain, dsVal, dsTest, dsInfo] = create_datastores(T, config.inputSize, ...
    config.batchSize, config.valRatio);

% Verify a single batch
fprintf('\nVerifying training batch...\n');
reset(dsTrain);
[Xsample, hSample, mSample] = next(dsTrain);
fprintf('  Image batch size: [%s]\n', strjoin(string(size(Xsample)), ' x '));
fprintf('  Hour labels size: [%s]\n', strjoin(string(size(hSample)), ' x '));
fprintf('  Min labels size:  [%s]\n', strjoin(string(size(mSample)), ' x '));
reset(dsTrain);

% Show a few sample images
fig = figure('Name', 'Sample Training Images', 'Position', [100 100 1200 300]);
for i = 1:min(5, size(Xsample, 4))
    subplot(1, 5, i);
    img = extractdata(Xsample(:,:,:,i));
    imshow(uint8(img));

    h = hSample(i);
    m = mSample(i);
    if isa(h, 'dlarray'), h = extractdata(h); end
    if isa(m, 'dlarray'), m = extractdata(m); end
    title(sprintf('%02d:%02d', h, m - 1), 'FontSize', 12);
end
sgtitle('Sample Training Images', 'FontSize', 14);

fprintf('\nStep 3 complete.\n\n');

%% STEP 4 — Build Model
fprintf('=== STEP 4: Build Two-Head ResNet-18 ===\n\n');

net = build_clock_model(12, 60);

fprintf('Step 4 complete.\n\n');

%% STEP 5 — Train Model
fprintf('=== STEP 5: Train Model ===\n\n');

% Check if a trained model already exists
if isfile(config.savePath)
    answer = input('Trained model found. Retrain? (y/n): ', 's');
    if ~strcmpi(answer, 'y')
        fprintf('Loading existing model...\n');
        loaded = load(config.savePath);
        net = loaded.net;
        if isfield(loaded, 'trainingLog')
            trainingLog = loaded.trainingLog;
        end
        fprintf('Model loaded successfully.\n\n');
    else
        trainOpts.maxEpochs    = config.maxEpochs;
        trainOpts.initialLR    = config.initialLR;
        trainOpts.savePath     = config.savePath;
        [net, trainingLog] = train_clock_model(net, dsTrain, dsVal, trainOpts);
    end
else
    trainOpts.maxEpochs    = config.maxEpochs;
    trainOpts.initialLR    = config.initialLR;
    trainOpts.savePath     = config.savePath;
    [net, trainingLog] = train_clock_model(net, dsTrain, dsVal, trainOpts);
end

fprintf('Step 5 complete.\n\n');

%% STEP 6 — Evaluate on Test Set
fprintf('=== STEP 6: Evaluate Model ===\n\n');

metrics = evaluate_model(net, dsTest, dsInfo.testTable);

% Print target comparison
fprintf('\n--- Target vs Actual ---\n');
fprintf('  Hour accuracy:   %.1f%%  (target: > 85%%)\n', metrics.hourAccuracy);
fprintf('  Minute accuracy: %.1f%%  (target: > 60%%)\n', metrics.minuteAccuracy);
fprintf('  MATE:            %.1f min (target: < 10 min)\n', metrics.MATE);
fprintf('  Within 5 min:    %.1f%%  (target: > 70%%)\n', metrics.within5min);

fprintf('\nStep 6 complete.\n\n');

%% STEP 7 — Demo Inference on Random Test Images
fprintf('=== STEP 7: Demo Inference ===\n\n');

% Pick 3 random test images
rng('shuffle');
testPaths = dsInfo.testTable.imagePath;
nDemo = min(3, numel(testPaths));
demoIdx = randperm(numel(testPaths), nDemo);

for i = 1:nDemo
    imgPath = testPaths(demoIdx(i));
    trueH = dsInfo.testTable.hour(demoIdx(i));
    trueM = dsInfo.testTable.minute(demoIdx(i));

    fprintf('\n--- Demo image %d/%d ---\n', i, nDemo);
    fprintf('Image: %s\n', imgPath);
    fprintf('Ground truth: %02d:%02d\n', trueH, trueM);

    [timeStr, conf] = predict_time(net, imgPath, config.inputSize);

    err = circular_time_error(conf.predHour, conf.predMinute, trueH, trueM);
    fprintf('Time error: %d minutes\n', err);
end

fprintf('\n=== Pipeline Complete ===\n');
fprintf('All results saved in: results/\n');
