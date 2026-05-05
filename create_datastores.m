function [dsTrain, dsVal, dsTest, info] = create_datastores(T, inputSize, batchSize, valRatio)
%CREATE_DATASTORES Build train/val/test datastores from annotation table.
%   [dsTrain, dsVal, dsTest, info] = create_datastores(T, inputSize, batchSize, valRatio)
%
%   Splits the training data into train/val, creates combined datastores
%   with image preprocessing, and wraps them in minibatchqueue objects
%   for the custom training loop.
%
%   Inputs:
%     T         - annotation table from load_annotations()
%     inputSize - [height width], e.g. [224 224]  (default: [224 224])
%     batchSize - mini-batch size                  (default: 32)
%     valRatio  - fraction of train for validation (default: 0.15)
%
%   Outputs:
%     dsTrain - minibatchqueue for training (with augmentation)
%     dsVal   - minibatchqueue for validation
%     dsTest  - minibatchqueue for testing
%     info    - struct with label vectors and split sizes

    if nargin < 2 || isempty(inputSize), inputSize = [224 224]; end
    if nargin < 3 || isempty(batchSize), batchSize = 32; end
    if nargin < 4 || isempty(valRatio),  valRatio  = 0.15; end

    % =====================================================================
    % 1. Split train data into train + val (stratified by hour)
    % =====================================================================
    trainMask = T.split == "train";
    testMask  = T.split == "test";

    Ttrain_all = T(trainMask, :);
    Ttest      = T(testMask,  :);

    % Stratified split: maintain hour distribution in train and val
    rng(42, 'twister');  % reproducibility
    nTrain = height(Ttrain_all);
    cv = cvpartition(Ttrain_all.hour, 'HoldOut', valRatio);

    Ttrain = Ttrain_all(training(cv), :);
    Tval   = Ttrain_all(test(cv), :);

    fprintf('Datastore splits:\n');
    fprintf('  Train: %d samples\n', height(Ttrain));
    fprintf('  Val:   %d samples\n', height(Tval));
    fprintf('  Test:  %d samples\n', height(Ttest));

    % =====================================================================
    % 2. Create base datastores for each split
    % =====================================================================
    dsTrain = createMBQ(Ttrain, inputSize, batchSize, true);
    dsVal   = createMBQ(Tval,   inputSize, batchSize, false);
    dsTest  = createMBQ(Ttest,  inputSize, batchSize, false);

    % =====================================================================
    % 3. Return metadata
    % =====================================================================
    info.trainTable = Ttrain;
    info.valTable   = Tval;
    info.testTable  = Ttest;
    info.numHours   = 12;
    info.numMinutes = 60;
    info.inputSize  = [inputSize 3];
    info.batchSize  = batchSize;
end


function mbq = createMBQ(Tsplit, inputSize, batchSize, doAugment)
%CREATEMBQ Create a minibatchqueue from an annotation table subset.

    % Image datastore
    imds = imageDatastore(Tsplit.imagePath);

    % Label datastores (store as numeric indices for one-hot encoding later)
    % Hour: 1..12, Minute: 1..60 (offset minute by +1 for 1-based indexing)
    hourIdx = double(Tsplit.hour);          % 1..12
    minIdx  = double(Tsplit.minute) + 1;    % 0..59 -> 1..60

    hourDs = arrayDatastore(hourIdx);
    minDs  = arrayDatastore(minIdx);

    % Combine into a single datastore
    cds = combine(imds, hourDs, minDs);

    % Transform: preprocess images (resize, gray2rgb, augment if training)
    tds = transform(cds, @(data) preprocessSample(data, inputSize, doAugment));

    % Create minibatchqueue
    %   Output 1: images  [H x W x 3 x B] -> dlarray 'SSCB'
    %   Output 2: hour indices  [1 x B]    -> plain double (for one-hot in loop)
    %   Output 3: minute indices [1 x B]   -> plain double (for one-hot in loop)
    mbq = minibatchqueue(tds, 3, ...
        'MiniBatchSize', batchSize, ...
        'MiniBatchFormat', {'SSCB', '', ''}, ...
        'OutputAsDlarray', [true, false, false], ...
        'OutputCast', {'single', 'double', 'double'}, ...
        'PartialMiniBatch', 'return');
end


function out = preprocessSample(data, inputSize, doAugment)
%PREPROCESSSAMPLE Process one sample from the combined datastore.
%   data is a 1x3 cell: {image, hourIdx, minuteIdx}

    img = data{1};

    % --- Handle channel issues ---
    if size(img, 3) == 1
        img = repmat(img, [1 1 3]);   % grayscale -> RGB
    elseif size(img, 3) == 4
        img = img(:,:,1:3);           % RGBA -> RGB
    end

    % --- Training augmentation ---
    if doAugment
        % Random rotation [-15, +15] degrees (no flip!)
        angle = -15 + 30 * rand();
        img = imrotate(img, angle, 'bilinear', 'crop');

        % Random brightness jitter [-20, +20] intensity
        jitter = -20 + 40 * rand();
        img = im2uint8(double(img) + jitter);

        % Random contrast adjustment [0.8, 1.2]
        contrast = 0.8 + 0.4 * rand();
        img = im2uint8(double(img) * contrast);
    end

    % --- Resize to network input size ---
    img = imresize(img, inputSize(1:2));

    % --- Convert to single precision ---
    img = single(img);

    % --- Pass through labels unchanged ---
    out = {img, data{2}, data{3}};
end
