function T = load_annotations(jsonPath, imageBaseDir)
%LOAD_ANNOTATIONS Parse TickTockVQA annotations.json into a MATLAB table.
%   T = load_annotations(jsonPath, imageBaseDir)
%
%   Reads the JSON annotation file, validates image paths, and returns
%   a structured table ready for datastore creation.
%
%   Inputs:
%     jsonPath     - path to annotations.json
%     imageBaseDir - root folder containing 'images/train/' and 'images/test/'
%
%   Output:
%     T - table with columns:
%         imagePath   (string)  - absolute path to image file
%         hour        (double)  - hour 1..12
%         minute      (double)  - minute 0..59
%         hourLabel   (categorical) - "h01".."h12"
%         minuteLabel (categorical) - "m00".."m59"
%         split       (string)  - "train" or "test"
%         clockType   (string)  - type of clock
%         source      (string)  - data source

    fprintf('Loading annotations from: %s\n', jsonPath);

    % --- Read and decode JSON ---
    rawText = fileread(jsonPath);
    data = jsondecode(rawText);

    n = numel(data);
    fprintf('Parsed %d annotation records.\n', n);

    % --- Pre-allocate arrays ---
    imagePaths  = strings(n, 1);
    hours       = zeros(n, 1);
    minutes     = zeros(n, 1);
    hourLabels  = strings(n, 1);
    minLabels   = strings(n, 1);
    splits      = strings(n, 1);
    clockTypes  = strings(n, 1);
    sources     = strings(n, 1);

    % --- Define all valid categories upfront ---
    hourCats  = compose("h%02d", 1:12);    % "h01" .. "h12"
    minCats   = compose("m%02d", 0:59);    % "m00" .. "m59"

    % --- Parse each record ---
    validMask = true(n, 1);

    for i = 1:n
        rec = data(i);

        % Build image path: image_path is like "train/xxx.jpg"
        % We need to prepend "images/" and the base directory
        relPath = string(rec.image_path);
        fullPath = fullfile(imageBaseDir, 'images', relPath);

        % Check if file exists
        if ~isfile(fullPath)
            validMask(i) = false;
            continue;
        end

        imagePaths(i) = fullPath;

        % Hour (1..12) - clamp to valid range
        h = double(rec.hour);
        if h < 1 || h > 12
            validMask(i) = false;
            continue;
        end
        hours(i) = h;

        % Minute (0..59) - clamp to valid range
        m = double(rec.minute);
        if m < 0 || m > 59
            validMask(i) = false;
            continue;
        end
        minutes(i) = m;

        % Categorical labels with string prefix for correct ordering
        hourLabels(i) = sprintf("h%02d", h);
        minLabels(i)  = sprintf("m%02d", m);

        % Split: parse from image_path (starts with "train/" or "test/")
        if startsWith(relPath, "train")
            splits(i) = "train";
        elseif startsWith(relPath, "test")
            splits(i) = "test";
        else
            splits(i) = "unknown";
        end

        % Metadata
        if isfield(rec, 'clock_type') && ~isempty(rec.clock_type)
            clockTypes(i) = string(rec.clock_type);
        else
            clockTypes(i) = "Unknown";
        end

        if isfield(rec, 'source') && ~isempty(rec.source)
            sources(i) = string(rec.source);
        else
            sources(i) = "Unknown";
        end
    end

    % --- Filter to valid records only ---
    skipped = sum(~validMask);
    if skipped > 0
        fprintf('Skipped %d records (missing images or invalid hour/minute).\n', skipped);
    end

    imagePaths = imagePaths(validMask);
    hours      = hours(validMask);
    minutes    = minutes(validMask);
    hourLabels = hourLabels(validMask);
    minLabels  = minLabels(validMask);
    splits     = splits(validMask);
    clockTypes = clockTypes(validMask);
    sources    = sources(validMask);

    % --- Convert to categorical with fixed category ordering ---
    hourLabelCat = categorical(hourLabels, hourCats, hourCats);
    minLabelCat  = categorical(minLabels,  minCats,  minCats);

    % --- Build output table ---
    T = table(imagePaths, hours, minutes, hourLabelCat, minLabelCat, ...
              splits, clockTypes, sources, ...
              'VariableNames', {'imagePath', 'hour', 'minute', ...
                                'hourLabel', 'minuteLabel', ...
                                'split', 'clockType', 'source'});

    % --- Summary ---
    trainCount = sum(T.split == "train");
    testCount  = sum(T.split == "test");
    fprintf('\nAnnotation summary:\n');
    fprintf('  Valid records: %d\n', height(T));
    fprintf('  Train split:   %d\n', trainCount);
    fprintf('  Test split:    %d\n', testCount);
    fprintf('  Hour range:    %d - %d\n', min(T.hour), max(T.hour));
    fprintf('  Minute range:  %d - %d\n', min(T.minute), max(T.minute));
end
