function download_dataset(dataDir)
%DOWNLOAD_DATASET Download the TickTockVQA dataset from HuggingFace.
%   download_dataset(dataDir)
%
%   Downloads the dataset into dataDir using huggingface-cli.
%   Installs huggingface_hub Python package if not already available.
%
%   Input:
%     dataDir - target directory (e.g. 'data/dataset')

    annotFile = fullfile(dataDir, 'annotations.json');

    % --- Check if already downloaded ---
    if isfile(annotFile)
        imgDir = fullfile(dataDir, 'images');
        if isfolder(imgDir)
            trainCount = numel(dir(fullfile(imgDir, 'train', '*.jpg')));
            testCount  = numel(dir(fullfile(imgDir, 'test',  '*.jpg')));
            fprintf('Dataset already exists: %d train + %d test images.\n', ...
                trainCount, testCount);
            if (trainCount + testCount) >= 12000
                fprintf('Skipping download.\n');
                return;
            else
                fprintf('Image count seems low. Re-downloading...\n');
            end
        end
    end

    % --- Ensure output directory exists ---
    if ~isfolder(dataDir)
        mkdir(dataDir);
    end

    % --- Install huggingface_hub if needed ---
    fprintf('Checking for huggingface-cli...\n');
    [status, ~] = system('huggingface-cli --help');
    if status ~= 0
        fprintf('Installing huggingface_hub Python package...\n');
        [status, output] = system('pip install -U huggingface_hub');
        if status ~= 0
            error('Failed to install huggingface_hub:\n%s\n\nPlease install manually: pip install huggingface_hub', output);
        end
        fprintf('huggingface_hub installed successfully.\n');
    end

    % --- Download the dataset ---
    fprintf('\n=== Downloading TickTockVQA dataset (3.7 GB) ===\n');
    fprintf('This may take 10-30 minutes depending on your connection.\n\n');

    cmd = sprintf('huggingface-cli download jaeha-choi/TickTockVQA --repo-type dataset --local-dir "%s"', ...
        dataDir);

    fprintf('Running: %s\n\n', cmd);
    [status, output] = system(cmd, '-echo');

    if status ~= 0
        error(['Download failed. Output:\n%s\n\n' ...
               'Alternative: install git-lfs and run:\n' ...
               '  git lfs install\n' ...
               '  git clone https://huggingface.co/datasets/jaeha-choi/TickTockVQA "%s"\n'], ...
               output, dataDir);
    end

    % --- Validate download ---
    if ~isfile(annotFile)
        error('Download completed but annotations.json not found in %s', dataDir);
    end

    trainCount = numel(dir(fullfile(dataDir, 'images', 'train', '*.jpg')));
    testCount  = numel(dir(fullfile(dataDir, 'images', 'test',  '*.jpg')));
    fprintf('\n=== Download complete ===\n');
    fprintf('Train images: %d\n', trainCount);
    fprintf('Test images:  %d\n', testCount);
    fprintf('Total:        %d\n', trainCount + testCount);

    if (trainCount + testCount) < 12000
        warning('Expected ~12,483 images but found %d. Some files may be missing.', ...
            trainCount + testCount);
    end
end
