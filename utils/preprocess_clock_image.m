function img = preprocess_clock_image(filename, targetSize)
%PREPROCESS_CLOCK_IMAGE Read and preprocess a clock image for the network.
%   img = preprocess_clock_image(filename, targetSize)
%
%   Reads the image, resizes to targetSize, converts grayscale to RGB,
%   and returns as single precision (0-255 range for ImageNet normalization).
%
%   Inputs:
%     filename   - path to the image file
%     targetSize - [height width], e.g. [224 224]
%
%   Output:
%     img - single precision [H x W x 3] image

    img = imread(filename);

    % Handle grayscale -> RGB
    if size(img, 3) == 1
        img = repmat(img, [1 1 3]);
    end

    % Handle RGBA -> RGB
    if size(img, 3) == 4
        img = img(:,:,1:3);
    end

    % Resize to target dimensions
    img = imresize(img, targetSize(1:2));

    % Convert to single precision (ImageNet models expect 0-255 range)
    img = single(img);
end
