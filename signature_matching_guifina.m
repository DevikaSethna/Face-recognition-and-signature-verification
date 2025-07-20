function signature_matching_gui()
    net = alexnet(); % Load pretrained CNN
    inputSize = net.Layers(1).InputSize(1:2);
    featureLayer = 'fc7'; % Use feature layer for embeddings

    % Create GUI
    fig = uifigure('Name', 'Signature Matching (CNN-Based)', 'Position', [100 100 700 450]);

    uilabel(fig, 'Position', [50 380 120 30], 'Text', 'Input Signature:');
    txt1 = uieditfield(fig, 'text', 'Position', [170 380 350 30]);
    uibutton(fig, 'push', 'Text', 'Browse', 'Position', [540 380 100 30], ...
        'ButtonPushedFcn', @(btn, event) browse_image(txt1));

    uilabel(fig, 'Position', [50 330 120 30], 'Text', 'Dataset Folder:');
    txt2 = uieditfield(fig, 'text', 'Position', [170 330 350 30]);
    uibutton(fig, 'push', 'Text', 'Select Folder', 'Position', [540 330 100 30], ...
        'ButtonPushedFcn', @(btn, event) browse_folder(txt2));

    uibutton(fig, 'push', 'Text', 'Compare Signature', ...
        'Position', [250 260 200 50], ...
        'ButtonPushedFcn', @(btn, event) compare_with_dataset(txt1.Value, txt2.Value));

    lblResult = uilabel(fig, 'Position', [50 200 600 30], ...
        'HorizontalAlignment', 'center', 'FontSize', 14, 'FontWeight', 'bold', 'Text', '');

    function browse_image(editField)
        [file, path] = uigetfile({'*.*', 'All Image Files'; ...
                                  '*.jpg;*.jpeg', 'JPEG Files'; ...
                                  '*.png', 'PNG Files'; ...
                                  '*.bmp', 'BMP Files'; ...
                                  '*.tif;*.tiff', 'TIFF Files'; ...
                                  '*.gif', 'GIF Files'});
        if file
            editField.Value = fullfile(path, file);
        end
    end

    function browse_folder(editField)
        folder = uigetdir();
        if folder ~= 0
            editField.Value = folder;
        end
    end

    function feat = extract_cnn_features(img)
        if size(img, 3) == 1
            img = repmat(img, 1, 1, 3); % Convert grayscale to RGB
        end
        img = imresize(img, inputSize); % Resize to CNN input
        act = activations(net, img, featureLayer, 'OutputAs', 'rows');
        feat = double(act(:)); % Convert to vector
    end

    function compare_with_dataset(input_img_path, dataset_folder)
        if isempty(input_img_path) || isempty(dataset_folder) || ...
           ~isfile(input_img_path) || ~isfolder(dataset_folder)
            lblResult.Text = 'Please select valid input image and dataset folder!';
            lblResult.FontColor = [1, 0, 0];
            return;
        end

        input_img = imread(input_img_path);
        feat1 = extract_cnn_features(input_img);

        % Supported extensions
        exts = {'*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff', '*.gif'};
        files = [];
        for i = 1:length(exts)
            files = [files; dir(fullfile(dataset_folder, exts{i}))]; %#ok<AGROW>
        end

        if isempty(files)
            lblResult.Text = 'No images found in dataset folder.';
            lblResult.FontColor = [1, 0, 0];
            return;
        end

        best_score = inf;
        best_match = '';
        best_img = [];

        for k = 1:length(files)
            dataset_img = imread(fullfile(files(k).folder, files(k).name));
            feat2 = extract_cnn_features(dataset_img);

            % MSE for similarity
            min_len = min(length(feat1), length(feat2));
            mse = mean((feat1(1:min_len) - feat2(1:min_len)).^2);

            if mse < best_score
                best_score = mse;
                best_match = files(k).name;
                best_img = dataset_img;
            end
        end

        similarity = max(0, 100 - best_score * 1000); % You can tune this

        if similarity >= 92
            lblResult.Text = sprintf('\x2714 Match Found: %s (%.2f%% Similarity)', best_match, similarity);
            lblResult.FontColor = [0, 0.6, 0];
        else
            lblResult.Text = sprintf('\x2716 No Strong Match Found. Closest: %s (%.2f%% Similarity)', best_match, similarity);
            lblResult.FontColor = [1, 0.4, 0];
        end

        % Show match
        figure('Name', 'Comparison Result');
        subplot(1,2,1); imshow(input_img); title('Input Signature');
        subplot(1,2,2); imshow(best_img); title(['Best Match: ', best_match]);
    end
end
