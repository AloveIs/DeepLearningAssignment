%% LoadBatch
%
% Load a batch and perform preprocessing
%
% • X contains the image pixel data, has size d×N, is of type double or
% single and has entries between 0 and 1. N is the number of images
% (10000) and d the dimensionality of each image (3072=32×32×3).
% • Y is K×N (K= # of labels = 10) and contains the one-hot representation
% of the label for each image.
% • y is a vector of length N containing the label for each image. A note
% of caution. CIFAR-10 encodes the labels as integers between 0-9 but
% Matlab indexes matrices and vectors starting at 1. Therefore it may be
% easier to encode the labels between 1-10.
%
function [X, Y, y] = LoadBatch(filename, debug)

    %input check
    if nargin < 2
        debug = false;
    end

    % load batch
    Batch = load(filename);
    
    if debug
        fprintf("The batch has the following fields:\n");
        disp(Batch);
    end
    
    
    %% First extract the labels
    % Number of classes
    K = 10;
    
    y = double(Batch.labels' + 1);
    
    % Number of samples
    N = size(y,2);
    
    Y = zeros(K,N);
    % find indexes to set to one and set them              %Y = ind2vec(y)
    idx = sub2ind(size(Y), y, [1:N]);
    Y(idx) = 1;
    
    if debug
        fprintf("First 20 labels are:\n");
        disp(y(1:20));
        fprintf("In one-hot encoding is:\n");
        disp(Y(:,1:20));
    end
    
    %% Normalize data
    X = double(Batch.data)'/255.0;
    
end