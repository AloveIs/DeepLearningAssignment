

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Content of file : Assignment1.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% prepare the environment and constants
addpath('Datasets/cifar-10-batches-mat');

train_data = 'data_batch_1.mat';
val_data = 'data_batch_2.mat';
test_data = 'test_batch.mat';

% load data
[X_train, Y_train, y_train] = LoadBatch(train_data);
[X_val, Y_val, y_val] = LoadBatch(val_data);
[X_test, Y_test, y_test] = LoadBatch(test_data);

N = size(y_train,2);
K = size(Y_train,1);
d = size(X_train,1);


lambda = 1.0;

GDparams.n_batch = 100;
GDparams.eta = 0.01;
GDparams.n_epochs = 40;

% initialization
[W , b] = initialize_params(K,d);
% training
[Wstar, bstar] = MiniBatchGD(X_train, Y_train, GDparams, W, b, lambda, X_val, Y_val);


% test

% classification using best parameters
P = EvaluateClassifier(X_test, Wstar, bstar);
[argvalue, argmax] = max(P);
% compare with ground truth
R = argmax == y_test;

fprintf("Accuracy on test data is : %f",(sum(R))/size(Y_test,2)*100)

% show prototypes of the learnt W matrix
F = show_prototype(Wstar);


%% initialize_params
%
% Initialize the values for W and b
%
function [W , b] = initialize_params(K,d, std_dev)

    %input check
    if nargin < 3
        std_dev = 0.1;
    end

    W = std_dev * randn(K,d);
    b = std_dev * randn(K,1);
end



function s_im = show_prototype(W)
    figure();
    s_im = zeros(32, 32, 3, size(W,1));
    
    
    for i=1:size(W,1)
        im = reshape(W(i, :), 32, 32, 3);
        s_im(:,:,:,i) = (im - min(im(:))) / (max(im(:))- min(im(:)));
        s_im(:,:,:,i) = permute(s_im(:,:,:,i), [2, 1, 3]);
    end
    
    montage(s_im, 'Size', [3,4]);
    
end





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Content of file : LoadBatch.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Content of file : EvaluateClassifier.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% EvaluateClassifier
%
% Write a function that evaluates the network function, i.e. equations
% (1, 2), on multiple images and returns the results. 
% • each column of X corresponds to an image and it has size d×n.
% • W and b are the parameters of the network.
% • each column of P contains the probability for each label for the 
% image in the corresponding column of X. P has size K×n.
%
function P = EvaluateClassifier(X, W, b)
    % evaluate linear part
    s = W * X + b *  ones(1,size(X,2));
    
    % compute the softmax:
    % - numerators of the softmax:
    E = exp(s);
    % - denominators of the softmax:
    D = ones(size(W,1),1) * sum(E,1);
    
    % Divide each column by their sum
    % to have the softmax
    P = E./D;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Content of file : ComputeCost.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% ComputeCost
%
% Compute the cost function
%
% • each column of X corresponds to an image and X has size d×n.
% • each column of Y (K×n) is the one-hot ground truth label for the corre-
% sponding column of X or Y is the (1×n) vector of ground truth labels.
% • J is a scalar corresponding to the sum of the loss of the network’s
% predictions for the images in X relative to the ground truth labels and
% the regularization term on W.
function J = ComputeCost(X, Y, W, b, lambda)
    % get the evaluation of the current parameters for the batch
    P = EvaluateClassifier(X, W, b);
   
    %compute the cross-entropy part
    J = -mean(log(sum(Y .* P,1)));
   
    % add the regularizing term
    J =  J + lambda * sum(sum(W .* W,'double'),'double');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Content of file : MiniBatchGD.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% MiniBatchGD
%
% Perform the model update.
% where X contains all the training images, Y the labels for the training
% images, W, b are the initial values for the network’s parameters, lambda
% is the regularization factor in the cost function and GDparams is an object containing the parameter values n batch, eta
% and n epochs
%
function [Wstar, bstar] = MiniBatchGD(X, Y, GDparams, W, b, lambda, Xval, Yval)
    
    batch_size = int32(size(Y,2)/ GDparams.n_batch);
    
    % matrices to save cost and accuracy after each epoch
    C = zeros(GDparams.n_epochs,2);
    A = zeros(GDparams.n_epochs,2);
    
    
    for epoch = 1 : GDparams.n_epochs
        batch = 1;
        start_index  = 1;
        while start_index < size(X,2)
            
            if start_index >= size(X,2)
                break;
            end
            %get indexes of the batch data
            idx = start_index : min(start_index + batch_size -1, size(X,2));
            
            %update starting index
            start_index = start_index + batch_size;
            
            % index the actual data
            X_batch = X(:,idx);
            Y_batch = Y(:,idx);
            
            % update parameters
            P = EvaluateClassifier(X_batch, W, b);
            [grad_W, grad_b] = ComputeGradients(X_batch, Y_batch, P, W, lambda);
            W = W - GDparams.eta * grad_W;
            b = b - GDparams.eta * grad_b;
            batch = batch + 1;
        end
        
        
        % save the cost and accuracy after each epoch
        C(epoch,1) = ComputeCost(X, Y, W, b, lambda);
        C(epoch,2) = ComputeCost(Xval, Yval, W, b, lambda);
        
        A(epoch,1) = compute_accuracy(X, Y, W, b);
        A(epoch,2) = compute_accuracy(Xval, Yval, W, b);
    end
    
    
    % plot loss and accuracy of the network
    x = 1 : GDparams.n_epochs;
    plot(x, C(:,1),x, C(:,2));
    figure();
    plot(x, A(:,1),x, A(:,2));
    
    % set return values
    Wstar = W;
    bstar = b;
end



function acc = compute_accuracy(X,Y,W,b)

y = vec2ind(Y);

P = EvaluateClassifier(X, W, b);

[~, argmax] = max(P);

R = argmax == y;

acc = double(sum(R))/size(Y,2)*100;

end