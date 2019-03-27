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



