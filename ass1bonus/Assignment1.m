% Assignment 1

% prepare the environment and constants
addpath('./../ass1/Datasets/cifar-10-batches-mat');

train_data = 'data_batch_1.mat';
val_data = 'data_batch_2.mat';
test_data = 'test_batch.mat';


[X_train, Y_train, y_train] = LoadBatch(train_data);
[X_val, Y_val, y_val] = LoadBatch(val_data);
[X_test, Y_test, y_test] = LoadBatch(test_data);

N = size(y_train,2);
K = size(Y_train,1);
d = size(X_train,1);

[W , b] = initialize_params(K,d,10.0);
% lambda 0.02 , batch 50 eta 0.05 epoch 40 , std_dev 0.01
% lambda 0.05 , batch 50 eta 0.05 epoch 80 , std_dev 5.0
lambda = 0.005;

GDparams.n_batch = 10;
GDparams.eta = 0.05;
GDparams.n_epochs = 100;


%% best so far
% lambda 0.0, batch 1000 eta 0.05 epoch 100 , std_dev 10.0
% lambda 0.001, batch 1000 eta 0.05 epoch 100 , std_dev 10.0 (cool images)
% lambda 0.0, batch 1000 eta 0.05 epoch 100 , std_dev 10.0
% lambda 0.01, batch 10000 eta 0.05 epoch 20 , std_dev 10.0 (cool images)
% lambda 0.001, batch 10000 eta 0.05 epoch 20 , std_dev 10.0 (cool images)
% lambda 0.0001, batch 10000 eta 0.05 epoch 80 , std_dev 10.0 (cool images)
% P = EvaluateClassifier(X_test(:,1:15), W, b);
% [dW, db] = ComputeGradients(X_test(:,1:15), Y_test(:,1:15), P, W, lambda);
% [grad_b, grad_W] = ComputeGradsNum(X_test(:,1:15), Y_test(:,1:15), W, b, lambda, 0.001);
% 
% disp("a");
% pause;
% disp("b");
[Wstar, bstar] = MiniBatchGD(X_train, Y_train, GDparams, W, b, lambda, X_val, Y_val);


% test

P = EvaluateClassifier(X_test, Wstar, bstar);

[argvalue, argmax] = max(P);

R = argmax == y_test;

fprintf("Accuracy on test data is : %f\n",(sum(R))/size(Y_test,2)*100)

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
    
    montage(s_im, 'Size', [1,10]);
    
end



