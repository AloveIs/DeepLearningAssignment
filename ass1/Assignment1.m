% Assignment 1

% prepare the environment and constants
addpath('Datasets/cifar-10-batches-mat');

train_data = 'data_batch_1.mat';
val_data = 'data_batch_2.mat';
test_data = 'test_batch.mat';


[X_train, Y_train, y_train] = LoadBatch(train_data);
[X_val, Y_val, y_val] = LoadBatch(val_data);
[X_test, Y_test, y_test] = LoadBatch(test_data);

N = size(y_train,2);
K = size(Y_train,1);
d = size(X_train,1);

[W , b] = initialize_params(K,d);

lambda = 0.0;

GDparams.n_batch = 10;
GDparams.eta = 0.01;
GDparams.n_epochs = 300;

[Wstar, bstar] = MiniBatchGD(X_train, Y_train, GDparams, W, b, lambda, X_val, Y_val);


% test


count = 0;


P = EvaluateClassifier(X_test, Wstar, bstar);

[argvalue, argmax] = max(P);

R = argmax == y_test;

fprintf("Accuracy on test data is : %f",(sum(R))/size(Y_test,2)*100)




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



