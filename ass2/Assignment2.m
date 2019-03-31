% prepare the environment and constants
addpath('../ass1/Datasets/cifar-10-batches-mat');

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
%number of hidden nodes
m = 50;

lambda = 0.01;

GDparams.n_batch = 100;
GDparams.eta = 0.01;
GDparams.n_epochs = 40;

% initialization
[W , b] = initialize_params(K,m,d);
% 
% P  = EvaluateClassifier(X_train,W,b);
% J = ComputeCost(X_train, Y_train, W, b, lambda);

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

%% initialize_params
%
% Initialize the values for W and b
%
function [W , b] = initialize_params(K,m,d)

    %input check

    W1 = 1.0/sqrt(d) * randn(m,d);
    W2 = 1.0/sqrt(m) * randn(K,m);
    W = {W1,W2};
    b1 = 1.0/sqrt(d) * randn(m,1);
    b2 = 1.0/sqrt(m) * randn(K,1);
    b = {b1,b2};
end


