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

[W , b] = initialize_params(K,d,0.1);
% lambda 0.02 , batch 50 eta 0.05 epoch 40 , std_dev 0.01
% lambda 0.05 , batch 50 eta 0.05 epoch 80 , std_dev 5.0
lambda = 0.1;

GDparams.n_batch = 100;
GDparams.eta = 0.01;
GDparams.n_epochs = 40;

%% use all data
[X ,Y,y, X_val,Y_val,y_val ,X_test,Y_test, y_test] = use_all_data();
[Wstar, bstar] = MiniBatchGD(X, Y, GDparams, W, b, lambda, X_val, Y_val);
P = EvaluateClassifierSVM(X_test, Wstar, bstar);
[~, argmax] = max(P);
R = argmax == y_test;
fprintf("Accuracy on test data is : %f\n",(sum(R))/size(Y_test,2)*100)
show_prototype(Wstar);

fprintf("#########################\n");

pause;
close all;
%% Xavier Initialization

std_dev = 1.0/sqrt(double(size(X,1)));
Dev = std_dev*[0.001,0.01,0.1,0.5,1,5,10,20,50,100];
for i = 1:size(Dev,2)
    fprintf("Std dev : %f\n",std_dev);
    [W , b] = initialize_params(K,d,std_dev);
    [Wstar, bstar] = MiniBatchGD(X_train, Y_train, GDparams, W, b, lambda, X_val, Y_val);
    P = EvaluateClassifierSVM(X_test, Wstar, bstar);
    [~, argmax] = max(P);
    R = argmax == y_test;
    fprintf("Accuracy on test data is : %f\n",(sum(R))/size(Y_test,2)*100);
    fprintf("#########################\n");
    close all;
end
pause;

%% decaying the learning rate

T = zeros(30, 5);
L = logspace(-1,2,30);

for l = 1:30
    for j = 1:5
    fprintf("%d:%d\n",j,l);
    [W , b] = initialize_params(K,d,L(l));
    decay_rate = 0.1;
    GDparams.eta = 1.2;
    [Wstar, bstar] = MiniBatchGD(X_train, Y_train, GDparams, W, b, lambda, X_val, Y_val,decay_rate);
    P = EvaluateClassifierSVM(X_test, Wstar, bstar);
    [~, argmax] = max(P);
    R = argmax == y_test;
    T(l,j) = (sum(R))/size(Y_test,2)*100;
    fprintf("Accuracy on test data is : %f\n",T(l,j));
    show_prototype(Wstar);
    fprintf("#########################\n");
    close all;
    end
end

%% SVM

lambda = 0.1;

GDparams.n_batch = 100;
GDparams.eta = 0.01;
GDparams.n_epochs = 100;

L = [];

for r = 1:10

    [W , b] = initialize_params(K,d,0.5);
    [Wstar, bstar] = MiniBatchGDSVM(X_train, Y_train, GDparams, W, b, lambda, X_val, Y_val);


    % test

    P = EvaluateClassifierSVM(X_test, Wstar, bstar);

    [argvalue, argmax] = max(P);

    R = argmax == y_test;
    L(r) = (sum(R))/size(Y_test,2)*100;
    fprintf("Accuracy on test data is : %f\n",(sum(R))/size(Y_test,2)*100)

    F = show_prototype(Wstar);

end
close all;
% print mean and variance of the trials
disp(mean(L));
disp(sqrt(var(L)));

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

function [X ,Y,y, X_val,Y_val,y_val ,X_test,Y_test, y_test] = use_all_data()
    X = [];
    Y = [];
    y = [];
    for file = ["data_batch_1.mat","data_batch_3.mat","data_batch_4.mat", "data_batch_5.mat"]
        [X_t, Y_t, y_t] = LoadBatch(file);
        X = [X,X_t];
        Y = [Y,Y_t];
        y = [y,y_t];
    end
    [X_val, Y_val, y_val] = LoadBatch('data_batch_2.mat');
    [X_test, Y_test, y_test] = LoadBatch('test_batch.mat');
end


