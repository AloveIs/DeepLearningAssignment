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
GDparams.eta = 0.001;
GDparams.n_epochs = 200;
GDparams.n_step = 500;
GDparams.n_cycles = 1;
% initialization
[W , b] = initialize_params(K,m,d);


%% check gradient
if false
    P  = EvaluateClassifier(X_train(:,1:10),W,b);
    [grad_b, grad_W] = ComputeGradsNum(X_train(:,1:10), Y_train(:,1:10), W, b, lambda, 0.00001);
    [grad_W_mine, grad_b_mine] = ComputeGradients(X_train(:,1:10), Y_train(:,1:10), P, W, b, lambda);
    fprintf("Max abs divergence is: \n W1 %e W2 %e\nb1 %e b2 %e\n\n",  ...
    max(max(abs(grad_W_mine{1}-grad_W{1}))), max(max(abs(grad_W_mine{1}-grad_W{1}))), ...
        max(abs(grad_b_mine{1}-grad_b{1})), max(abs(grad_b_mine{2}-grad_b{2})));
end

%% training and testing the model
if false

    [Wstar, bstar] = MiniBatchGD(X_train, Y_train, GDparams, W, b, lambda, X_val, Y_val);

    % classification using best parameters
    P = EvaluateClassifier(X_test, Wstar, bstar);
    [argvalue, argmax] = max(P{2});
    % compare with ground truth
    R = argmax == y_test;

    fprintf("Accuracy on test data is : %f",(sum(R))/size(Y_test,2)*100);

end

%% serach lambda
if false
    GDparams.n_step = 900;
    GDparams.n_cycles = 2;


    %load all data
    [X ,Y,y, X_val,Y_val,y_val ,X_test,Y_test, y_test] = use_all_data();

    load('lambda_coarse.mat','L');
    l_min = -8;
    l_max = -1;

    
    %l_min = -3.3;
    %l_max = -2.4;

    %l_min = -1.6;
    %l_max = -1.0;


    for i=1:100
        tic;
        l = l_min + (l_max - l_min)*rand(1, 1);


        fprintf("##### %d )l = %f lambda = %f",i,l,10^l);
        l = 10^l;
        [W , b] = initialize_params(K,m,d);
        [Wstar, bstar] = MiniBatchGD(X, Y, GDparams, W, b, l, X_val, Y_val);
        close all;
        P = EvaluateClassifier(X_val, Wstar, bstar);
        [argvalue, argmax] = max(P{2});
        R = argmax == y_val;

        L(i,:) = [l, (sum(R))/size(Y_val,2)*100];
        fprintf("%d) Accuracy on test data is : %f",i,L(i,2));
        save('lambda_coarse.mat','L');
        toc;
    end

    save('lambda_coarse.mat','L');
    semilogx(L(:,1),L(:,2),'LineStyle','None','Marker','.')

end

if false
    GDparams.n_step = 900;
    GDparams.n_cycles = 2;


    %load all data
    [X ,Y,y, X_val,Y_val,y_val ,X_test,Y_test, y_test] = use_all_data();

    load('lambda_fine.mat','L');
    %l_min = -3.3;
    %l_max = -2.4;

    l_min = -4.0;
    l_max = -2.0;


    for i=23:60
        tic;
        l = l_min + (l_max - l_min)*rand(1, 1);


        fprintf("##### %d )l = %f lambda = %f",i,l,10^l);
        l = 10^l;
        [W , b] = initialize_params(K,m,d);
        [Wstar, bstar] = MiniBatchGD(X, Y, GDparams, W, b, l, X_val, Y_val);
        close all;
        P = EvaluateClassifier(X_val, Wstar, bstar);
        [argvalue, argmax] = max(P{2});
        R = argmax == y_val;

        L(i,:) = [l, (sum(R))/size(Y_val,2)*100];
        fprintf("%d) Accuracy on test data is : %f",i,L(i,2));
        save('lambda_fine.mat','L');
        toc;
    end

    save('lambda_fine.mat','L');
    semilogx(L(:,1),L(:,2),'LineStyle','None','Marker','.','MarkerSize',5)
    
    [V,I] = maxk(L(:,2),3);
    
    disp(L(I,:));
    
end


%% best model training


best_lambda = 3.052278022492892e-04;
GDparams.n_step = 900;
GDparams.n_cycles = 3;


%load all data
[X ,Y,y, X_val,Y_val,y_val ,X_test,Y_test, y_test] = use_all_data_2();

disp("hey I'm here")
[W , b] = initialize_params(K,m,d);
[Wstar, bstar] = MiniBatchGD(X, Y, GDparams, W, b, best_lambda, X_val, Y_val);
P = EvaluateClassifier(X_test, Wstar, bstar);
[argvalue, argmax] = max(P{2});
R = argmax == y_test;

fprintf("Accuracy on test data is : %f",(sum(R))/size(Y_test,2)*100);
disp("I'm done")



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

%% load all data
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
    X = [X,X_val(:,1:5000)];
    Y = [Y,Y_val(:,1:5000)];
    y = [y,y_val(1:5000)];
    
    X_val = X_val(:,5001:end);
    Y_val = Y_val(:,5001:end);
    y_val = y_val(5001:end);
    
    
end


function [X ,Y,y, X_val,Y_val,y_val ,X_test,Y_test, y_test] = use_all_data_2()
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
    X = [X,X_val(:,1:9000)];
    Y = [Y,Y_val(:,1:9000)];
    y = [y,y_val(1:9000)];
    
    X_val = X_val(:,9001:end);
    Y_val = Y_val(:,9001:end);
    y_val = y_val(9001:end);
    
    
end


