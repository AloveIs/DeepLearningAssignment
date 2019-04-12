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
m = [50,50];

lambda = 0.0001;

GDparams.n_batch = 100;
GDparams.eta = 0.001;
GDparams.n_epochs = 200;
GDparams.n_step = 500;
GDparams.n_cycles = 1;
% initialization
%[W , b] = initialize_params(K,m,d);
%P  = EvaluateClassifier(X_train(:,1:7),W,b);
%[grad_W_mine, grad_b_mine] = ComputeGradients(X_train(:,1:7), Y_train(:,1:7), P, W, b, lambda);
%% check gradient
if false
    [W , b] = initialize_params(K,m,d);
    P  = EvaluateClassifier(X_train(:,1:10),W,b);
    
    NetParams.W = W;
    NetParams.b = b;
    NetParams.use_bn = false;
    
    grads = ComputeGradsNumSlow(X_train(:,1:10), Y_train(:,1:10), NetParams, lambda, 0.00001);
    grad_b = grads.b;
    grad_W = grads.W;
    
    [grad_W_mine, grad_b_mine] = ComputeGradients(X_train(:,1:10), Y_train(:,1:10), P, W, b, lambda);
    
    for i = 1 : numel(grad_W_mine)
        fprintf("Max abs divergence is: \n W(%d) %e \nb(%d) %e \n\n",i,  ...
    max(max(abs(grad_W_mine{i}-grad_W{i}))),i, max(abs(grad_b_mine{i}-grad_b{i})));
        
    end

end

%% training and testing the model
if false
    m = [50,50];
    GDparams.n_step = 5 * 450;
    GDparams.n_cycles = 2;
    
    [X ,Y,y, X_val,Y_val,y_val ,X_test,Y_test, y_test] = use_all_data();
    [W , b] = initialize_params(K,m,d);
    [Wstar, bstar] = MiniBatchGD(X, Y, GDparams, W, b, lambda, X_val, Y_val);

    % classification using best parameters
    P = EvaluateClassifier(X_test, Wstar, bstar);
    [argvalue, argmax] = max(P{end});
    % compare with ground truth
    R = argmax == y_test;

    fprintf("Accuracy on test data is : %f",(sum(R))/size(Y_test,2)*100);

end


%% training and testing the 9-layer model
if false
    GDparams.n_step = 2 * 450;
    GDparams.n_cycles = 2;
    
    m = [50, 30, 20, 20, 10, 10, 10, 10];
    [X ,Y,y, X_val,Y_val,y_val ,X_test,Y_test, y_test] = use_all_data();
    [W , b] = initialize_params(K,m,d);
    [Wstar, bstar] = MiniBatchGD(X, Y, GDparams, W, b, lambda, X_val, Y_val);

    % classification using best parameters
    P = EvaluateClassifier(X_test, Wstar, bstar);
    [argvalue, argmax] = max(P{end});
    % compare with ground truth
    R = argmax == y_test;

    fprintf("Accuracy on test data is : %f",(sum(R))/size(Y_test,2)*100);

end

%% batch normalization
%% training and testing the model
if false
    m = [10,10,10];
    GDparams.n_step = 5 * 450;
    GDparams.n_cycles = 2;
    lambda = 1;
    
    %[X ,Y,y, X_val,Y_val,y_val ,X_test,Y_test, y_test] = use_all_data();
    NetParams = initialize_paramsBN(K,m,d);
    P  = EvaluateClassifierBN(X_train(:,1:10),NetParams);
    %C  = ComputeCostBN(X_train(:,1:40),Y_train(:,1:40),NetParams,1);
    
    grads = ComputeGradsNumSlow(X_train(:,1:10), Y_train(:,1:10), NetParams, lambda, 0.00001);
    grad_b = grads.b;
    grad_W = grads.W;
    
    [grad_W_mine, grad_b_mine,  grad_gammas_mine, grad_betas_mine] = ComputeGradientsBN(X_train(:,1:10), Y_train(:,1:10), P, NetParams, lambda);
    
    for i = 1 : numel(grad_W_mine)
        if i == numel(grad_W_mine)
                fprintf("Max abs divergence is: \n W(%d) %e \nb(%d) %e \n\n",i,  ...
        max(max(abs(grad_W_mine{i}-grad_W{i}))),i, ...
        max(abs(grad_b_mine{i}-grad_b{i})));
        else
        fprintf("Max abs divergence is: \n W(%d) %e \nb(%d) %e \ngamma(%d) %e \nbeta(%d) %e\n\n",i,  ...
        max(max(abs(grad_W_mine{i}-grad_W{i}))),i, ...
        max(abs(grad_b_mine{i}-grad_b{i})),...
        i,max(abs(grad_gammas_mine{i}-grads.gammas{i})),...
        i,max(abs(grad_betas_mine{i}-grads.betas{i}))...
        );
        end
    end
    
    %NetParams_star = MiniBatchGDBN(X, Y, GDparams, NetParams, lambda, X_val, Y_val);

    % classification using best parameters
    %P = EvaluateClassifierBN(X_test, NetParams_star);
    %[argvalue, argmax] = max(P{end,3});
    % compare with ground truth
    %R = argmax == y_test;

    %fprintf("Accuracy on test data is : %f",(sum(R))/size(Y_test,2)*100);

end

if false
    m = [50,50];
    GDparams.n_step = 5 * 450;
    GDparams.n_cycles = 2;
    
    [X ,Y,y, X_val,Y_val,y_val ,X_test,Y_test, y_test] = use_all_data();
    [W , b] = initialize_params(K,m,d);
    [Wstar, bstar] = MiniBatchGD(X, Y, GDparams, W, b, lambda, X_val, Y_val);

    % classification using best parameters
    P = EvaluateClassifier(X_test, Wstar, bstar);
    [argvalue, argmax] = max(P{end});
    % compare with ground truth
    R = argmax == y_test;

    fprintf("Accuracy on test data is : %f",(sum(R))/size(Y_test,2)*100);

end


%% training and testing the 9-layer model
if false
    GDparams.n_step = 2 * 450;
    GDparams.n_cycles = 2;
    
    m = [50, 30, 20, 20, 10, 10, 10, 10];
    [X ,Y,y, X_val,Y_val,y_val ,X_test,Y_test, y_test] = use_all_data();
    [W , b] = initialize_params(K,m,d);
    [Wstar, bstar] = MiniBatchGD(X, Y, GDparams, W, b, lambda, X_val, Y_val);

    % classification using best parameters
    P = EvaluateClassifier(X_test, Wstar, bstar);
    [argvalue, argmax] = max(P{end});
    % compare with ground truth
    R = argmax == y_test;

    fprintf("Accuracy on test data is : %f",(sum(R))/size(Y_test,2)*100);

end

%% batch normalization
%% training and testing the model
if false
    m = [10,10,10];
    GDparams.n_step = 5 * 450;
    GDparams.n_cycles = 2;
    lambda = 1;
    
    %[X ,Y,y, X_val,Y_val,y_val ,X_test,Y_test, y_test] = use_all_data();
    NetParams = initialize_paramsBN(K,m,d);
    P  = EvaluateClassifierBN(X_train(:,1:10),NetParams);
    %C  = ComputeCostBN(X_train(:,1:40),Y_train(:,1:40),NetParams,1);
    
    grads = ComputeGradsNumSlow(X_train(:,1:10), Y_train(:,1:10), NetParams, lambda, 0.00001);
    grad_b = grads.b;
    grad_W = grads.W;
    
    [grad_W_mine, grad_b_mine,  grad_gammas_mine, grad_betas_mine] = ComputeGradientsBN(X_train(:,1:10), Y_train(:,1:10), P, NetParams, lambda);
    
    for i = 1 : numel(grad_W_mine)
        if i == numel(grad_W_mine)
                fprintf("Max abs divergence is: \n W(%d) %e \nb(%d) %e \n\n",i,  ...
        max(max(abs(grad_W_mine{i}-grad_W{i}))),i, ...
        max(abs(grad_b_mine{i}-grad_b{i})));
        else
        fprintf("Max abs divergence is: \n W(%d) %e \nb(%d) %e \ngamma(%d) %e \nbeta(%d) %e\n\n",i,  ...
        max(max(abs(grad_W_mine{i}-grad_W{i}))),i, ...
        max(abs(grad_b_mine{i}-grad_b{i})),...
        i,max(abs(grad_gammas_mine{i}-grads.gammas{i})),...
        i,max(abs(grad_betas_mine{i}-grads.betas{i}))...
        );
        end
    end
    
    %NetParams_star = MiniBatchGDBN(X, Y, GDparams, NetParams, lambda, X_val, Y_val);

    % classification using best parameters
    %P = EvaluateClassifierBN(X_test, NetParams_star);
    %[argvalue, argmax] = max(P{end,3});
    % compare with ground truth
    %R = argmax == y_test;

    %fprintf("Accuracy on test data is : %f",(sum(R))/size(Y_test,2)*100);

end

if true
    
    m = [50, 30, 20, 20, 10, 10, 10, 10];
    GDparams.n_step = 2 * 450;
    GDparams.n_cycles = 5;
    lambda = 0.0001;
    
    [X ,Y,y, X_val,Y_val,y_val ,X_test,Y_test, y_test] = use_all_data();
    NetParams = initialize_paramsBN(K,m,d);

    
    NetParams_star = MiniBatchGDBN(X, Y, GDparams, NetParams, lambda, X_val, Y_val);

    % classification using best parameters
    P = EvaluateClassifierBN(X_test, NetParams_star);
    [argvalue, argmax] = max(P{end,3});
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

if false
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

end

%% initialize_params
%
% Initialize the values for W and b
%
function [W , b] = initialize_params(K,m,d)
    
    W = {};
    b = {};
    
    i = 1;
    while i <= size(m,2)
        %input check
        if i ==1
           W{i} = 1.0/sqrt(d) * randn(m(i),d);
           b{i} = 1.0/sqrt(d) * randn(m(i),1);
        else
           W{i} = 1.0/sqrt(m(i-1)) * randn(m(i),m(i-1)); 
           b{i} = 1.0/sqrt(m(i-1)) * randn(m(i),1);
        end
        i = i + 1;
    end
    W{i} = 1.0/sqrt(m(end)) * randn(K,m(end));
    b{i} = 1.0/sqrt(m(end)) * randn(K,1);
end
%% initialize_params
%
% Initialize the values for W and b and all the batch normalization
% parameters
%
function NetParams = initialize_paramsBN(K,m,d)
    
    NetParams.use_bn = true;
    
    NetParams.W = {};
    NetParams.b = {};
    NetParams.gammas = {};
    NetParams.betas = {};
    
    i = 1;
    while i <= size(m,2)
        %input check
        if i ==1
           NetParams.W{i} = 1.0/sqrt(d) * randn(m(i),d);
           NetParams.b{i} = 1.0/sqrt(d) * randn(m(i),1);
        else
           NetParams.W{i} = 1.0/sqrt(m(i-1)) * randn(m(i),m(i-1)); 
           NetParams.b{i} = 1.0/sqrt(m(i-1)) * randn(m(i),1);
        end
        NetParams.gammas{i} = randn(m(i),1); 
        NetParams.betas{i} = randn(m(i),1);
        i = i + 1;
    end
    NetParams.W{i} = 1.0/sqrt(m(end)) * randn(K,m(end));
    NetParams.b{i} = 1.0/sqrt(m(end)) * randn(K,1);
   
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


