

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Content of file : Assignment2.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


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

GDparams.n_step = 900;
GDparams.n_cycles = 2;


%load all data
[X ,Y,y, X_val,Y_val,y_val ,X_test,Y_test, y_test] = use_all_data();

load('lambda_res.mat','L');
l_min = -5;
l_max = 1;

for i = 9:50
    tic;
    l = l_min + (l_max - l_min)*rand(1, 1);
    
    l = 10^l;
    fprintf("##### %d ) lambda = %f",i,l);
    [W , b] = initialize_params(K,m,d);
    [Wstar, bstar] = MiniBatchGD(X_train, Y_train, GDparams, W, b, l, X_val, Y_val);
    close all;
    P = EvaluateClassifier(X_val, Wstar, bstar);
    [argvalue, argmax] = max(P{2});
    R = argmax == y_val;

    L(i,:) = [l, (sum(R))/size(Y_val,2)*100];
    fprintf("%d) Accuracy on test data is : %f",i,L(i,2));
    save('lambda_res.mat','L');
    toc;
end

save('lambda_res.mat','L');


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
    X = double(Batch.data)';
    
    mean_X = mean(X, 2);
    std_X = std(X, 0, 2);
    
    X = X - repmat(mean_X, [1, size(X, 2)]);
    X = X ./ repmat(std_X, [1, size(X, 2)]);
    
    
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
    s1 = W{1} * X + b{1} *  ones(1,size(X,2));
    
    Z = max(0,s1);
    
    s2 = W{2} * Z + b{2} *  ones(1,size(X,2));
    
    % compute the softmax:
    % - numerators of the softmax:
    E = exp(s2);
    % - denominators of the softmax:
    D = ones(size(W,1),1) * sum(E,1);
    
    % Divide each column by their sum
    % to have the softmax
    P = { Z ,E./D};
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
    J = -mean(log(sum(Y .* P{2},1)));
    
    J2 = compute_regularization(W,lambda);
    
    % add the regularizing term
    J =  J + lambda*J2;
end



function J2 = compute_regularization(W, lambda)
    J2 = 0;

    if nargin < 2
        lambda = 1;
    end
    
    if lambda == 0
        return;
    end
    
    for k=1:length(W)
        Wi = W{k};
        J2 = J2 + sum(sum(Wi .* Wi,'double'),'double');
    end


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
    
    batch_size = int32(GDparams.n_batch);
    
    eta_min = 1e-5;
    eta_max = 1e-1;
    n_s = GDparams.n_step;
    cycles= GDparams.n_cycles;
    
    Rounds=2*(cycles*n_s);
    
    % matrices to save cost and accuracy after each epoch
    C = zeros(Rounds,4);
    A = zeros(Rounds,2);
    etas = zeros(Rounds,1);
    
    eta = eta_min;
    eta_step = (eta_max-eta_min)/(n_s);
    start_index  = 1;
    sign = 1;
    
    for rounds = 1 : Rounds
        if mod(rounds,100)==0
            fprintf("Round %d of %d : %.2f %%\n",rounds,Rounds,100*rounds/Rounds);
        end
        if start_index >= size(X,2)
            start_index = 1;
        end
        %get indexes of the batch data
        idx = start_index : min(start_index + batch_size -1, size(X,2));

        % index the actual data
        X_batch = X(:,idx);
        Y_batch = Y(:,idx);

        %update starting index
        start_index = start_index + batch_size;

        P = EvaluateClassifier(X_batch, W, b);
        [grad_W, grad_b] = ComputeGradients(X_batch, Y_batch, P, W,b, lambda);

        W{1} = W{1} - eta * grad_W{1};
        b{1} = b{1} - eta * grad_b{1};
        W{2} = W{2} - eta * grad_W{2};
        b{2} = b{2} - eta * grad_b{2};
              
        %save statistics
        C(rounds,1) = ComputeCost(X, Y, W, b, lambda);
        C(rounds,2) = ComputeCost(Xval, Yval, W, b, lambda);
        C(rounds,3) = ComputeCost(X, Y, W, b, 0);
        C(rounds,4) = ComputeCost(Xval, Yval, W, b, 0);
        
        A(rounds,1) = compute_accuracy(X, Y, W, b);
        A(rounds,2) = compute_accuracy(Xval, Yval, W, b);
        etas(rounds) = eta;
        
        %update eta
        eta = eta + sign*eta_step;
        if(eta >= (eta_max-1e-9))
            eta = eta_max;
            sign = -sign;
        elseif (eta <= (eta_min+1e-9))
            eta = eta_min;
            sign = -sign;
        end
    end
    
    % plot loss and accuracy of the network
    x = 1 : 100 : Rounds;
    plot(x, C(x,1),x, C(x,2));
    figure();
    plot(x, C(x,3),x, C(x,4));
    figure();
    plot(x, A(x,1),x, A(x,2));
    %figure();
    %plot(x-1, etas);
    save("test_fig3","C","A");
    % set return values
    Wstar = W;
    bstar = b;
end



function acc = compute_accuracy(X,Y,W,b)

y = vec2ind(Y);

P = EvaluateClassifier(X, W, b);

[~, argmax] = max(P{2});

R = argmax == y;

acc = double(sum(R))/size(Y,2)*100;

end