-e 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Content of file : Assignment1Bonus.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


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


-e 

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
    
    
    %% First do the labels
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
    
    %% Process Images
    X = double(Batch.data)'/255.0;
    
end-e 

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
end-e 

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
end-e 

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
function [Wstar, bstar] = MiniBatchGD(X, Y, GDparams, W, b, lambda, Xval, Yval, decay_rate)
    
    if nargin < 9
        decay_rate = 1.0;
    end
    
    Gradients = [];
    
    batch_size = int32(GDparams.n_batch);
    eta = GDparams.eta;
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
            %save gradients computed
            Gradients = [Gradients,grad_W];
            W = W - eta * grad_W;
            b = b - eta * grad_b;
            batch = batch + 1;
        end
        %update eta
        eta = decay_rate * eta;
        
        % save the cost and accuracy after each epoch
        C(epoch,1) = ComputeCost(X, Y, W, b, lambda);
        C(epoch,2) = ComputeCost(Xval, Yval, W, b, lambda);
        
        A(epoch,1) = compute_accuracy(X, Y, W, b);
        A(epoch,2) = compute_accuracy(Xval, Yval, W, b);
                
    end
    
    % print informations on the gradient statistics
    fprintf("mean : %f, std_dev : %f\n", mean(Gradients(:)), sqrt(var(Gradients(:))));
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

end-e 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Content of file : ComputeCostSVM.m
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
function J = ComputeCostSVM(X, Y, W, b, lambda)

    P = EvaluateClassifierSVM(X, W, b);
    
    % compute the hinge loss function
    
    %find the true class value and broadcast it into a matrix
    B = sum(P .* Y, 1);
    B = ones(size(P,1),1)*B;
    
    J = P - B + 1;
    %max function
    J(J < 0) = 0;


    J(boolean(Y)) = 0;
    J = 1.0 /double(size(X,2))  * sum(sum(J,2));
    
    J =  J + lambda * sum(sum(W .* W,'double'),'double');
end-e 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Content of file : EvaluateClassifierSVM.m
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
function P = EvaluateClassifierSVM(X, W, b)
    % evaluate linear part
    B = b *  ones(1,size(X,2));
    P = W * X + B;
end-e 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Content of file : MiniBatchGDSVM.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [Wstar, bstar] = MiniBatchGDSVM(X, Y, GDparams, W, b, lambda, Xval, Yval)
    
    batch_size = int32(GDparams.n_batch);
    C = zeros(GDparams.n_epochs,2);
    A = zeros(GDparams.n_epochs,2);
    
    disp(ComputeCostSVM(X, Y, W, b, lambda));
    disp(ComputeCostSVM(Xval, Yval, W, b, lambda));
    
    for epoch = 1 : GDparams.n_epochs
        batch = 1;
        start_index  = 1;
        while start_index < size(X,2)         
            if start_index >= size(X,2)
                break;
            end
            
            idx = start_index : min(start_index + batch_size -1, size(X,2));
            %fprintf("%d\t%d\n",idx(1), idx(end));
            
            %update starting
            start_index = start_index + batch_size;
            
            X_batch = X(:,idx);
            Y_batch = Y(:,idx);
            
            P = EvaluateClassifierSVM(X_batch, W, b);
            [grad_W, grad_b] = ComputeGradientsSVM(X_batch, Y_batch, P, W, lambda);
            W = W - GDparams.eta * grad_W;
            b = b - GDparams.eta * grad_b;
            batch = batch + 1;
        end
        
        C(epoch,1) = ComputeCostSVM(X, Y, W, b, lambda);
        C(epoch,2) = ComputeCostSVM(Xval, Yval, W, b, lambda);
        fprintf("Epoch : %d\ttest: %f \tval: %f\n", epoch,C(epoch,1),C(epoch,2));
        
        A(epoch,1) = compute_accuracy(X, Y, W, b);
        A(epoch,2) = compute_accuracy(Xval, Yval, W, b);
    end
    
    
    
    x = 1 : GDparams.n_epochs;
    figure();
    plot(x, C(:,1),x, C(:,2));
    figure();
    plot(x, A(:,1),x, A(:,2));
    Wstar = W;
    bstar = b;
end



function acc = compute_accuracy(X,Y,W,b)

y = vec2ind(Y);

P = EvaluateClassifierSVM(X, W, b);

[~, argmax] = max(P);

R = argmax == y;

acc = double(sum(R))/size(Y,2)*100;

end

