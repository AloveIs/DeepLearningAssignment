%% ComputeGradients
%
% • each column of X corresponds to an image and it has size d×n.
% • each column of Y (K×n) is the one-hot ground truth label for the cor-
% responding column of X.
% • each column of P contains the probability for each label for the image
% in the corresponding column of X. P has size K×n.
% • grad W is the gradient matrix of the cost J relative to W and has size
% K×d.
% 6• grad b is the gradient vector of the cost J relative to b and has size
% K×1.
%
%
function [grad_W, grad_b] = ComputeGradients(X, Y, P, W, lambda)
    % compute useful constant
    batch_size = double(size(X,2));

    %initialization
    grad_W = zeros(size(W));
    grad_b = zeros(size(Y,1), 1);
    
    % compute g as defined on the slides
    g = -(Y-P);
    
    % use g to compute the 2 gradients
    grad_b = 1.0/ batch_size * sum(g,2);
    
    grad_W = g * X';
    % add the term about the regularization derivative
    grad_W = 1.0/ batch_size * grad_W + 2 * lambda * W;
end
