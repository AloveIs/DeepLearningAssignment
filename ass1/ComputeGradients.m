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

    batch_size = double(size(X,2));


    grad_W = zeros(size(W));
    grad_b = zeros(size(Y,1), 1);
    
    g = -(Y-P);
    
    grad_b = 1.0/ batch_size * sum(g,2);
    
    grad_W = g * X';
    
    
    grad_W = 1.0/ batch_size * grad_W + 2 * lambda * W;
end


function [grad_W, grad_b] = ComputeGradientsFull(X, Y, P, W, lambda)

    batch_size = size(X,2);


    grad_W = zeros(size(W));
    grad_b = zeros(no, 1);

    % forward
    
    
    
    
    
    % backward
    
    % $ -1 / y^T*p  * y^T $
    dl_dp = -1.0 * bsxfun(@rdivide, Y, (sum(Y .* P,1)));
    % should be of size K
    
    % derivative of the softmax
    
    dp_ds = zeros(size(P,1),size(P,2),batch_size); 
    
    for i = 1 : batch_size
       dp_ds = diag(P(:,i)) - P(:,i)*P(:,i)';
    end
    
    
    grad_W = grad_W + 2 * lambda * W;
end