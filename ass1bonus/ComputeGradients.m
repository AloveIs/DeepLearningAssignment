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

    %hinge derivative, (in zero == 1)
    H = zeros(size(P));
    
    H((P - (P .* Y)+ 1) > 0) = 1;
    Yidx = boolean(Y);
    H(Yidx) = 0;
    %H(Yidx) = 1.0 * sum(H,1);
    %-1.0 * (size(Y,1)-1);
  
    
    grad_b = 1.0 /double(size(X,2)) * sum(H,2);
    
    grad_W = H * X';
    
    
    grad_W = 1.0 /double(size(X,2)) * grad_W + 2 * lambda * W;
end