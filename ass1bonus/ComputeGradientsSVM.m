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
function [grad_W, grad_b] = ComputeGradientsSVM(X, Y, P, W, lambda)

    batch_size = double(size(X,2));

    %hinge derivative, (in zero == 1)
    H = zeros(size(P));
    B = sum(P .* Y, 1);
    B = ones(size(P,1),1)*B;    
    
    % where max(.,.) in not zero
    H((P - B + 1) > 0) = 1;
    Yidx = boolean(Y);
    H(Yidx) = 0;
    % gradient corresponding at the true class
    H(Yidx) = -1.0 * sum(H,1);
  
    
    grad_b = sum(H,2) /batch_size;
    
    grad_W = H * X';
    
    
    grad_W = grad_W/batch_size + 2 * lambda * W;
end