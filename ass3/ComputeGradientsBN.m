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
function [grad_W, grad_b] = ComputeGradients(X, Y, P, W, b, lambda)
    % compute useful constant
    batch_size = double(size(X,2));
    
    l = numel(W);
    grad_b = {};
    grad_W = {};
    % compute g as defined on the slides
    % for lest layer
    g = -(Y-P{end});
    
    %input to the layer
    H = P{l-1};

%     disp(size(g))
%     disp(size(H))
%     disp(size(W{l}))
%     grad_W{l} = 1.0/ batch_size * (g * H') + 2 * lambda * W{l};
%     grad_b{l} = 1.0/ batch_size * sum(g,2);
%     l = l - 1;
    
    while l > 1
        
        grad_b{l} = 1.0/ batch_size * sum(g,2);
        grad_W{l} = 1.0/ batch_size * (g * P{l-1}') + 2 * lambda * W{l};
        
        g = (g' * W{l})';
        H = P{l-1};
        g(H==0) = 0;
  
%         H = P{l};
%         disp(size(g))
%         disp(size(W{l}))
% 
% 
%         grad_b{l} = 1.0/ batch_size * sum(g,2);
%         grad_W{l} = 1.0/ batch_size * (g * P{l-1}') + 2 * lambda * W{l};
        l = l - 1;
    end
    
    grad_b{1} = 1.0/ batch_size * sum(g,2);
    grad_W{1} = 1.0/ batch_size * (g * X') + 2 * lambda * W{1};
    
%     H = P{1};
%     
%     grad_b2 = 1.0/ batch_size * sum(g,2);
%     grad_W2 = 1.0/ batch_size * (g * H') + 2 * lambda * W{2};
%     % use g to compute the 2 gradients
% 
%     g = (g' * W{2})';
%     g(H==0) = 0;
%     
%     
%     grad_b1 = 1.0/ batch_size * sum(g,2);
%     grad_W1 = 1.0/ batch_size * (g * X') + 2 * lambda * W{1};
%     
%     grad_b = {grad_b1, grad_b2};
%     grad_W = {grad_W1, grad_W2};
end
