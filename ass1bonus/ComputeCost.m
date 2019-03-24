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

    P = EvaluateClassifier(X, W, b);
    
    % compute the hinge loss function
    
    %compute the max
    J = max(0, P - (P .* Y)+ 1);
    %P(:,1:6)
    %Y(:,1:6)
    %J(:,1:6)
    
    J = 1.0 /double(size(X,2))  * sum(sum(J));
    
    J =  J + lambda * sum(sum(W .* W,'double'),'double');
end