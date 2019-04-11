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
    J = -mean(log(sum(Y .* P{end},1)));
    
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