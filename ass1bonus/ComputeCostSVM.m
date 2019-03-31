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
end