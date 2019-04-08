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