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
    
    layers = numel(W);
    z = {};
    
    for l = 1 : layers
        if l == 1
            s = W{l} * X + b{l} *  ones(1,size(X,2));
        else
            s = W{l} * z{l-1} + b{l} *  ones(1,size(z{l-1},2));
        end
        if l ~= layers
            z{l} = max(0,s);
        end
    end
    
    % compute the softmax:
    % - numerators of the softmax:
    E = exp(s);
    % - denominators of the softmax:
    D = ones(size(E,1),1) * sum(E,1);
    z{layers} = E./D;
    % Divide each column by their sum
    % to have the softmax
    P = z;
end