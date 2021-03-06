%% EvaluateClassifier
%
% Write a function that evaluates the network function, i.e. equations
% (1, 2), on multiple images and returns the results. 
% • each column of X corresponds to an image and it has size d×n.
% • W and b are the parameters of the network.
% • each column of P contains the probability for each label for the 
% image in the corresponding column of X. P has size K×n.
%
function P = EvaluateClassifierBN(X, NetParams, mu_MA, v_MA)


    if nargin < 3
       P = no_MA(X, NetParams);
    else   
       P = yes_MA(X, NetParams, mu_MA, v_MA);
    end
    
    
    
end



function P = no_MA(X, NetParams)
    % evaluate linear part
    eps = 1e-9;
    layers = numel(NetParams.W);
    z = {};
    
    for l = 1 : layers
        if l == 1
            s = NetParams.W{l} * X + NetParams.b{l} *  ones(1,size(X,2));
        else
            s = NetParams.W{l} * z{l-1,3} + NetParams.b{l} *  ones(1,size(z{l-1,3},2));
        end
        
        if l ~= layers
            z{l,1} = s;
            % batch normalize
            z{l,4} = mean(s,2);
            s = s - z{l,4} * ones(1,size(s,2));
            
            z{l,5} = (sum((s.^2),2))./size(s,2) + eps;
            s = s ./ (sqrt(z{l,5}* ones(1,size(s,2))));
            z{l,2} = s;
            % apply gamma and beta
                        
            s = (NetParams.gammas{l}* ones(1,size(s,2))) .* s;
            s = s + NetParams.betas{l} *  ones(1,size(s,2));
            
            z{l,6} = s;
            
            %apply relu
            z{l,3} = max(0,s);
        end
    end
    
    %last layer
    z{layers,1} = s;
    % compute the softmax:
    % - numerators of the softmax:
    E = exp(s);
    % - denominators of the softmax:
    D = ones(size(E,1),1) * sum(E,1);
    
    z{layers,3} = E./D;
    % Divide each column by their sum
    % to have the softmax
    P = z;
end

%
% (l,1) -> Wx+b              s
% (l,2) -> batch_norm(Wx+b)  s_hat
% (l,3) -> ReLU(...)         X
% (l,4) -> means
% (l,5) -> variances
% (l,6) -> gamma s^ + beta   s_tilde




function P = yes_MA(X, NetParams, mu_MA, v_MA)
    % evaluate linear part
    eps = 1e-9;
    layers = numel(NetParams.W);
    z = {};
    
    for l = 1 : layers
        if l == 1
            s = NetParams.W{l} * X + NetParams.b{l} *  ones(1,size(X,2));
        else
            s = NetParams.W{l} * z{l-1,3} + NetParams.b{l} *  ones(1,size(z{l-1,3},2));
        end
        
        if l ~= layers
            z{l,1} = s;
            % batch normalize
            z{l,4} = mu_MA{l};
            s = s - z{l,4} * ones(1,size(s,2));
            
            z{l,5} = v_MA{l};
            s = s ./ (sqrt(z{l,5}* ones(1,size(s,2))));
            z{l,2} = s;
            % apply gamma and beta
                        
            s = (NetParams.gammas{l}* ones(1,size(s,2))) .* s;
            s = s + NetParams.betas{l} *  ones(1,size(s,2));
            
            z{l,6} = s;
            
            %apply relu
            z{l,3} = max(0,s);
        end
    end
    
    %last layer
    z{layers,1} = s;
    % compute the softmax:
    % - numerators of the softmax:
    E = exp(s);
    % - denominators of the softmax:
    D = ones(size(E,1),1) * sum(E,1);
    
    z{layers,3} = E./D;
    % Divide each column by their sum
    % to have the softmax
    P = z;
end