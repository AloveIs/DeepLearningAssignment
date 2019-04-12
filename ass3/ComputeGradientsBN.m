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
function [grad_W, grad_b, grad_gammas, grad_betas] = ComputeGradientsBN(X, Y, P, NetParams, lambda)
    % compute useful constant
    batch_size = double(size(X,2));


    W = NetParams.W;
    b = NetParams.b;

    l = numel(W);
    grad_b = {};
    grad_W = {};
    grad_gammas = {};
    grad_betas = {};
    % compute g as defined on the slides
    % for lest layer
    g = -(Y-P{end,3});

    %last layer
    H = P{l-1,3};
    grad_b{l} = 1.0/ batch_size * sum(g,2);
    grad_W{l} = 1.0/ batch_size * (g * H') + 2 * lambda * W{l};

    %update gradient
    g = (W{l}' * g);
    g(H==0) = 0;
    l = l - 1;

    while l >= 1
        
        grad_gammas{l} = mean(g .* P{l,2},2);
        grad_betas{l} = mean(g,2);
        
        g = g .* (NetParams.gammas{l} * ones(1,batch_size));
        
        % BatchNormBackPass
        
        
        sigma_1 = P{l,5}.^(-0.5);
        sigma_2 = P{l,5}.^(-1.5);
        
        size(sigma_1)
        
        G1 = g .* (sigma_1 * ones(1,batch_size));
        G2 = g .* (sigma_2 * ones(1,batch_size));
        D = P{l,1} - P{l,4} * ones(1,batch_size);
        c = (G2 .* D) * ones(batch_size,1);
        g = G1 - (1/batch_size) * (G1 * ones(batch_size,1))* ones(1,batch_size) ...
            - (1/batch_size) * (D .* (c * ones(1,batch_size)));
        
        % end batch norm back pass
        if l == 1
            H = X;
        else
            H = P{l-1,3};
        end

        grad_b{l} = 1.0/ batch_size * sum(g,2);
        grad_W{l} = 1.0/ batch_size * (g * H') + 2 * lambda * W{l};


        if l > 1
            %update gradient
            g = (W{l}' * g);
            g(H==0) = 0;
        end


        l = l - 1;
    end

end
