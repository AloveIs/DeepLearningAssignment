%% MiniBatchGD
%
% Perform the model update.
% where X contains all the training images, Y the labels for the training
% images, W, b are the initial values for the network’s parameters, lambda
% is the regularization factor in the cost function and GDparams is an object containing the parameter values n batch, eta
% and n epochs
%
function [Wstar, bstar] = MiniBatchGD(X, Y, GDparams, W, b, lambda, Xval, Yval)
    
    batch_size = int32(GDparams.n_batch);
    
    eta_min = 1e-5;
    eta_max = 1e-1;
    n_s = GDparams.n_step;
    cycles= GDparams.n_cycles;
    
    Rounds=2*(cycles*n_s);
    
    % matrices to save cost and accuracy after each epoch
    C = zeros(Rounds,4);
    A = zeros(Rounds,2);
    etas = zeros(Rounds,1);
    
    eta = eta_min;
    eta_step = (eta_max-eta_min)/(n_s);
    start_index  = 1;
    sign = 1;
    
    for rounds = 1 : Rounds
        if mod(rounds,100)==0
            fprintf("Round %d of %d : %.2f %%\n",rounds,Rounds,100*rounds/Rounds);
        end
        if start_index >= size(X,2)
            start_index = 1;
        end
        %get indexes of the batch data
        idx = start_index : min(start_index + batch_size -1, size(X,2));

        % index the actual data
        X_batch = X(:,idx);
        Y_batch = Y(:,idx);

        %update starting index
        start_index = start_index + batch_size;

        P = EvaluateClassifier(X_batch, W, b);
        [grad_W, grad_b] = ComputeGradients(X_batch, Y_batch, P, W,b, lambda);

        W{1} = W{1} - eta * grad_W{1};
        b{1} = b{1} - eta * grad_b{1};
        W{2} = W{2} - eta * grad_W{2};
        b{2} = b{2} - eta * grad_b{2};
              
        %save statistics
        C(rounds,1) = ComputeCost(X, Y, W, b, lambda);
        C(rounds,2) = ComputeCost(Xval, Yval, W, b, lambda);
        C(rounds,3) = ComputeCost(X, Y, W, b, 0);
        C(rounds,4) = ComputeCost(Xval, Yval, W, b, 0);
        
        A(rounds,1) = compute_accuracy(X, Y, W, b);
        A(rounds,2) = compute_accuracy(Xval, Yval, W, b);
        etas(rounds) = eta;
        
        %update eta
        eta = eta + sign*eta_step;
        if(eta >= (eta_max-1e-9))
            eta = eta_max;
            sign = -sign;
        elseif (eta <= (eta_min+1e-9))
            eta = eta_min;
            sign = -sign;
        end
    end
    
    % plot loss and accuracy of the network
    x = 1 : 100 : Rounds;
    plot(x, C(x,1),x, C(x,2));
    figure();
    plot(x, C(x,3),x, C(x,4));
    figure();
    plot(x, A(x,1),x, A(x,2));
    %figure();
    %plot(x-1, etas);
    save("test_fig3","C","A");
    % set return values
    Wstar = W;
    bstar = b;
end



function acc = compute_accuracy(X,Y,W,b)

y = vec2ind(Y);

P = EvaluateClassifier(X, W, b);

[~, argmax] = max(P{2});

R = argmax == y;

acc = double(sum(R))/size(Y,2)*100;

end