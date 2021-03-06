%% MiniBatchGD
%
% Perform the model update.
% where X contains all the training images, Y the labels for the training
% images, W, b are the initial values for the network’s parameters, lambda
% is the regularization factor in the cost function and GDparams is an object containing the parameter values n batch, eta
% and n epochs
%
function [Wstar, bstar] = MiniBatchGD(X, Y, GDparams, W, b, lambda, Xval, Yval, decay_rate)
    
    if nargin < 9
        decay_rate = 1.0;
    end
    
    Gradients = [];
    
    batch_size = int32(GDparams.n_batch);
    eta = GDparams.eta;
    % matrices to save cost and accuracy after each epoch
    C = zeros(GDparams.n_epochs,2);
    A = zeros(GDparams.n_epochs,2);
    
    
    for epoch = 1 : GDparams.n_epochs
        batch = 1;
        start_index  = 1;
        while start_index < size(X,2)
            
            if start_index >= size(X,2)
                break;
            end
            %get indexes of the batch data
            idx = start_index : min(start_index + batch_size -1, size(X,2));
            
            %update starting index
            start_index = start_index + batch_size;
            
            % index the actual data
            X_batch = X(:,idx);
            Y_batch = Y(:,idx);
            
            % update parameters
            P = EvaluateClassifier(X_batch, W, b);
            [grad_W, grad_b] = ComputeGradients(X_batch, Y_batch, P, W, lambda);
            %save gradients computed
            Gradients = [Gradients,grad_W];
            W = W - eta * grad_W;
            b = b - eta * grad_b;
            batch = batch + 1;
        end
        %update eta
        eta = decay_rate * eta;
        
        % save the cost and accuracy after each epoch
        C(epoch,1) = ComputeCost(X, Y, W, b, lambda);
        C(epoch,2) = ComputeCost(Xval, Yval, W, b, lambda);
        
        A(epoch,1) = compute_accuracy(X, Y, W, b);
        A(epoch,2) = compute_accuracy(Xval, Yval, W, b);
                
    end
    
    % print informations on the gradient statistics
    fprintf("mean : %f, std_dev : %f\n", mean(Gradients(:)), sqrt(var(Gradients(:))));
    % plot loss and accuracy of the network
    x = 1 : GDparams.n_epochs;
    plot(x, C(:,1),x, C(:,2));
    figure();
    plot(x, A(:,1),x, A(:,2));
    
    % set return values
    Wstar = W;
    bstar = b;
end



function acc = compute_accuracy(X,Y,W,b)

y = vec2ind(Y);

P = EvaluateClassifier(X, W, b);

[~, argmax] = max(P);

R = argmax == y;

acc = double(sum(R))/size(Y,2)*100;

end