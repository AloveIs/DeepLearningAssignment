function [Wstar, bstar] = MiniBatchGD(X, Y, GDparams, W, b, lambda, Xval, Yval)
    
    batch_size = int32(size(Y,2)/ GDparams.n_batch);
    C = zeros(GDparams.n_epochs,2);
    A = zeros(GDparams.n_epochs,2);
    
    disp(ComputeCost(X, Y, W, b, lambda));
    disp(ComputeCost(Xval, Yval, W, b, lambda));
    
    for epoch = 1 : GDparams.n_epochs
        batch = 1;
        start_index  = 1;
        while start_index < size(X,2)          %batch = 1 : GDparams.n_batch
            %fprintf("epoch %d - batch %d\n",epoch, batch);
            if start_index >= size(X,2)
                break;
            end
            
            idx = start_index : min(start_index + batch_size -1, size(X,2));
            %fprintf("%d\t%d\n",idx(1), idx(end));
            
            %update starting
            start_index = start_index + batch_size;
            
            X_batch = X(:,idx);
            Y_batch = Y(:,idx);
            
            P = EvaluateClassifier(X_batch, W, b);
            [grad_W, grad_b] = ComputeGradients(X_batch, Y_batch, P, W, lambda);
            W = W - GDparams.eta * grad_W;
            b = b - GDparams.eta * grad_b;
            batch = batch + 1;
        end
        %fprintf("Batches performed : %d", batch-1);
        %fprintf("###################\n");
        C(epoch,1) = ComputeCost(X, Y, W, b, lambda);
        C(epoch,2) = ComputeCost(Xval, Yval, W, b, lambda);
        fprintf("Epoch : %d\ttest: %f \tval: %f\n", epoch,C(epoch,1),C(epoch,2));
        
        A(epoch,1) = compute_accuracy(X, Y, W, b);
        A(epoch,2) = compute_accuracy(Xval, Yval, W, b);
    end
    
    
    
    x = 1 : GDparams.n_epochs;
    figure();
    semilogy(x, C(:,1),x, C(:,2));
    figure();
    plot(x, A(:,1),x, A(:,2));
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

