function [RNN, M] = backward(RNN, X, Y, M)

    grads = ComputeGradients(RNN, X, Y);
    eps = 1e-14;

    for f = {"b","c","U","W","V"}
        
        % clip gradients to avoid exploding gradient
        grads.(f{1}) = max(min(grads.(f{1}), 5), -5);

        M.(f) = M.(f) + grads.(f).^2;
        RNN.(f) = RNN.(f{1}) - RNN.eta*(grads.(f)./(M.(f) + eps).^(0.5));
    end

end