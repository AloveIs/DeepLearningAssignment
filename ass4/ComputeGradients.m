function [grads] = ComputeGradients(X, Y, RNN, FORWARD)

    N = RNN.seq_length;
    dH_do = zeros(RNN.seq_length, RNN.m);
    dA_do = zeros(RNN.seq_length, RNN.m);
    
    P = FORWARD.p;
    H = FORWARD.h;

    A = FORWARD.a;
    
    dL_do = -(Y - P)';
    grads.V = dL_do' * H';
    
    grads.c = sum(dL_do',2);
    

    % compute backwards the gradients
    dL_dh(N, :) = dL_do(N, :) * RNN.V;  
    dL_da(N, :) = dL_dh(N, :) * diag(1 - (tanh(A(:, N))).^2);
    
    
    for t = N - 1 : -1 : 1
        dL_dh(t, :) = dL_do(t, :) * RNN.V + dL_da(t + 1, :) * RNN.W;
        dL_da(t, :) = dL_dh(t, :) * diag(1 - (tanh(A(:, t))).^2);
    end

    grads.b = sum(dL_da',2);
    
    % H containing the h0 and wothout the last hidden state
    H_shifted = [FORWARD.h0, H(:,1:end-1)];

    grads.W = dL_da' * H_shifted';
    grads.U = dL_da' * X'; 

end