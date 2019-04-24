function FWRD = forward(RNN, X, Y, h0)
    
    if nargin < 4
        h0 = zeros(RNN.m, 1);
    end

    n = size(X,2);

    loss = zeros(1,n);
    
    a = zeros(RNN.m, n);
    h = zeros(RNN.m, n);
    o = zeros(RNN.K, n);
    p = zeros(RNN.K, n);
    
    for t = 1 : n
        
        if t == 1
            prev_h = h0;
        else
            prev_h = h(:, t-1);
        end
        a(:, t) = RNN.W * prev_h  + RNN.U * X(:, t) + RNN.b;
        h(:, t) = tanh(a(:, t));
        o(:, t) = RNN.V * h(:, t) + RNN.c;
        
        p(:, t) = softmax(o(:, t));
        
        loss(t) = Y(:, t)' * p(:, t);
        
    end
    
    FWRD.a = a;
    FWRD.h = h;
    FWRD.h0 = h0;
    FWRD.o = o;
    FWRD.p = p;
    FWRD.loss = -sum(log(loss));
    
end



function res = softmax(a)

    E = exp(a);
    res = E/sum(E);

end