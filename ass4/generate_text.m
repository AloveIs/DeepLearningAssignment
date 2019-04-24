function txt = generate_text(RNN, n, h0, x0)

    I = eye(83);

    if nargin < 4
       x0 = zeros(83,1);
       x0(1) = 1;
    end
    
    if nargin < 3
        h0 = zeros(RNN.m, 1);
    end

    txt = zeros(1,n);
    
    h = h0;
    x = x0;
    
    for t = 1 : n
        
        a = RNN.W * h  + RNN.U * x + RNN.b;
        h = tanh(a);
        o = RNN.V * h + RNN.c;
        
        p = softmax(o);
        
        cp = cumsum(p);
        smpl = rand;
        ixs = find(cp - smpl >0);
        ii = ixs(1);
        
        txt(t) = ii;
        x = I(:,ii);
    end
end



function res = softmax(a)

    E = exp(a);
    res = E/sum(E);

end