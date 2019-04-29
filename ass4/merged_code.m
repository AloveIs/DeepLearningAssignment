

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Content of file : Ass4.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


book_fname = 'goblet_book.txt';
fid = fopen(book_fname,'r');
book_data = fscanf(fid,'%c');
fclose(fid);

book_chars = unique(book_data);


char_to_ind = containers.Map('KeyType','char','ValueType','int32');
ind_to_char = containers.Map('KeyType','int32','ValueType','char');

K = length(book_chars);

for c = 1 : K
   char_to_ind(book_chars(c)) = c;
   ind_to_char(c) = book_chars(c);
    
end

%% initialization and parameters

% std dev
sig = 0.01;

RNN.K = K;
RNN.m = 100;
RNN.seq_length = 25;
RNN.eta = 0.1;

m = RNN.m;

RNN.b = zeros(m,1);
RNN.c = zeros(K,1);

RNN.U = randn(m, K)*sig;
RNN.W = randn(m, m)*sig;
RNN.V = randn(K, m)*sig;

RNN.n_epoch = 10;

txt_idx = generate_text(RNN,100);

indxs2txt(txt_idx, ind_to_char)


%% gradient
seq_len = 100;

X_chars = one_hot(book_data(1:RNN.seq_length),char_to_ind);
Y_chars = one_hot(book_data(2:RNN.seq_length+1),char_to_ind);


if true

    FORWARD = forward(RNN , X_chars,Y_chars);
    grad = ComputeGradients(X_chars,Y_chars,RNN , FORWARD);

    num_grads = ComputeGradsNum(X_chars, Y_chars, RNN, 1e-5);
    max(max(abs(grad.V- num_grads.V)))
    max(max(abs(grad.U- num_grads.U)))
    max(max(abs(grad.W- num_grads.W)))
    max(max(abs(grad.b- num_grads.b)))
    max(max(abs(grad.c- num_grads.c)))
    
end



[RNN_trained, h, smoothL] = ADAGrad(book_data,RNN,char_to_ind,ind_to_char);
txt_idx = generate_text(RNN_trained,300,h);

indxs2txt(txt_idx, ind_to_char)



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Content of file : ADAGrad.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [RNN, h, SmoothLoss] = ADAGrad(x,RNN,char_to_ind,ind_to_char)

    len = length(x);
    Loss = [];
    SmoothLoss = [];
%     for param = {"U","V","W","c","b"}
%         disp(param)
%         M.(param) = zeros(size(RNN.(param)));
%     end
    
    step = 0;

    M.U = zeros(size(RNN.U));
    M.V = zeros(size(RNN.V));
    M.W = zeros(size(RNN.W));
    M.c = zeros(size(RNN.c));
    M.b = zeros(size(RNN.b));
    h = zeros(RNN.m,1);
    smooth_loss = 0;
    

    X = one_hot(x(1:RNN.seq_length-1),char_to_ind);
    txt_idx = generate_text(RNN,200,h,X(:,end));
    disp(indxs2txt(txt_idx, ind_to_char));
    
    
    for epoch = 1 : RNN.n_epoch
        fprintf("epoch : %d \n",epoch);
        e = 1;
        h = zeros(RNN.m,1);
        while e < len - RNN.seq_length -1
           X = one_hot(x(e:e+RNN.seq_length-1),char_to_ind);
           Y = one_hot(x(e+1:e+RNN.seq_length),char_to_ind); 
            
           

           
           FORWARD = forward(RNN , X,Y,h);
           
           if smooth_loss == 0
               smooth_loss = FORWARD.loss;
           else 
               smooth_loss = 0.999*smooth_loss + 0.001*FORWARD.loss;
           end
           
           h = FORWARD.h(:,end);
           
           grad = ComputeGradients(X,Y,RNN , FORWARD);
           
           for param = {"U","V","W","c","b"} % AdaGrad
              M.(param{1}) = M.(param{1})+grad.(param{1}) .^ 2;
              
              %avoid exploding gradient
              grad.(param{1}) = max(min(grad.(param{1}), 5), -5);


              
              RNN.(param{1}) = RNN.(param{1})-RNN.eta * grad.(param{1}) ./ sqrt(M.(param{1})+ 1e-15);
           end
           
           
           step = step + 1;
           
           e = e+RNN.seq_length;
           if mod(e-1,25*10000) == 0
               txt_idx = generate_text(RNN,200,h,X(:,end));
                disp(indxs2txt(txt_idx, ind_to_char));
           end
           
           
           if mod(e-1,25*100) == 0
              Loss = [Loss;FORWARD.loss];
              if length(SmoothLoss) == 0 %#ok<ISMT>
                SmoothLoss = [FORWARD.loss];
              else
                SmoothLoss = [SmoothLoss; smooth_loss];
              end
              fprintf("%d) step = %d loss = %f perc : %.2f %%\n",epoch, step, smooth_loss ,e *100 / len);
           
           end
        end
    end
    
    disp("#####################\n Finsihed Training \n\n");
    txt_idx = generate_text(RNN,1000,h,X(:,end));
    
    disp(indxs2txt(txt_idx, ind_to_char))
    
    
    figure();
    plot(Loss);
    figure();
    plot(SmoothLoss(2:end));
    
    disp("ciao")
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Content of file : backward.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Content of file : forward.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Content of file : indxs2txt.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function txt = indxs2txt(idxs, idx_to_char)
    txt = "";
    
    for i = 1 :length(idxs)
        txt = txt + idx_to_char(idxs(i));
    end
    
end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Content of file : ComputeGradients.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Content of file : ComputeLoss.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function loss = ComputeLoss(X, Y, RNN, h)
W = RNN.W;
U = RNN.U;
V = RNN.V;
b = RNN.b;
c = RNN.c;
n = size(X, 2);
loss = 0;

for t = 1 : n
    at = W*h + U*X(:, t) + b;
    h = tanh(at);
    o = V*h + c;
    pt = exp(o);
    p = pt/sum(pt);

    loss = loss - log(Y(:, t)'*p);
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Content of file : generate_text.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Content of file : one_hot.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function Y = one_hot(X, char_to_idx)
    
    L = length(X);
    Y = zeros(length(char_to_idx.keys()) ,L);

    for i = 1 : L
        Y(char_to_idx(X(i)),i) = 1; 
    end
end