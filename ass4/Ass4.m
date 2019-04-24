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

