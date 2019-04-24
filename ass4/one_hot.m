function Y = one_hot(X, char_to_idx)
    
    L = length(X);
    Y = zeros(length(char_to_idx.keys()) ,L);

    for i = 1 : L
        Y(char_to_idx(X(i)),i) = 1; 
    end
end