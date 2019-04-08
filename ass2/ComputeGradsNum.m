function [grad_b, grad_W] = ComputeGradsNum(X, Y, W, b, lambda, h)

no = size(W, 1);
d = size(X, 1);

grad_W{1} = zeros(size(W{1}));
grad_b{1} = zeros(size(b{1}));

c = ComputeCost(X, Y, W, b, lambda);

for i=1:length(grad_b{1})
    b_try = b{1};
    b_try(i) = b_try(i) + h;
    c2 = ComputeCost(X, Y, W, {b_try,b{2}}, lambda);
    grad_b{1}(i) = (c2-c) / h;
end

for i=1:numel(W{1})   
    
    W_try = W{1};
    W_try(i) = W_try(i) + h;
    c2 = ComputeCost(X, Y, {W_try,W{2}}, b, lambda);
    
    grad_W{1}(i) = (c2-c) / h;
end

grad_W{2} = zeros(size(W{2}));
grad_b{2} = zeros(size(b{2}));


for i=1:length(grad_b{2})
    b_try = b{2};
    b_try(i) = b_try(i) + h;
    c2 = ComputeCost(X, Y, W, {b{1},b_try}, lambda);
    grad_b{2}(i) = (c2-c) / h;
end

for i=1:numel(W{2})   
    
    W_try = W{2};
    W_try(i) = W_try(i) + h;
    c2 = ComputeCost(X, Y, {W{1},W_try}, b, lambda);
    
    grad_W{2}(i) = (c2-c) / h;
end



end