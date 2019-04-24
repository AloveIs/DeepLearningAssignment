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