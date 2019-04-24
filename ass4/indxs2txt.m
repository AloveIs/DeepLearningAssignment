function txt = indxs2txt(idxs, idx_to_char)
    txt = "";
    
    for i = 1 :length(idxs)
        txt = txt + idx_to_char(idxs(i));
    end
    
end


