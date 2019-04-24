
all_tweets = "";


for fname = {"master_2012.json","master_2013.json", ...
        "master_2014.json","master_2015.json","master_2016.json", ...
        "master_2017.json", "master_2018.json"}
    
    disp("Decoding " + fname{1});
    
    decoded = jsondecode(fileread(fname{1}));
    if fname == "master_2017.json"  || fname == "master_2018.json"
        for i = 1 : length(decoded) 
           all_tweets = all_tweets + decoded{i}.full_text + "\n";
        end
        
    else
        for i = 1 : length(decoded) 
           all_tweets = all_tweets + decoded{i}.text + "\n";
        end
    end
end
