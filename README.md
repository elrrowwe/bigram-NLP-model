# bigram-NLP-model
This is my implementation of the n-gram language model, as described by Andrej Karpathy in his makemore series.

# How does it work?
An n-gram language model works by assigning probabilities to each n-gram encountered in the training set and predicting the next letter in a sequence, based on the probabilities calculated earlier; a bigram model is a generalization of the n-gram one. 

# Implementation description (bigram)
1) Create a string-to-integer tokenizer (essentially a converter from strings to integers) and a reverse, integer-to-string mapping;
2) Put all the counts of the bigrams, encountered in the data set into a tensor (***counts_tensor***). The position of the count of a bigram is determined by the mapping from that bigram to integers. Each row, column label is a letter (or the special '.' token), and each other entry is the count of the combination of the two symbols at a given position; 
3) Normalizing the tensor by dividing each entry by the total number of counts so that a tensor with probabilities is produced;
4) Iteratively sampling from the tensor by first sampling from the starting token row, then determining the current row by the lastly sampled letter (i.e., we sample the letter '*a*' -- we switch to the row with label '*a*', sample from the row...);
5) Outputting the sampled characters, which are supposed to resemble real names;
6) Calculating the negative [log likelihood](https://www.statisticshowto.com/log-likelihood-function/) of all the bigrams, thus estimating the quality of the model (the lower the negative log likelihood -- the better the model is).

#ToDo
Further goals include experimenting with splitting the data, adding more layers to the network and hacking the code in arbitrary ways. 
