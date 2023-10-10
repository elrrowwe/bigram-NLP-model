import matplotlib.pyplot as plt     
import torch
from utils import stoi, itos

"""
This file contains my implementation of the bigram language model, as described by Andrej Karpathy in his 
'The spelled-out intro to language modeling: building makemore' video.
"""

#opening the data file, taken from Andrej's github repo 
words = open('names.txt', 'r').read().splitlines()

#defining a tensor to store bigram counts, based on which bigram probabilities will be calculated later
counts_tensor = torch.zeros((28,28), dtype=torch.int32)

for w in words:
    chars = ['.'] + list(w) + ['.'] #adding special start and end tokens to each word
    for i,j in zip(chars, chars[1:]):
        ind_i = int(stoi(i))
        ind_j = int(stoi(j))

        counts_tensor[ind_i, ind_j] += 1

#preprocessing the tensor for efficiency 
P_prep = (counts_tensor+1).float() #adding 1 to smoothen the model, since if there is some unencountered bigram in the test dataset, its likelihood will be 0, which is undesirable 
P_prep /= counts_tensor.sum(dim=1, keepdim=True)
#the dimension argument for the sum operation should be 1, since we want to sum up all the row entries, thus sum "horizontally", creating a 27 by 1 tensor 

#the sampling loop

for _ in range(20):
    ind = 0
    output = []
    while True:
        p = P_prep[ind] #the tensor (row of counts_tensor), corresponding to the current letter 

        ind = torch.multinomial(p, num_samples=1, replacement=True).item() #sampling a character from the probability distribution tensor
        output.append(str(itos(ind)))
        if ind == 0: #if the index sampled is the end token -- break the loop
            break
    print(''.join(output))

#the total log likelihood
ll = 0.0
n = 0
#calculating the log likelihood of each bigram for model quality estimation 
for w in words:
    chars = ['.'] + list(w) + ['.'] #adding special start and end tokens to each word
    for i,j in zip(chars, chars[1:]):
        ind_i = int(stoi(i))
        ind_j = int(stoi(j))
        prob = P_prep[ind_i, ind_j]
        prob_log = torch.log(prob)
        ll += prob_log
        n += 1 #keeping track of the number of bigrams to later on calculate the average log likelihood (ll/n)
print(f'{-ll=}')
print(f'the average log likelihood: {-ll/n}')
