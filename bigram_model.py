import matplotlib.pyplot as plt     
import torch

"""
This file contains my implementation of the bigram language model, as described by Andrej Karpathy in his 
'The spelled-out intro to language modeling: building makemore' video.
"""

#opening the data file, taken from Andrej's github repo 
words = open('names.txt', 'r').read().splitlines()

counts_dict = {} #a dictionary to store
for w in words:
    chars = ['<S>'] + list(w) + ['<E>'] #adding special start and end tokens to each word. they later on will be substituted for '.'
    for i,j in zip(chars, chars[1:]):
        bigram = (i,j)
        counts_dict[bigram] = counts_dict.get(bigram, 0) + 1 #basically adding 1 to the value of a at bigram each time that bigram is encountered

counts_dict = sorted(counts_dict.items(), key = lambda key_value: -key_value[1]) #sorts the a dictionary, using the count of each bigram as the key
#since the sorted function sorts items in the ascending order, it is necessary to make key_value negative to attain the descending order
    
#defining a tensor to store bigram counts, based on which bigram probabilities will be calculated later
counts_tensor = torch.zeros((28,28), dtype=torch.int32)

chars = sorted(list(set(''.join(words)))) #a sorted list of all the characters in the data set 

#defining a simple string to integer tokenizer (a mapping from strings to integers) 
stoi = {s:i+1 for i,s in enumerate(chars)}

#adding a token for a special character, since it is not included in the dataset 
stoi['.'] = 0

#now, defining a mapping from integers to strings (essentially reverses the stoi tokenizer)
itos = {i:s for s,i in stoi.items()}

for w in words:
    chars = ['.'] + list(w) + ['.'] #adding special start and end tokens to each word
    for i,j in zip(chars, chars[1:]):
        ind_i = stoi[i]
        ind_j = stoi[j]

        counts_tensor[ind_i, ind_j] += 1

#visualizing the tensor in a pretty way,took this from karpathy (doesn't work on my machine, there is most probably some compatibility issue) 
# plt.figure(figsize=(16,16))
# plt.imshow(counts_tensor, cmap='Blues')
# for i in range(27):
#     for j in range(27):
#         chstr = itos[i] + itos[j]
#         plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
#         plt.text(j, i, counts_tensor[i, j].item(), ha="center", va="top", color='gray')
# plt.axis('off')
# plt.show()

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
        output.append(itos[ind])
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
        ind_i = stoi[i]
        ind_j = stoi[j]
        prob = P_prep[ind_i, ind_j]
        prob_log = torch.log(prob)
        ll += prob_log
        n += 1 #keeping track of the number of bigrams to later on calculate the average log likelihood (ll/n)
print(f'{-ll=}')
print(f'the average log likelihood: {-ll/n}')
