import matplotlib.pyplot as plt     
import torch

"""
This file contains my implementation of the bigram language model, as described by Andrej Karpathy in his 
'The spelled-out intro to language modeling: building makemore' video.
"""

#opening the data file, taken from Andrej's github repo 
words = open('names.txt', 'r').read().splitlines()

print(f'the first 10 words from the list: \n {words[:10]}')

counts_dict = {} #a dictionary to store
for w in words:
    chars = ['<S>'] + list(w) + ['<E>'] #adding special start and end tokens to each word
    for i,j in zip(chars, chars[1:]):
        bigram = (i,j)
        counts_dict[bigram] = counts_dict.get(bigram, 0) + 1 #basically adding 1 to the value of a at bigram each time that bigram is encountered 
counts_dict = sorted(counts_dict.items(), key = lambda key_value: -key_value[1]) #sorts the a dictionary, using the count of each bigram as the key
#since the sorted function sorts items in the ascending order, it is necessary to make key_value negative to attain the descending order
    
#defining a tensor to store bigram counts, based on which bigram probabilities will be calculated later
counts_tensor = torch.zeros((28,28), dtype=torch.int32)

chars = sorted(list(set(''.join(words)))) #a string, containing the whole dataset (all the names, joined)

#defining a simple string to integer tokenizer 
stoi = {s:i for i,s in enumerate(chars)}

#adding tokens for the special characters, since they are not included in the dataset (a mapping from strings to integers)
stoi['<S>'] = 26 
stoi['<E>'] = 27

#now, defining a mapping from integers to strings (essentially reverses the stoi tokenizer)
itos = {i:s for i,s in enumerate(chars)}
itos[26] = '<S>'
itos[27] = '<E>'

for w in words:
    chars = ['<S>'] + list(w) + ['<E>'] #adding special start and end tokens to each word
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