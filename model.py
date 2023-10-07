import torch

"""
This file contains my implementation of the bigram language model, as described by Andrej Karpathy in his 
'The spelled-out intro to language modeling: building makemore' video.
"""

#opening the data file, taken from Andrej's github repo 
words = open('names.txt', 'r').read().splitlines()

print(words[:10])

for w in words[:1]:
    for i,j in zip(w, w[1:]):
        print(i,j)

#word beginning and end tokens
start = '<S>'
end = '<E>'