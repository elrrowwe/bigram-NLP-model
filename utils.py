"""
This file contains utility functions used throughout the code. 
"""

words = open('names.txt', 'r').read().splitlines()
chars = sorted(list(set(''.join(words)))) 

#creating lookup dictionaries for tokenization 
_stoi = {s:i+1 for i,s in enumerate(chars)}
_stoi['.'] = 0

_itos = {i:s for s,i in _stoi.items()}

#a function to tokenize a string into an int 
def stoi(char):
    return _stoi[char]

#a function to "detokenize" an int into a string 
def itos(ind):
    return _itos[ind]