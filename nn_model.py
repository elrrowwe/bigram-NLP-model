import torch 

"""
This file contains the neural network class, used for generating the look-up bi(tri)gram counts table. 
The network is set up to be a one-layer one, though it could be further complexified to achieve results
better than these obtained with a counting lookup table. 
"""

#the network
class NN():
    def __init__(self, W_dim, num):   #since we'd like to do backprop for W later
        self.W = torch.randn((W_dim), requires_grad=True) #randomly generated weights
        self.num = num

    #the softmax activation function 
    def softmax(self, z):
        e_z = z.exp()
        
        P = e_z / e_z.sum(1, keepdim=True)

        return P
    
    #the log likelihood loss (since we are doing classification)
    def loss(self, probs, y):
        nll = (-probs[torch.arange(self.num), y].log()).mean()  #regularizing the model so that it produces no weird probabilities (0's, inf's etc)

        return nll
    
    #the forward pass function 
    def forward(self, inp):
        self.inp = inp

        #multiplying the weights matrix and the input
        out = inp @ self.W

        #activating the output with softmax (P for probabilities)
        P = self.softmax(out)

        return P 
    
    def backward(self, l, learning_rate):
        self.W.grad = None #resetting the gradient of W
        l.backward()

        self.W.data += -learning_rate * self.W.grad
    
    def train(self, X, y, learning_rate=0.01, epochs=100):
        for iteration in range(epochs+1):
            self.learning_rate = learning_rate
            dense = self.forward(X)

            l = self.loss(dense, y, )

            self.backward(l, self.learning_rate)

            print(f'epoch: {iteration}, loss: {l}')

        return self.W