import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity="tanh"):
        super(RNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity
        if self.nonlinearity not in ["tanh", "relu"]:
            raise ValueError("Invalid nonlinearity selected for RNN.")

        self.x2h = nn.Linear(input_size, hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, hidden_size, bias=bias)

        self.reset_parameters()


    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)


    def forward(self, input, hx=None):

        # Inputs:
        #       input: of shape (batch_size, input_size)
        #       hx: of shape (batch_size, hidden_size)
        # Output:
        #       hy: of shape (batch_size, hidden_size)

        if hx is None:
            hx = Variable(input.new_zeros(input.size(0), self.hidden_size))

        hy = (self.x2h(input) + self.h2h(hx))

        if self.nonlinearity == "tanh":
            hy = torch.tanh(hy)
        else:
            hy = torch.relu(hy)

        return hy



'''input: Tensor of shape (batch_size, input_size).
hx: Previous hidden state (batch_size, hidden_size).
hx = input.new_zeros(input.size(0), self.hidden_size)
input.size(0): This fetches the batch size.
self.hidden_size: This specifies the size of the hidden state.
input.new_zeros(...): Creates a zero tensor with the same data type and'''



'''hy = (self.x2h(input) + self.h2h(hx))
This calculates:
Wx.xt +  wh.ht-1+b
'''
