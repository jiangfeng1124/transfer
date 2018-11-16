import numpy as np
import torch
from torch.autograd import Variable

def one_hot(indices, n=2, cuda=True):
    ''' Convert indices to one-hot representation
        - default n = 2, indicating binary classification

        Arguments:
            - indices: indices of batch_size
            - n (default: 2): dimension of the resulting one-hot vector

        Return: one-hot representation (batch_size * n)
    '''
    one_hot_rep = torch.FloatTensor(indices.shape[0], 2).zero_()
    one_hot_rep = Variable(one_hot_rep)
    if cuda:
        one_hot_rep = one_hot_rep.cuda()
    one_hot_rep.scatter_(1, indices.unsqueeze(1), 1)

    return one_hot_rep

def softmax(data):
    ''' Element-wise normalization over a list of FloatTensors

        Arguments:
            - data: list of FloatTensors with the same length
    '''
    data_exp = [ torch.exp(x) for x in data ]
    return [ x / sum(data_exp) for x in data_exp ]

