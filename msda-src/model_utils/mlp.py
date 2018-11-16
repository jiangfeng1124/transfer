import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from .basic import ModelBase
from .basic import get_activation_module
from .basic import indent

import random

class MLP(ModelBase):

    @staticmethod
    def add_config(cfgparser):
        super(MLP, MLP).add_config(cfgparser)
        cfgparser.add_argument("--n_in", type=int, help="input dimension", default=5000)
        cfgparser.add_argument("--n_d", type=int, help="hidden dimension")
        cfgparser.add_argument("--activation", "--act", type=str, help="activation func")
        cfgparser.add_argument("--dropout", type=float, help="dropout prob")

    def __init__(self, configs, n_in = None, n_d = None):
        super(MLP, self).__init__(configs)
        self.n_d = n_d or configs.n_d or 500
        self.n_in = n_in or configs.n_in or 5000
        self.activation = configs.activation or 'tanh'
        self.dropout = configs.dropout or 0.0
        self.use_cuda = configs.cuda

        self.dropout_op = nn.Dropout(self.dropout)
        activation_module = get_activation_module(self.activation)

        self.seq = nn.Sequential()
        self.seq.add_module('linear', nn.Linear(self.n_in, self.n_d))
        self.seq.add_module('activation', activation_module())
        if self.dropout > 0:
            self.seq.add_module('dropout', nn.Dropout(p=configs.dropout))

        ### initialize
        nn.init.xavier_normal(self.seq[0].weight)
        nn.init.constant(self.seq[0].bias, 0.1)
        # self.seq[0].bias.requires_grad = False

        self.n_out = self.n_d

        self.build_output_op()

    def forward(self, batch):
        hidden = self.seq(batch)

        return hidden

class MLP_D(nn.Module):
    def __init__(self, ninput, noutput, layers,
                 activation=nn.LeakyReLU(0.2), gpu=False):
        super(MLP_D, self).__init__()
        self.ninput = ninput
        self.noutput = noutput

        layer_sizes = [ninput] + [int(x) for x in layers.split('-')]
        self.layers = []

        for i in range(len(layer_sizes)-1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            self.layers.append(layer)
            self.add_module("layer"+str(i+1), layer)

            # No batch normalization after first layer
            if i != 0:
                bn = nn.BatchNorm1d(layer_sizes[i+1], eps=1e-05, momentum=0.1)
                self.layers.append(bn)
                self.add_module("bn"+str(i+1), bn)

            self.layers.append(activation)
            self.add_module("activation"+str(i+1), activation)

        layer = nn.Linear(layer_sizes[-1], noutput)
        self.layers.append(layer)
        self.add_module("layer"+str(len(self.layers)), layer)

        self.init_weights()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        x = torch.mean(x)
        return x

    def init_weights(self):
        init_std = 0.02
        for layer in self.layers:
            try:
                layer.weight.data.normal_(0, init_std)
                layer.bias.data.fill_(0)
            except:
                pass


class MLP_G(nn.Module):
    def __init__(self, ninput, noutput, layers,
                 activation=nn.ReLU(), gpu=True):
        super(MLP_G, self).__init__()
        self.ninput = ninput
        self.noutput = noutput

        layer_sizes = [ninput] + [int(x) for x in layers.split('-')]
        self.layers = []

        for i in range(len(layer_sizes)-1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            self.layers.append(layer)
            self.add_module("layer"+str(i+1), layer)

            bn = nn.BatchNorm1d(layer_sizes[i+1], eps=1e-05, momentum=0.1)
            self.layers.append(bn)
            self.add_module("bn"+str(i+1), bn)

            self.layers.append(activation)
            self.add_module("activation"+str(i+1), activation)

        layer = nn.Linear(layer_sizes[-1], noutput)
        self.layers.append(layer)
        self.add_module("layer"+str(len(self.layers)), layer)
        self.layers.append(activation)

        self.init_weights()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x

    def init_weights(self):
        init_std = 0.02
        for layer in self.layers:
            try:
                layer.weight.data.normal_(0, init_std)
                layer.bias.data.fill_(0)
            except:
                pass


