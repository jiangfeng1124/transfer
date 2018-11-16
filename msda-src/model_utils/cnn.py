import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from .basic import ModelBase
from .basic import get_activation_module

import random

# torch.manual_seed(1234)
# random.seed(1234)

class CNN(ModelBase):

    @staticmethod
    def add_config(cfgparser):
        '''
        At the moment, we stick to one-layer convolution
        '''
        super(CNN, CNN).add_config(cfgparser)
        cfgparser.add_argument("--n_d", type=int, help="hidden dimension (for CNN: #output channel)")
        cfgparser.add_argument("--activation", "--act", type=str, help="activation func")
        cfgparser.add_argument("--dropout", type=float, help="dropout prob")
        cfgparser.add_argument("--num_layers", type=int, help="number of non-linear layers")
        cfgparser.add_argument("--kernel_width", type=str, help="width of kernel")

    def __init__(self, embedding_layer, configs):
        super(CNN, self).__init__(configs)
        self.embedding_layer = embedding_layer
        self.embedding = embedding_layer.embedding
        self.n_e = embedding_layer.n_d # (kernel_width, n_e)
        self.n_d = configs.n_d or 300
        self.activation = configs.activation or 'tanh'
        self.dropout = configs.dropout or 0.0
        self.num_layers = configs.num_layers or 0
        self.kernel_widths = map(int, (configs.kernel_width).split(','))
        self.use_cuda = configs.cuda

        self.dropout_op = nn.Dropout(self.dropout)
        Ci = 1
        Co = self.n_d
        kernel_Ws = self.kernel_widths
        D = self.n_e
        padding_Ws = map(lambda k : int((k - 1) / 2.0), kernel_Ws) # K should be odd number
        padding_H = 0
        convs = []
        for k, w in zip(kernel_Ws, padding_Ws):
            convs.append(nn.Conv2d(Ci, Co, (k, D), padding = (w, padding_H)))
        self.convs = nn.ModuleList(convs)

        activation_module = get_activation_module(self.activation)
        self.seq = seq = nn.Sequential()
        self.n_out = len(self.convs) * self.n_d
        for i in range(self.num_layers):
            seq.add_module('linear-{}'.format(i),
                nn.Linear(self.n_out, self.n_out)
            )
            seq.add_module('activation-{}'.format(i),
                activation_module()
            )
            if self.dropout > 0:
                seq.add_module('dropout-{}'.format(i),
                    nn.Dropout(p=configs.dropout)
                )
        for layer in seq:
            try:
                nn.init.xavier_normal(layer.weight)
                nn.init.constant(layer.bias, 0.1)
            except:
                pass

        self.build_output_op() # depends on criterion

    def forward(self, batch):
        # (len, batch_size)
        emb = self.embedding(batch) # (len, batch_size, n_e)
        emb = emb.transpose(0, 1)   # (batch_size, len, n_e)

        # print emb.size()
        assert emb.dim() == 3

        if self.dropout > 0:
            emb = self.dropout_op(emb)

        emb = emb.unsqueeze(1)
        # print emb.size()
        outputs = [ F.relu(conv(emb)).squeeze(3) for conv in self.convs ] # [(batch_size, Co|d, len)]
        # outputs = [ F.max_pool1d(output, output.size(2)).squeeze(2) for output in outputs ]
        output = torch.cat(outputs, 1)
        output = F.max_pool1d(output, output.size(2)).squeeze(2) # (batch_size, Co|d)
        # print output.size()

        if self.num_layers > 0:
            output = self.seq(output)

        return output

