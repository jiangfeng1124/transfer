import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from .basic import ModelBase
from .basic import get_activation_module
from .basic import indent

import random

class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()

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

    def forward(self, batch):
        hidden = self.seq(batch)

        return hidden
