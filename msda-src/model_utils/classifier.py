import random
import sys
import argparse
import numpy as np

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F

class Classifier(nn.Module):

    def __init__(self, input_size, mlp_size, output):
        super(Classifier, self).__init__()
        self.proj1 = nn.Linear(input_size, mlp_size)
        self.proj2 = nn.Linear(mlp_size, output)

    def forward(self, embeddings):
        return self.proj2(self.proj1(embeddings))

