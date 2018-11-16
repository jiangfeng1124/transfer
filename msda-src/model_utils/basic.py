
import torch
import torch.nn as nn

import random
import math

def get_activation_module(name):
    name = name.lower()
    if name == 'relu':
        return nn.ReLU
    elif name == 'tanh':
        return nn.Tanh
    elif name == 'sigmoid':
        return nn.Sigmoid
    else:
        raise Exception("Unknown activation type: {}".format(
            name
        ))

def indent(text, amount, ch=' '):
    padding = ch*amount
    return ''.join(padding + line for line in text.splitlines(True))

def init_weight(module):
    if isinstance(module, nn.Linear):
        # nn.init.xavier_normal(module.weight)
        nn.init.xavier_normal(module.weight)
        if module.bias is not None:
            nn.init.constant(module.bias, 0.1)
            # module.bias.requires_grad = False

def normalize_2d(x, eps=1e-8):
    assert x.dim() == 2
    l2 = x.norm(2,1)
    #print x.data.size()
    #print l2.data.size()
    return x/(l2+eps).expand_as(x)

def cosine_similarity(u, v):
    u2 = normalize_2d(u)
    v2 = normalize_2d(v)
    assert u2.dim() == 2
    assert v2.dim() == 2
    return (u2*v2).sum(1)

class ModelBase(nn.Module):

    @staticmethod
    def add_config(cfgparser):
        cfgparser.add_argument("--criterion", type=str, required=False,
            default="classification", help="which to use for training"
        )

    def __init__(self, configs):
        super(ModelBase, self).__init__()
        self.criterion = configs.criterion

    def build_output_op(self):
        crt = self.criterion
        if crt == 'classification':
            self.output_op = nn.Linear(self.n_out, 1)
            # init_weight(self.output_op)
        elif crt == 'classification2':
            self.output_op = nn.Linear(self.n_out, 2)
            # init_weight(self.output_op)
        elif crt == 'cosine':
            self.output_op = cosine_similarity
        else:
            raise Excpetion("Unknown criterion: {}".format(crt))

    def compute_similarity(self, left, right):
        crt = self.criterion
        if crt == 'classification':
            self.output_op.weight.data.clamp_(min=1e-6)
            out = left*right
            hidden = self.output_op(out) # (batch, 1)
            return torch.cat((-hidden, hidden), 1)
        elif crt == 'classification2':
            self.output_op.weight[0].data.zero_()
            self.output_op.weight[1].data.clamp_(min=1e-6)
            out = left*right
            return self.output_op(out)
        elif crt == 'cosine':
            return self.output_op(left, right)
        else:
            raise Excpetion("Unknown criterion: {}".format(crt))

    def compute_loss(self, output, target):
        crt = self.criterion
        if crt == 'classification':
            return nn.functional.cross_entropy(output, target)
        elif crt == 'classification2':
            return nn.functional.cross_entropy(output, target)
        elif crt == 'cosine':
            k = target.size(0)/2
            #print target[0]
            assert target[0] == 1
            assert target[k] == 0
            hinge_loss = (output[k:]+0.25-output[:k]).clamp(min=0.0)
            return hinge_loss.mean()
        else:
            raise Excpetion("Unknown criterion: {}".format(crt))


