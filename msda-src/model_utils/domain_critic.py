
import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F

from flip_gradient import flip_gradient

from functools import partial
import utils

class ClassificationD(nn.Module):
    @staticmethod
    def add_config(cfgparser):
        return

    def __init__(self, encoder, configs):
        super(ClassificationD, self).__init__()
        self.n_in = encoder.n_out
        self.n_out = 100

        self.seq = nn.Sequential(
            nn.Linear(self.n_in, self.n_out),
            nn.ReLU(),
            # nn.Linear(self.n_out, self.n_out),
            # nn.ReLU(),
            nn.Linear(self.n_out, 2)
        )

        for module in self.seq:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal(module.weight)
                nn.init.constant(module.bias, 0.1)
                # module.bias.requires_grad = False

    def forward(self, x):
        x_grl = flip_gradient(x)
        return self.seq(x_grl)

    def compute_loss(self, output, labels):
        return nn.functional.cross_entropy(output, labels)

class MMD(nn.Module):
    @staticmethod
    def add_config(cfgparser):
        return

    def __init__(self, encoder, configs):
        super(MMD, self).__init__()
        self.sigmas = torch.FloatTensor([
            1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1,
            1, 5, 10, 15, 20, 25, 30, 35, 100,
            1e3, 1e4, 1e5, 1e6
        ])
        if configs.cuda:
            self.sigmas = self.sigmas.cuda()
        self.gaussian_kernel = partial(utils.gaussian_kernel_matrix, sigmas=self.sigmas)

    def forward(self, hs, ht):
        loss_value = utils.maximum_mean_discrepancy(hs, ht, kernel=self.gaussian_kernel)
        return torch.clamp(loss_value, min=1e-4)

class CoralD(nn.Module):
    @staticmethod
    def add_config(cfgparser):
        return

    def __init__(self, encoder, configs):
        super(CoralD, self).__init__()

    def forward(self, hs, ht):
        _D_s = torch.sum(hs, 0, keepdim=True)
        _D_t = torch.sum(ht, 0, keepdim=True)
        C_s = (torch.matmul(hs.t(), hs) - torch.matmul(_D_s.t(), _D_s) / hs.shape[0]) / (hs.shape[0] - 1)
        C_t = (torch.matmul(ht.t(), ht) - torch.matmul(_D_t.t(), _D_t) / ht.shape[0]) / (ht.shape[0] - 1)
        loss_value = torch.sum((C_s - C_t) * (C_s - C_t)) # can use `norm'
        return loss_value

class WassersteinD(nn.Module):
    @staticmethod
    def add_config(cfgparser):
        return

    def __init__(self, encoder, configs):
        super(WassersteinD, self).__init__()
        if configs.cond == "concat": # concatenation or outer product
            self.n_in = encoder.n_out + 2
        elif configs.cond == "outer":
            self.n_in = encoder.n_out * 2
        else:
            self.n_in = encoder.n_out
        self.n_out = 100
        # self.cuda = 1 if configs.cuda else 0
        # self.lambda_gp = configs.lambda_gp

        self.seq = nn.Sequential(
            nn.Linear(self.n_in, self.n_out),
            nn.ReLU(),
            # nn.Linear(self.n_out, self.n_out),
            # nn.ReLU(),
            nn.Linear(self.n_out, 1)
        )

        for module in self.seq:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal(module.weight)
                nn.init.constant(module.bias, 0.1)

    def forward(self, hs, ht):
        # self.clip_weights()
        alpha = torch.FloatTensor(hs.shape[0], 1).uniform_(0., 1.)
        # if self.cuda:
        alpha = alpha.cuda()
        alpha = Variable(alpha)

        hdiff = hs - ht
        interpolates = ht + (alpha * hdiff)
        # interpolates.requires_grad = True
        h_whole = torch.cat([hs, ht, interpolates])
        wd_loss = self.seq(hs).mean() - self.seq(ht).mean()

        # gradient penalty
        # Ref implementation:
        # - https://github.com/caogang/wgan-gp/blob/master/gan_mnist.py
        disc_whole = self.seq(h_whole)
        gradients = autograd.grad(outputs=disc_whole, inputs=h_whole,
                                  grad_outputs=torch.ones(disc_whole.size()).cuda(), \
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradient_panelty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return (wd_loss, gradient_panelty)

    # not as good as gradient penalty
    def clip_weights(self, val_range=0.01):
        for p in self.parameters():
            p.data.clamp_(min=-val_range, max=val_range)

