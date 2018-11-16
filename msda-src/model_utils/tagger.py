import random
import sys
import argparse
import numpy as np

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
from .basic import ModelBase


class Tagger(ModelBase):

    @staticmethod
    def add_config(cfgparser):
        super(Tagger, Tagger).add_config(cfgparser)


    def get_var(self, x, volatile=False):
        x = Variable(x, volatile=volatile)
        return x.cuda() if self.args.CUDA else x


    def __init__(self, args, vw, vc, vt, wc, UNK, CUNK, pad_char):
        super(Tagger, self).__init__(args)
        WEMBED_SIZE = args.WEMBED_SIZE
        CEMBED_SIZE = args.CEMBED_SIZE
        HIDDEN_SIZE = args.HIDDEN_SIZE
        MLP_SIZE = args.MLP_SIZE
        self.args = args
        ntags = args.ntags
        nwords = args.nwords
        nchars = args.nchars

        self.vw = vw
        self.vc = vc
        self.vt = vt
        self.wc = wc
        self.UNK = UNK
        self.CUNK = CUNK
        self.pad_char = pad_char

        self.lookup_w = nn.Embedding(nwords, WEMBED_SIZE, padding_idx=self.UNK)
        self.lookup_c = nn.Embedding(nchars, CEMBED_SIZE, padding_idx=self.CUNK)
        self.lstm = nn.LSTM(WEMBED_SIZE, HIDDEN_SIZE, 1, bidirectional=True)
        self.lstm_c_f = nn.LSTM(CEMBED_SIZE, WEMBED_SIZE / 2, 1)
        self.lstm_c_r = nn.LSTM(CEMBED_SIZE, WEMBED_SIZE / 2, 1)
        #self.proj1 = nn.Linear(2 * HIDDEN_SIZE, MLP_SIZE)
        #self.proj2 = nn.Linear(MLP_SIZE, ntags)

    def forward(self, words, volatile=False):
        word_ids = []
        needs_chars = []
        char_ids = []
        for i, w in enumerate(words):
            if self.wc[w] > 5:
                word_ids.append(self.vw.w2i[w])
            else:
                word_ids.append(self.UNK)
                needs_chars.append(i)
                char_ids.append([self.pad_char] + [self.vc.w2i.get(c, self.CUNK) for c in w] \
                                                + [self.pad_char])
        embeddings = self.lookup_w(self.get_var(torch.LongTensor(word_ids), volatile=volatile))
        if needs_chars:
            max_len = max(len(x) for x in char_ids)
            fwd_char_ids = [ids + [self.pad_char \
                            for _ in range(max_len - len(ids))] for ids in char_ids]
            rev_char_ids = [ids[::-1] + [self.pad_char \
                            for _ in range(max_len - len(ids))] for ids in char_ids]
            char_embeddings = torch.cat([
                    self.lstm_c_f(self.lookup_c(self.get_var(torch.LongTensor(fwd_char_ids).t())))[0],
                    self.lstm_c_r(self.lookup_c(self.get_var(torch.LongTensor(rev_char_ids).t())))[0]
                ], 2)
            unk_embeddings = torch.cat([char_embeddings[len(words[j]) + 1, i].unsqueeze(0) for i, j in enumerate(needs_chars)], 0)
            embeddings = embeddings.index_add(0, self.get_var(torch.LongTensor(needs_chars)), unk_embeddings)
        return self.lstm(embeddings.unsqueeze(1))[0].squeeze(1)
        #return self.proj2(self.proj1(self.lstm(embeddings.unsqueeze(1))[0].squeeze(1)))
