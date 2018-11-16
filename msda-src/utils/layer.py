import sys
import gzip
import random

import numpy as np
import torch
from io import say
import torch.nn as nn

# torch.manual_seed(1234)
# random.seed(1234)

class EmbeddingLayer(object):
    def __init__(self, n_d, words, embs=None, fix_emb=True, bos='<s>', eos='</s>', oov='<oov>', pad='<pad>'):
        '''
        if not fix_emb: use the words that only appears in training data
        '''
        word2id = {}
        if embs is not None:
            embwords, embvecs = embs
            if fix_emb: # only fill word2id with words in pre-trained embeddings when fix_emb
                for word in embwords:
                    word2id[word] = len(word2id)

            say("{} pre-trained word embeddings loaded.\n".format(len(embwords)))
            if n_d != len(embvecs[0]):
                say("[WARNING] n_d ({}) != word vector size ({}). Use {} for embeddings.\n".format(
                    n_d, len(embvecs[0]), len(embvecs[0])
                ))
                n_d = len(embvecs[0])

        if words is not None: # fix_emb should be False
            assert (not fix_emb)
            for word in words:
                if word not in word2id:
                    word2id[word] = len(word2id)

        extra_tokens = [bos, eos, oov, pad]
        for tok in extra_tokens:
            if tok not in word2id:
                word2id[tok] = len(word2id)

        say("{} embedded words in total\n".format(len(word2id)))

        self.word2id = word2id
        self.n_V, self.n_d = len(word2id), n_d
        self.oovid = word2id[oov]
        self.padid = word2id[pad]
        self.embedding = nn.Embedding(self.n_V, n_d)
        self.embedding.weight.data.uniform_(-0.25, 0.25)
        self.embedding.weight.data[self.padid].zero_()
        self.embedding.weight.data[self.oovid].zero_()

        if embs is not None:
            weight = self.embedding.weight
            tor_embvecs = torch.from_numpy(embvecs)
            if fix_emb:
                weight.data[:len(embwords)].copy_(tor_embvecs)
            else:
                for word, wid in word2id.items():
                    if word in embwords:
                        # initialize with pre-trained word embeddings
                        weight.data[wid].copy_(tor_embvecs[embwords.index(word)])

        if fix_emb:
            self.embedding.weight.requires_grad = False


class EmbeddingLayerOld(object):
    def __init__(self, n_d, words, embs=None, fix_emb=True, bos='<s>', eos='</s>', oov='<oov>', pad='<pad>'):
        '''
            Note: initialization of the extra tokens: [ bos, eos, oov, pad ]
        '''
        word2id = {}
        if embs is not None:
            embwords, embvecs = embs
            for word in embwords:
                assert word not in word2id, "Duplicate words in pre-trained embeddings"
                word2id[word] = len(word2id)

            say("{} pre-trained word embeddings loaded.\n".format(len(word2id)))
            if n_d != len(embvecs[0]):
                say("[WARNING] n_d ({}) != word vector size ({}). Use {} for embeddings.\n".format(
                    n_d, len(embvecs[0]), len(embvecs[0])
                ))
                n_d = len(embvecs[0])

        # if not fix_emb:
        for word in words:
            if word not in word2id:
                word2id[word] = len(word2id)

        extra_tokens = [bos, eos, oov, pad]
        for tok in extra_tokens:
            if tok not in word2id:
                word2id[tok] = len(word2id)

        say("{} embedded words in total\n".format(len(word2id)))

        self.word2id = word2id
        self.n_V, self.n_d = len(word2id), n_d
        self.oovid = word2id[oov]
        self.padid = word2id[pad]
        self.embedding = nn.Embedding(self.n_V, n_d)

        if embs is not None:
            weight = self.embedding.weight
            weight.data[:len(embwords)].copy_(torch.from_numpy(embvecs))
            say("embedding shape: {}\n".format(weight.size()))

        if fix_emb:
            self.embedding.weight.requires_grad = False

