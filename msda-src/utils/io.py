import sys
import gzip
import random

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

from sklearn.datasets import load_svmlight_file
from collections import defaultdict

import matplotlib
matplotlib.use("Pdf")
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from matplotlib.backends.backend_pdf import PdfPages

def say(s, stream=sys.stdout):
    stream.write("{}".format(s))
    stream.flush()

def load_embedding_npz(path):
    data = np.load(path)
    return data['words'], data['vals']

def load_embedding_txt(path, dim=300):
    file_open = gzip.open if path.endswith(".gz") else open
    words = []
    vals = []

    words_selected = set()
    with file_open(path) as fin:
        for line in fin:
            line = line.strip()
            if line:
                cols = line.split()
                if len(cols) != (dim + 1):
                    continue
                word = cols[0]
                if word in words_selected:
                    continue
                words.append(word)
                words_selected.add(word)
                vals += [ float(x) for x in cols[-dim:] ]

    return words, np.asarray(vals).reshape(len(words), -1)

def load_embedding(path):
    if path.endswith(".npz"):
        return load_embedding_npz(path)
    else:
        return load_embedding_txt(path)

### use pytorch.utils.data to generate data batches
def pad_collate_fn(batch):
    labels, inputs = zip(*batch)
    pad_token = '<pad>'
    lengths = [ len(seq) for seq in inputs ]
    max_len = max(lengths)
    inputs = [ seq + [pad_token]*(max_len-len(seq)) for seq in inputs ]

    return torch.LongTensor(labels), inputs, torch.LongTensor(lengths)

class SentiDataset(data.Dataset):
    def __init__(self, file_path, bos='<s>', eos='</s>'):
        self.file_path = file_path

        self.data = []
        self.words = {}

        self.label_dict = {1: 1, 0: 0}
        # self.label_dict = {1: 0, 2: 0, 4: 1, 5: 1}

        with open(self.file_path) as fin:
            for line in fin:
                cols = line.strip().split(' ')
                label, sent = int(cols[0]), cols[1:]
                if label not in self.label_dict:
                    continue
                assert label in self.label_dict
                # if label not in self.labels:
                #     self.labels[label] = len(self.labels)
                sent = [bos] + sent + [eos]
                self.data.append([self.label_dict[label], sent])
                for word in sent:
                    self.words[word] = self.words.setdefault(word, 0) + 1

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class SentiDomainDataset(data.Dataset):
    def __init__(self, file_path, bos='<s>', eos='</s>', domain=0):
        self.file_path = file_path

        self.data = []
        self.words = {}

        self.label_dict = {-1} # unlabeled data
        # self.label_dict = {1, 2, 4}
        # self.label_dict = {1: 0, 2: 0, 4: 1}
        with open(self.file_path) as fin:
            for line in fin:
                cols = line.strip().split(' ')
                label, sent = int(cols[0]), cols[1:]
                if label not in self.label_dict:
                    continue
                assert label in self.label_dict
                sent = [bos] + sent + [eos]
                ### use domain label instead
                self.data.append([int(domain), sent])
                for word in sent:
                    self.words[word] = self.words.setdefault(word, 0) + 1

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class SentiTfidfDataset(data.Dataset):
    ''' Load text datasets and convert into tf-idf representation '''
    # for tf-idf representation, bos/eos are not necessary
    def __init__(self, file_path, vectorizer):
        self.file_path = file_path

        self.corpus = []
        self.labels = []
        # self.words = {}
        # self.label_dict = {1: 0, 2: 0, 4: 1, 5: 1}
        self.label_dict = {1: 0, 2: 1, 4: 2, 5: 3}
        assert (vectorizer is not None)
        with open(self.file_path) as fin:
            for line in fin:
                cols = line.strip().split(' ')
                label, sent = int(cols[0]), cols[1:]
                if label not in self.label_dict:
                    continue
                assert label in self.label_dict
                self.corpus.append(' '.join(sent))
                self.labels.append(self.label_dict[label])
        data = vectorizer.transform(self.corpus).toarray().astype(np.float32)
        self.X = torch.from_numpy(data)
        self.Y = torch.LongTensor(self.labels)
        # assert (len(self.data) == len(self.labels))
        # self.data = zip(self.labels, self.data)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.X.shape[0]

class SentiDomainTfidfDataset(data.Dataset):
    def __init__(self, file_path, vectorizer, domain=0):
        self.file_path = file_path

        self.corpus = []
        self.labels = []
        # self.words = {}
        self.label_dict = [1, 2, 4, 5]
        assert (vectorizer is not None)
        with open(self.file_path) as fin:
            for line in fin:
                cols = line.strip().split(' ')
                label, sent = int(cols[0]), cols[1:]
                if label not in self.label_dict:
                    continue
                assert label in self.label_dict
                self.corpus.append(' '.join(sent))
                self.labels.append(int(domain))
        data = vectorizer.transform(self.corpus).toarray().astype(np.float32)
        self.X = torch.from_numpy(data)
        self.Y = torch.LongTensor(self.labels)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.X.shape[0]

class AmazonDataset(data.Dataset):
    ''' Load svmlight-formated datasets '''
    def __init__(self, file_path):
        self.file_path = file_path
        X, Y = load_svmlight_file(file_path) # X is a sparse matrix
        # L = [X[i].nonzero()[0].shape[0] for i in range(X.shape[0])]
        X = X.todense().astype(np.float32)
        Y = np.array((Y + 1) / 2, dtype=int)
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.X.shape[0]

class AmazonDomainDataset(data.Dataset):
    ''' Load svmlight-formated datasets, using domain as labels '''
    def __init__(self, file_path, domain=0):
        self.file_path = file_path
        X, Y = load_svmlight_file(file_path) # Y is synthetic label, not used
        X = X.todense().astype(np.float32)
        self.X = torch.from_numpy(X)
        self.Y = torch.LongTensor([domain] * self.X.shape[0])

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.X.shape[0]

class SentimentCorpusLoader(object):
    ''' Assume each sentence per line
        @ replaced with pytorch.utils.data
    '''
    def __init__(self, file_path, batch_size, bos='<s>', eos='</s>', shuffle=True):
        # self.file_paths = file_paths
        self.file_path = file_path
        self.batch_size = batch_size
        self.pos = 0
        self.data = []
        self.words = {}

        with open(self.file_path) as fin:
            for line in fin:
                cols = line.strip().split(' ')
                label, sent = cols[0], cols[1:]
                sent = [bos] + sent + [eos]
                self.data.append([int(label), sent])
                for word in sent:
                    self.words[word] = self.words.setdefault(word, 0) + 1

        if shuffle:
            random.shuffle(self.data)
        self.tot = len(self.data)

    def shuffle(self):
        random.shuffle(self.data)

    def __iter__(self):
        pos, tot, batch_size = self.pos, self.tot, self.batch_size
        while pos < tot:
            yield self.data[pos:pos+batch_size]
            pos += batch_size

class SentimentCorpus(object):
    '''
    Sentence-level corpus for sentiment classification
    # unused yet
    '''
    def __init__(self, file_path, bos='<s>', eos='</s>'):

        self.file_path = file_path
        self.data = []
        self.words = {}

        with open(self.file_path) as fin:
            for line in fin:
                cols = line.strip().split(' ')
                label, sent = cols[0], cols[1:]
                sent = [bos] + sent + [eos]
                self.data.append([int(label), sent])
                for word in sent:
                    self.words[word] = self.words.setdefault(word, 0) + 1

def deep_iter(x):
    if isinstance(x, list) or isinstance(x, tuple):
        for u in x:
            for v in deep_iter(u):
                yield v
    else:
        yield x

def pad(sequences, pad_token='<pad>', pad_left=False):
    max_len = max(len(seq) for seq in sequences)
    if pad_left:
        return [ [pad_token] * (max_len - len(seq)) + seq for seq in sequences ]
    return [ seq + [pad_token] * (max_len - len(seq)) for seq in sequences ]

def make_batch(emblayer, sequences, oov='<oov>'):
    # sequences = pad(sequences, pad_left = pad_left)
    batch_size = len(sequences)
    length = len(sequences[0])
    word2id, oovid = emblayer.word2id, emblayer.oovid
    data = torch.LongTensor([ word2id.get(w, word2id.get(w.lower(), oovid)) \
                                for seq in sequences for w in seq ])
    assert data.size(0) == batch_size * length

    return data.view(batch_size, length).t().contiguous() # length * batch_size

def pad_iter(emblayer, batch_loader):
    # batchify = lambda sequences : make_batch(emblayer, sequences, pad_left = pad_left)
    for smps in batch_loader:
        # labels = [ smp[0] for smp in smps ]
        # inputs = [ smp[1] for smp in smps ]
        # inputs = map(batchify, inputs)
        labels, inputs, lengths = smps
        inputs = make_batch(emblayer, inputs)

        yield (inputs, torch.LongTensor(labels))

class CrossDomainSentiLoader(object):
    def __init__(self, src_file_path, tgt_file_path, batch_size, bos = '<s>', eos = '</s>', shuffle = True):
        self.batch_size = batch_size
        self.src_file_path = src_file_path
        self.tgt_file_path = tgt_file_path
        self.words = {}

        self.src_data = self.read_domain_corpus(self.src_file_path, bos, eos, 1)
        self.tgt_data = self.read_domain_corpus(self.tgt_file_path, bos, eos, 0)

        if shuffle:
            random.shuffle(self.src_data)
            random.shuffle(self.tgt_data)

        self.src_tot = len(self.src_data)
        self.tgt_tot = len(self.tgt_data)

        assert(self.src_tot >= self.batch_size and self.tgt_tot >= self.batch_size)

    def read_domain_corpus(self, path, bos, eos, label):
        data = []
        with open(path) as fin:
            for line in fin:
                cols = line.strip().split(' ')
                sent = cols[1:]
                sent = [bos] + sent + [eos]
                data.append([int(label), sent])
                for word in sent:
                    self.words[word] = self.words.setdefault(word, 0) + 1

        return data

    def __iter__(self):
        batch_size = self.batch_size
        while True:
            yield ( random.sample(self.src_data, batch_size),
                    random.sample(self.tgt_data, batch_size) )

def cross_pad_iter(emblayer, batch_loader, cross_domain_batch_loader, pad_left=False):
    for task_smps, domain_d_smps in zip(batch_loader, cross_domain_batch_loader):
        task_labels = [ smp[0] for smp in task_smps ]
        task_inputs = [ smp[1] for smp in task_smps ]
        task_inputs = make_batch(emblayer, task_inputs, pad_left = pad_left)

        src_domain_d_smps, tgt_domain_d_smps = domain_d_smps
        src_domain_d_labels = [ smp[0] for smp in src_domain_d_smps ]
        src_domain_d_inputs = [ smp[1] for smp in src_domain_d_smps ]
        tgt_domain_d_labels = [ smp[0] for smp in tgt_domain_d_smps ]
        tgt_domain_d_inputs = [ smp[1] for smp in tgt_domain_d_smps ]

        domain_d_labels = src_domain_d_labels + tgt_domain_d_labels
        domain_d_inputs = src_domain_d_inputs + tgt_domain_d_inputs
        domain_d_inputs = make_batch(emblayer, domain_d_inputs, pad_left = pad_left)

        yield (task_inputs, torch.LongTensor(task_labels),
                domain_d_inputs, torch.LongTensor(domain_d_labels))

def plot_embedding(X, y, source_num, file_name):
    # positive_colors = ['red', 'tan', 'sandybrown']
    # negative_colors = ['blue', 'orange', 'black']
    pp = PdfPages(file_name)
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    plt.figure(figsize=(5, 5))

    xs = X[:source_num]
    xt = X[source_num:]
    ys = y[:source_num]
    yt = y[source_num:]

    index_neg_s = np.where(ys == 0)[0]
    index_pos_s = np.where(ys == 1)[0]
    xs_pos = xs[index_pos_s]
    xs_neg = xs[index_neg_s]
    plt.scatter(xs_pos[:, 0], xs_pos[:, 1], color='red', alpha=0.7, s=5, label='xs positive samples')
    plt.scatter(xs_neg[:, 0], xs_neg[:, 1], color='blue', alpha=0.7, s=5, label='xs negative samples')

    index_neg_t = np.where(yt == 0)[0]
    index_pos_t = np.where(yt == 1)[0]
    xt_pos = xt[index_pos_t]
    xt_neg = xt[index_neg_t]
    plt.scatter(xt_pos[:, 0], xt_pos[:, 1], color='purple', alpha=0.7, s=5, label='xt positive samples')
    plt.scatter(xt_neg[:, 0], xt_neg[:, 1], color='green', alpha=0.7, s=5, label='xt negative samples')

    pp.savefig()
    pp.close()

def ms_plot_embedding_sep(X, y, source_num, file_name):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    markers = ['o', '.', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd']
    xss = []
    yss = []
    start = 0
    for i, source in enumerate(source_num):
        xs = X[start:(start+source)]
        ys = y[start:(start+source)]
        start += source
        xss.append(xs)
        yss.append(ys)

    xt = X[start:]
    yt = y[start:]

    for i, (xs, ys) in enumerate(zip(xss, yss)):
        pp = PdfPages("%s.%d.pdf" % (file_name, i))
        plt.figure(figsize=(5, 5))

        index_neg_s = np.where(ys == 0)[0]
        index_pos_s = np.where(ys == 1)[0]
        xs_pos = xs[index_pos_s]
        xs_neg = xs[index_neg_s]
        plt.scatter(xs_pos[:, 0], xs_pos[:, 1], color='red', marker=markers[i], alpha=0.7, s=2, label='xs positive samples')
        plt.scatter(xs_neg[:, 0], xs_neg[:, 1], color='blue', marker=markers[i], alpha=0.7, s=2, label='xs negative samples')

        index_neg_t = np.where(yt == 0)[0]
        index_pos_t = np.where(yt == 1)[0]
        xt_pos = xt[index_pos_t]
        xt_neg = xt[index_neg_t]

        plt.scatter(xt_pos[:, 0], xt_pos[:, 1], color='purple', alpha=0.7, s=2, label='xt positive samples')
        plt.scatter(xt_neg[:, 0], xt_neg[:, 1], color='green', alpha=0.7, s=2, label='xt negative samples')

        pp.savefig()
        pp.close()

def ms_plot_embedding_uni(X, y, source_num, file_name):
    # positive_colors = ['red', 'tan', 'sandybrown']
    # negative_colors = ['blue', 'orange', 'black']
    pp = PdfPages(file_name)
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    plt.figure(figsize=(5, 5))

    markers = ['o', '.', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd']
    xss = []
    yss = []
    start = 0
    for i, source in enumerate(source_num):
        xs = X[start:(start+source)]
        ys = y[start:(start+source)]
        start += source
        xss.append(xs)
        yss.append(ys)

    xt = X[start:]
    yt = y[start:]

    for i, (xs, ys) in enumerate(zip(xss, yss)):
        index_neg_s = np.where(ys == 0)[0]
        index_pos_s = np.where(ys == 1)[0]
        xs_pos = xs[index_pos_s]
        xs_neg = xs[index_neg_s]
        plt.scatter(xs_pos[:, 0], xs_pos[:, 1], color='red', marker=markers[i], alpha=0.7, s=5, label='xs positive samples')
        plt.scatter(xs_neg[:, 0], xs_neg[:, 1], color='blue', marker=markers[i], alpha=0.7, s=5, label='xs negative samples')

        index_neg_t = np.where(yt == 0)[0]
        index_pos_t = np.where(yt == 1)[0]
        xt_pos = xt[index_pos_t]
        xt_neg = xt[index_neg_t]

        plt.scatter(xt_pos[:, 0], xt_pos[:, 1], color='purple', alpha=0.7, s=5, label='xt positive samples')
        plt.scatter(xt_neg[:, 0], xt_neg[:, 1], color='green', alpha=0.7, s=5, label='xt negative samples')

    pp.savefig()
    pp.close()


