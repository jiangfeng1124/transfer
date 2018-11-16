# coding: utf-8
from __future__ import print_function
import time
start = time.time()

from collections import Counter, defaultdict
import random
import sys
import argparse
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
from conll_preprocess import file_conll

sys.path.append("../")
from model_utils import get_model_class
from model_utils.classifier import Classifier
from utils.op import softmax

torch.manual_seed(10)
random.seed(10)

def domain_encoding(batches, args, encoder):

    s = time.time()

    statistics = []

    domain_batches = []

    for i in range(args.num_train_files):
        domain_batch = []
        for j in range(args.train_samples):
            domain_batch.append(batches[j][i])
        domain_batches.append(domain_batch)

    for i,domain_batch in enumerate(domain_batches):
        S = []
        for batch in domain_batch:
            words, golds = zip(*batch)
            S.append(torch.mean(encoder(words), dim=0, keepdim=True))
        S = torch.cat(S,0)
        mu_S = torch.mean(S, dim=0, keepdim=True)
        cov_S = (torch.matmul((S - mu_S).t(), S - mu_S)) / (S.shape[0] - 1)
        I = torch.eye(cov_S.shape[1], cov_S.shape[1])
        if args.CUDA: I = Variable(I).cuda()
        covi_S = (cov_S + args.cov_gamma * I).inverse()
        statistics.append((mu_S, covi_S))

    return statistics


class Vocab:

    def __init__(self, w2i=None):
        if w2i is None: w2i = defaultdict(lambda: len(w2i))
        self.w2i = dict(w2i)
        self.i2w = {i: w for w, i in w2i.items()}

    @classmethod
    def from_corpus(cls, corpus):
        w2i = defaultdict(lambda: len(w2i))
        for sent in corpus:
            [w2i[word] for word in sent]
        return Vocab(w2i)

    def size(self):
        return len(self.w2i.keys())


def read(fname):
    ''' Read a POS-tagged file where each line is of the form "word1|tag2 word2|tag2 ..."
        Yields lists of the form [(word1,tag1), (word2,tag2), ...]
    '''
    with open(fname, "r") as fh:
        for line in fh:
            line = line.strip().split()
            sent = [tuple(x.rsplit("|", 1)) for x in line]
            yield sent


def evaluate(encoder, args, batch_trains, classifier, classifiers, \
             eval_sents, domain_encs):

    good_sent = bad_sent = good = bad = 0.0

    for sent in eval_sents:
        words, golds = zip(*sent)
        #alphas = [mahalanobis_metric_fast(encoder(words, \
        #                 volatile=True), domain_stats[0], domain_stats[1])\
        #                 for domain_stats in domain_encs]
        #print(alphas)
        #alphas = softmax(alphas)
        #if args.CUDA:
        #    alphas = [ alpha.cuda() for alpha in alphas ]
        #alphas = [ Variable(alpha) for alpha in alphas ]
        probs = [ ath(encoder(words, volatile=True)) for ath\
                  in classifiers ]
        #probs = [ classifier(encoder(words, volatile=True)) for ath in \
        #          classifiers ]

        #outputs = sum([ alpha.unsqueeze(1).repeat(1, args.ntags) * output_i\
        #                for (alpha, output_i) in zip(alphas, probs) ])
        outputs = sum(probs)
        tags = [encoder.vt.i2w[i] for i in \
                outputs.data.max(1)[1].cpu().view(-1)]

        if tags == list(golds): good_sent += 1
        else: bad_sent += 1
        for go, gu in zip(golds, tags):
            if go == gu: good += 1
            else: bad += 1
    print ("tag_acc=%.4f, sent_acc=%.4f" % (good/(good+bad), good_sent/(good_sent+bad_sent)))
    return (1.0*good)/(good+bad)


def mahalanobis_metric(p, S, args):

    mu_S = torch.mean(S, dim=0, keepdim=True)
    cov_S = (torch.matmul((S - mu_S).t(), S - mu_S)) 
    I = torch.eye(p.shape[1], p.shape[1])
    if args.CUDA: I = Variable(I).cuda()
    covi_S = (cov_S + args.cov_gamma * I).inverse()

    mahalanobis_distances = (p - mu_S).mm(covi_S).mm((p - mu_S).t())

    return mahalanobis_distances.diag().sqrt().data


def mahalanobis_metric_fast(p, mu, covi):

    mean_p = torch.mean(p, dim=0, keepdim=True)

    mahalanobis_distances = (mean_p - mu).mm(covi).mm((mean_p - mu).t())
    mahalanobis_distances = 1.0/mahalanobis_distances
    return mahalanobis_distances.diag().sqrt().data.expand(p.size(0))

def train_model(args):

    CEMBED_SIZE = args.CEMBED_SIZE
    WEMBED_SIZE = args.WEMBED_SIZE
    HIDDEN_SIZE = args.HIDDEN_SIZE
    MLP_SIZE = args.MLP_SIZE
    SPARSE = args.SPARSE
    TIMEOUT = args.TIMEOUT

    num_train_files = 0

    best_dev = 0.0
    best_test = 0.0

    if args.train:
        train = file_conll(args.train).tupled_data
    if args.multi_train:
        train = [ ]
        for file_name in args.multi_train:
            train  += file_conll(file_name).tupled_data[:args.train_samples]
            num_train_files += 1
        assert len(train) % args.train_samples == 0
    if args.dev:
        dev = file_conll(args.dev).tupled_data
    if args.test:
        test = file_conll(args.test).tupled_data

    args.num_train_files = num_train_files

    batch_trains = [ ]
    for i in range(args.train_samples):
        batch = [ ]
        for j in range(num_train_files):
            batch.append(train[j*args.train_samples+i])
        batch_trains.append(batch)

    words = []
    tags = []
    chars = set()
    wc = Counter()
    for sent in (train+dev+test):
        for w, p in sent:
            words.append(w)
            tags.append(p)
            wc[w] += 1
            chars.update(w)
    words.append("_UNK_")
    chars.add("_UNK_")
    chars.add("<*>")

    vw = Vocab.from_corpus([words])
    vt = Vocab.from_corpus([tags])
    vc = Vocab.from_corpus([chars])
    UNK = vw.w2i["_UNK_"]
    CUNK = vc.w2i["_UNK_"]
    pad_char = vc.w2i["<*>"]

    nwords = vw.size()
    ntags = vt.size()
    nchars = vc.size()
    print ("nwords=%r, ntags=%r, nchars=%r" % (nwords, ntags, nchars))

    args.ntags = ntags
    args.nwords = nwords
    args.nchars = nchars
    encoder_class = get_model_class("tagger")
    encoder_class.add_config(parser)
    encoder = encoder_class(args, vw, vc, vt, wc, UNK, CUNK, pad_char)

    classifier = Classifier(2*HIDDEN_SIZE, MLP_SIZE, ntags)
    classifiers = [ ]
    for ind in range(num_train_files):
        classifiers.append(Classifier(2*HIDDEN_SIZE, MLP_SIZE, ntags))

    requires_grad = lambda x : x.requires_grad

    optimizer_encoder = optim.Adam(encoder.parameters(), weight_decay = 1e-4)
    task_params = list(classifier.parameters())
    for x in classifiers:
        task_params += list(x.parameters())
    optimizer_classifier = optim.Adam(filter(requires_grad, task_params),\
                                      weight_decay = 1e-4)

    if args.CUDA:
        map(lambda m: m.cuda(), [encoder] + [classifier] + classifiers)

    print("startup time: %r" % (time.time() - start))
    start_time = time.time()

    i = 0

    for ITER in range(50):
        #random.shuffle(batch_trains)
        #encoder, classifier, optimizer_encoder, optimizer_classifier = \
        #           train_epoch(encoder, classifier, classifiers, batch_trains,\
        #           dev, test, optimizer_encoder, optimizer_classifier,\
        #           start_time, i)
        train_epoch(encoder, classifier, classifiers, batch_trains,\
                    dev, test, optimizer_encoder, optimizer_classifier, \
                    start_time, i)
        print("epoch %r finished" % ITER)
        domain_encs = None#domain_encoding(batch_trains, args, encoder)
        curr_dev = evaluate(encoder, args, batch_trains, classifier, classifiers, \
                 dev, domain_encs)
        curr_test = evaluate(encoder, args, batch_trains, classifier, classifiers, \
                 test, domain_encs)
        if curr_dev > best_dev:
            best_dev = curr_dev
            best_test = curr_test

    print(best_dev, best_test) 


def train_epoch(encoder, classifier, classifiers, batch_trains, dev, \
                test, optimizer_encoder, optimizer_classifier, start, I):

    all_time = dev_time = all_tagged = this_tagged = this_loss = 0

    mtl_criterion = nn.CrossEntropyLoss()
    moe_criterion = nn.NLLLoss()

    domain_encs = None

    for ind,batch in enumerate(batch_trains):

        optimizer_encoder.zero_grad()
        optimizer_classifier.zero_grad()
        loss_mtl = []
        loss_moe = []
        ms_outputs = []
        hiddens = []
        train_labels = []

        for source_ind,s in enumerate(batch):
            I += 1
            #if I % 200 == 1:
            #    domain_encs = domain_encoding(batch_trains, args, encoder)
            words, golds = zip(*s)
            hidden = encoder(words)
            outputs = []
            hiddens.append(hidden)

            for sthi in classifiers: 
                #output = classifier(hidden)
                output = sthi(hidden)
                outputs.append(output)

            ms_outputs.append(outputs)
            train_labels.append(encoder.get_var(torch.LongTensor\
                              ([encoder.vt.w2i[t] for t in golds])))

            preds = ms_outputs[source_ind][source_ind]
            loss = mtl_criterion(preds, \
                                 train_labels[-1])
            loss_mtl.append(loss)

        #source_ids = range(len(batch))
        #for i in source_ids:
        #    support_ids = [x for x in source_ids if x != i]
        #    support_alphas = [mahalanobis_metric_fast(hiddens[i], \
        #                       domain_encs[j][0], domain_encs[j][1])\
        #                        for j in support_ids]
        #    support_alphas = softmax(support_alphas)
        #    support_alphas = [ Variable(alpha) for alpha in support_alphas ]
        #    if args.CUDA:
        #        support_alphas = [ alpha.cuda() for alpha in support_alphas ]

        #    output_moe_i = sum([ alpha.unsqueeze(1).repeat(1, args.ntags) * \
        #                         F.softmax(ms_outputs[i][id], dim=1) \
        #                   for alpha, id in zip(support_alphas, support_ids) ])
        #    loss_moe.append(moe_criterion(torch.log(output_moe_i), \
        #                    train_labels[i]))

        loss_mtl = sum(loss_mtl)
        #loss_moe = sum(loss_moe)
        #loss = args.lambda_moe * loss_mtl + (1.0 - args.lambda_moe) * loss_moe
        loss = loss_mtl
        loss.backward()
        optimizer_encoder.step()
        optimizer_classifier.step()


    print("\n\nEnded last epoch.\n")
    print(I)

    return encoder, classifier, optimizer_encoder, optimizer_classifier


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--CEMBED_SIZE', type=int, default=100, help='char embedding size')
    parser.add_argument('--WEMBED_SIZE', type=int, default=100, help='embedding size')
    parser.add_argument('--HIDDEN_SIZE', type=int, default=100, help='hidden size')
    parser.add_argument('--MLP_SIZE', type=int, default=100, help='embedding size')
    parser.add_argument('--SPARSE', type=int, default=1, help='sparse update 0/1')
    parser.add_argument('--TIMEOUT', type=int, default=10000, help='timeout in seconds')
    parser.add_argument('--dev', type=str, required=False, help='dev file')
    parser.add_argument('--test', type=str, required=False, help='test file')
    parser.add_argument('--train', type=str, required=False, help='training file')
    parser.add_argument('--multi_train', nargs='+', required=False, \
                         help='List of multiple sources for training')
    parser.add_argument('--CUDA', default=1, type=int)
    parser.add_argument('--cov_gamma', default=1.0, type=float)
    parser.add_argument('--ntags', default=0, type=int)
    parser.add_argument('--nwords', default=0, type=int)
    parser.add_argument('--nchars', default=0, type=int)
    parser.add_argument('--UNK', default=0, type=int)
    parser.add_argument('--CUNK', default=0, type=int)
    parser.add_argument('--pad_char', default='', type=str)
    parser.add_argument('--train_samples', default=1000, type=int)
    parser.add_argument('--num_train_files', default=0, type=int)
    parser.add_argument("--lambda_moe", type=float, default=0.2)
    args = parser.parse_args()

    train_model(args)

