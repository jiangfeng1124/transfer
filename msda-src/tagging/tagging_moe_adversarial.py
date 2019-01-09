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
from model_utils import get_model_class, get_critic_class
from model_utils.domain_critic import ClassificationD, MMD, CoralD, WassersteinD
from model_utils.classifier import Classifier
from utils.op import softmax

torch.manual_seed(10)
random.seed(10)

class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        #b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = x * torch.log(x)
        b = -1.0 * b.sum()
        return b

def domain_encoding(batches, args, encoder):

    s = time.time()

    statistics = []

    domain_batches = []


    for i in range(args.num_train_files):
        domain_batch = []
        for j in range(len(batches)):
            domain_batch.append(batches[j][i])
        domain_batches.append(domain_batch)

    for i,domain_batch in enumerate(domain_batches):
        S = []
        for batch in domain_batch:
            words, golds = zip(*batch)
            S.append(torch.mean(encoder(words), dim=0, keepdim=True))
        S = torch.cat(S,0)
        mu_S = torch.mean(S, dim=0, keepdim=True)
        #cov_S = (torch.matmul((S - mu_S).t(), S - mu_S)) / (S.shape[0] - 1)
        #I = torch.eye(cov_S.shape[1], cov_S.shape[1])
        #if args.CUDA: I = Variable(I).cuda()
        #covi_S = (cov_S + args.cov_gamma * I).inverse()
        #statistics.append((mu_S, covi_S))
        statistics.append((mu_S, mu_S))

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
    """
    Read a POS-tagged file where each line is of the form "word1|tag2 word2|tag2 ..."
    Yields lists of the form [(word1,tag1), (word2,tag2), ...]
    """
    with open(fname, "r") as fh:
        for line in fh:
            line = line.strip().split()
            sent = [tuple(x.rsplit("|", 1)) for x in line]
            yield sent


def evaluate(encoder, args, batch_trains, classifier, classifiers, \
             eval_sents, domain_encs, Us):

    good_sent = bad_sent = good = bad = 0.0

    for sent in eval_sents:
        words, golds = zip(*sent)
        #print(words)
        #alphas = softmax([mahalanobis_metric_fast(encoder(words, \
        #                 volatile=True), domain_stats[0], domain_stats[1], U)\
        #                 for domain_stats,U in zip(domain_encs, Us)])
        alphas = [-mahalanobis_metric_fast(encoder(words, \
                  volatile=True), domain_stats[0], domain_stats[1], U)\
                  for domain_stats,U in zip(domain_encs, Us)]
        #alphas = softmax([ (1 - x / sum(alphas)) \
        #                   for x in alphas ])
        alphas = softmax(alphas)
        if args.CUDA:
            alphas = [ alpha.cuda() for alpha in alphas ]
        alphas = [ Variable(alpha) for alpha in alphas ]
        print(alphas)
        probs = [ classifier(encoder(words, volatile=True)) for classifier\
                  in classifiers ]
        #print(probs)
        outputs = sum([ alpha.unsqueeze(1).repeat(1, args.ntags) * output_i\
                        for (alpha, output_i) in zip(alphas, probs) ])
        #print(outputs)
        tags = [encoder.vt.i2w[i] for i in \
                outputs.data.max(1)[1].cpu().view(-1)]
        #print([encoder.vt.w2i[t] for t in tags])

        if tags == list(golds): good_sent += 1
        else: bad_sent += 1
        for go, gu in zip(golds, tags):
            if go == gu: good += 1
            else: bad += 1
    print ("tag_acc=%.4f, sent_acc=%.4f" % (good/(good+bad), good_sent/(good_sent+bad_sent)))
    return (1.0*good)/(good+bad)


def mahalanobis_metric(p, mu_S, U, args):

    #old_p_size = p.size(0)
    #p = torch.mean(p, dim=0, keepdim=True)
    #mahalanobis_distances_new = (p - mu_S).mm(U.mm(U.t())).mm((p - mu_S).t())
    #mahalanobis_distances_new = mahalanobis_distances_new.diag().sqrt().\
    #                            expand(old_p_size)
    #return mahalanobis_distances_new.clamp(0.1, 2)

    mahalanobis_distances_new = (p - mu_S).mm(U.mm(U.t())).mm((p - mu_S).t())
    mahalanobis_distances_new = mahalanobis_distances_new.diag().sqrt()
    return mahalanobis_distances_new.clamp(0.1, 2)

    cov_S = (torch.matmul((S - mu_S).t(), S - mu_S)) 
    I = torch.eye(p.shape[1], p.shape[1])
    if args.CUDA: I = Variable(I).cuda()
    covi_S = (cov_S + args.cov_gamma * I).inverse()

    mahalanobis_distances = (p - mu_S).mm(covi_S).mm((p - mu_S).t())

    #return mahalanobis_distances.diag().sqrt().data.expand(old_p_size)


def mahalanobis_metric_fast(p, mu, covi, U):

    #mean_p = torch.mean(p, dim=0, keepdim=True)
    #mahalanobis_distances_new = (mean_p - \
    #                            mu).mm(U.mm(U.t())).mm((mean_p - mu).t())
    #mahalanobis_distances_new = mahalanobis_distances_new.diag().sqrt().\
    #                            expand(p.size(0))
    #return mahalanobis_distances_new.data
    mahalanobis_distances_new = (p - mu).mm(U.mm(U.t())).mm((p - mu).t())
    mahalanobis_distances_new = mahalanobis_distances_new.diag().sqrt()
    return mahalanobis_distances_new.data

def train_model(args):

    CEMBED_SIZE = args.CEMBED_SIZE
    WEMBED_SIZE = args.WEMBED_SIZE
    HIDDEN_SIZE = args.HIDDEN_SIZE
    MLP_SIZE = args.MLP_SIZE
    SPARSE = args.SPARSE
    TIMEOUT = args.TIMEOUT

    num_train_files = 0

    Us = []
    batch_trains = []

    if args.train:
        train = file_conll(args.train).tupled_data
    if args.multi_train:
        train = [ ]
        for file_name in args.multi_train:
            if "ontonotes" in file_name:
                print(len(file_conll(file_name).tupled_data))
                train += file_conll(file_name).tupled_data[:args.num_twitter]
            else:
                train  += file_conll(file_name).\
                          tupled_data[:args.train_samples]
            num_train_files += 1
            U = torch.FloatTensor(2*args.HIDDEN_SIZE, args.m_rank)
            nn.init.xavier_uniform(U)
            Us.append(U)
        train_combined = [ ]
        train = [ ]
        max_samples = args.train_samples
        for file_name in args.multi_train:
            train += file_conll(file_name).tupled_data[:args.train_samples]
            if "ontonotes" in file_name:
                train_combined.append(file_conll(file_name).tupled_data\
                                     [:args.train_samples])
            elif "total.conllu" in file_name:
                train_combined.append(file_conll(file_name).tupled_data\
                                      [:args.num_twitter])
                max_samples = max(args.num_twitter, args.train_samples)
            else:
                train_combined.append(file_conll(file_name).tupled_data\
                                      [:args.train_samples])
        args.train_samples = max_samples
        for j in range(max_samples):
            current_batch = [ ]
            for i in range(num_train_files):
                current_batch.append(train_combined[i][j%len(train_combined[i])])
            batch_trains.append(current_batch)

    if args.dev:
        dev = file_conll(args.dev).tupled_data
    if args.test:
        test = file_conll(args.test).tupled_data
    
    if args.dev == args.test:
        print("Dividing test and dev equally")
        test = test[len(test)/2:]
        dev = dev[:len(dev)/2]

    args.num_train_files = num_train_files
    args.cuda = args.CUDA

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

    critic_class = get_critic_class(args.critic)
    critic_class.add_config(parser)
    critic = critic_class(encoder, args)


    classifiers = [ ]
    for ind in range(num_train_files):
        classifiers.append(Classifier(2*HIDDEN_SIZE, MLP_SIZE, ntags))

    requires_grad = lambda x : x.requires_grad

    if args.CUDA:
        map(lambda m: m.cuda(), [encoder] + [classifier] + classifiers + \
            [critic])
        Us = [ Variable(U.cuda(), requires_grad=True) for U in Us ]

    else:
        Us = [ Variable(U, requires_grad=True) for U in Us ]

    optimizer_encoder = optim.Adam(encoder.parameters(), lr = 1e-3,\
                                   weight_decay = 1e-4)
    task_params = list(classifier.parameters())
    for x in classifiers:
        task_params += list(x.parameters())
    task_params += Us
    task_params += list(critic.parameters())
    optimizer_classifier = optim.Adam(filter(requires_grad, task_params),
                                      lr = 1e-3, weight_decay = 1e-4)

    print("startup time: %r" % (time.time() - start))
    start_time = time.time()

    i = 0

    best_test = 0
    best_dev = 0


    for ITER in range(args.epochs):
        #random.shuffle(batch_trains)
        encoder, classifier, optimizer_encoder, optimizer_classifier = \
                   train_epoch(encoder, classifier, classifiers, critic,\
                   batch_trains, dev, test, optimizer_encoder,\
                   optimizer_classifier, start_time, i, Us)
        print("epoch %r finished" % ITER)
        domain_encs = domain_encoding(batch_trains, args, encoder)
        curr_dev = evaluate(encoder, args, batch_trains, classifier, classifiers, \
                 dev, domain_encs, Us)
        curr_test = evaluate(encoder, args, batch_trains, classifier, classifiers, \
                 test, domain_encs, Us)
        if curr_dev > best_dev:
            best_dev = curr_dev
            best_test = curr_test

    print(best_dev, best_test)


def train_epoch(encoder, classifier, classifiers, critic, batch_trains, dev, \
                test, optimizer_encoder, optimizer_classifier, start, I, Us):

    all_time = dev_time = all_tagged = this_tagged = this_loss = 0

    mtl_criterion = nn.CrossEntropyLoss()
    moe_criterion = nn.NLLLoss()
    entropy_criterion = HLoss()

    domain_encs = None

    net_encoder_grad = 0.0
    net_classifier_grad = 0.0
    net_U_grad = 0.0

    avg_encoder_norm = 0.0
    avg_classifier_norm = 0.0
    avg_U_norm = 0.0

    unlabeled_data = []
    for s in (dev+test):
        words, _ = zip(*s)
        unlabeled_data.append(words)

    unlabeled_ctr = 0

    for ind,batch in enumerate(batch_trains):

        optimizer_encoder.zero_grad()
        optimizer_classifier.zero_grad()
        loss_mtl = []
        loss_moe = []
        loss_entropy = []
        loss_dan = []
        ms_outputs = []
        hiddens = []
        avg_hiddens = []
        target_avg_hiddens = []
        train_labels = []

        for source_ind,s in enumerate(batch):
            I += 1
            if (I%args.train_samples) < 2:
                I += 2
            domain_encs = domain_encoding(batch_trains[max((I%args.train_samples)-2,0):\
                                          ((I)%args.train_samples)],args, encoder)
            words, golds = zip(*s)
            hidden = encoder(words)
            outputs = []
            hiddens.append(hidden)
            avg_hidden = torch.sum(hidden, dim=0)/hidden.size(0)
            avg_hiddens.append(avg_hidden)

            for classifier in classifiers:
                output = classifier(hidden)
                outputs.append(output)

            ms_outputs.append(outputs)
            train_labels.append(encoder.get_var(torch.LongTensor\
                              ([encoder.vt.w2i[t] for t in golds])))

            preds = ms_outputs[source_ind][source_ind]
            loss = mtl_criterion(preds, \
                                 train_labels[-1])
            loss_mtl.append(loss)


        for i in range(len(avg_hiddens)):
            unlabeled_ctr = (unlabeled_ctr+1)%len(unlabeled_data)
            target_words = unlabeled_data[unlabeled_ctr]
            target_hidden = encoder(target_words)
            target_avg = torch.sum(target_hidden, dim=0)/target_hidden.size(0)
            target_avg_hiddens.append(target_avg)

        avg_hiddens = torch.stack(avg_hiddens, dim=0)
        target_avg_hiddens = torch.stack(target_avg_hiddens, dim=0)

        source_labels = torch.LongTensor([0]*avg_hiddens.size(0))
        target_labels = torch.LongTensor([1]*target_avg_hiddens.size(0))
        if args.CUDA:
            source_labels = Variable(source_labels.cuda())
            target_labels = Variable(target_labels.cuda())
        else:
            source_labels = Variable(source_labels)
            target_labels = Variable(target_labels)

        if args.critic is not None:
            critic_label = torch.cat((source_labels, target_labels))
            if isinstance(critic, ClassificationD):
                critic_output = critic(torch.cat(avg_hiddens, target_avg_hiddens))
                loss_dan.append(critic.compute_loss(critic_output, critic_label))
            else:
                critic_output = critic(avg_hiddens, target_avg_hiddens)
                loss_dan.append(critic_output)


        source_ids = range(len(batch))
        for i in source_ids:
            support_ids = [x for x in source_ids if x != i]
            support_alphas = [-mahalanobis_metric(hiddens[i], \
                               domain_encs[j][0], Us[j], args)\
                                for j in support_ids]
            #support_alphas = [ (1 - x / sum(support_alphas)) for \
            #                           x in support_alphas ]
            support_alphas = softmax(support_alphas)

            source_alphas = [-mahalanobis_metric(hiddens[i], \
                               domain_encs[j][0], Us[j], args)\
                                for j in source_ids]
            source_alphas = softmax(source_alphas)
            if args.CUDA:
                support_alphas = [ alpha.cuda() for alpha in support_alphas ]
                source_alphas = [ alpha.cuda() for alpha in source_alphas ]

            output_moe_i = sum([ alpha.unsqueeze(1).repeat(1, args.ntags) * \
                                 F.softmax(ms_outputs[i][id], dim=1) \
                           for alpha, id in zip(support_alphas, support_ids) ])
            loss_moe.append(moe_criterion(torch.log(output_moe_i), \
                            train_labels[i]))
            source_alphas = torch.stack(source_alphas, dim=0)
            source_alphas = source_alphas.permute(1, 0)
            entropy_loss = entropy_criterion(source_alphas)
            loss_entropy.append(entropy_loss) 

        loss_mtl = sum(loss_mtl)
        loss_moe = sum(loss_moe)
        loss_dan = sum(loss_dan)
        loss_entropy = sum(loss_entropy)
        loss = (1.0 - args.lambda_moe) * loss_mtl + args.lambda_moe * loss_moe
        loss += args.lambda_critic * loss_dan
        loss += 0.0 * loss_entropy
        loss.backward()
        encoder_gradient = 0.0
        encoder_norm = 0.0
        for param in encoder.parameters():
            encoder_gradient += torch.sum(param.grad)
            encoder_norm += torch.norm(param, 2).cpu().data[0]
        classifier_gradient = 0.0
        classifier_norm = 0.0
        classifier_params = []
        for classifier in classifiers:
            classifier_params += classifier.parameters()
        for param in classifier_params:
            classifier_gradient += torch.sum(param.grad)
            classifier_norm += torch.norm(param, 2).cpu().data[0]
        u_gradient = 0.0
        u_norm = 0.0
        for U in Us:
            u_gradient += torch.sum(U.grad)
            u_norm += torch.norm(U, 2).cpu().data[0]
        net_encoder_grad += encoder_gradient.cpu().data[0]
        net_classifier_grad += classifier_gradient.cpu().data[0]
        net_U_grad += u_gradient.cpu().data[0]
        optimizer_encoder.step()
        optimizer_classifier.step()

    print("Encoder gradient ", net_encoder_grad, " classifier gradient ", \
           net_classifier_grad, " Us gradient ", net_U_grad)

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
    parser.add_argument("--critic", type=str)
    parser.add_argument('--CUDA', default=1, type=int)
    parser.add_argument('--cuda', default=1, type=int)
    parser.add_argument('--cov_gamma', default=1.0, type=float)
    parser.add_argument('--ntags', default=0, type=int)
    parser.add_argument('--nwords', default=0, type=int)
    parser.add_argument('--nchars', default=0, type=int)
    parser.add_argument('--UNK', default=0, type=int)
    parser.add_argument('--CUNK', default=0, type=int)
    parser.add_argument('--pad_char', default='', type=str)
    parser.add_argument('--train_samples', default=750, type=int)
    parser.add_argument('--num_train_files', default=0, type=int)
    parser.add_argument("--lambda_critic", type=float, default=0.2)
    parser.add_argument("--lambda_moe", type=float, default=0.2)
    parser.add_argument("--lambda_entropy", type=float, default=0.01)
    parser.add_argument("--m_rank", type=int, default=100)
    parser.add_argument("--criterion", type=str, default="")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--num_twitter", type=int, default=750)
    args = parser.parse_args()

    print(args)

    train_model(args)

## To run
# time python tagging_mop2.py --multi_train ${list-of-source-files} --dev ${dev-target-file} --test ${test-target-file}

