import sys, os, glob
import argparse
import time
import random
from copy import copy, deepcopy
from termcolor import colored, cprint

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data as data

sys.path.append('../')
from model_utils import get_model_class, get_critic_class
from model_utils.domain_critic import ClassificationD, MMD, CoralD, WassersteinD
from utils.io import AmazonDataset, AmazonDomainDataset
from utils.io import say
from utils.op import softmax

from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE

class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        #b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = x * torch.log(x)
        b = -1.0 * b.sum()
        return b

class L1Norm(nn.Module):
    def __init__(self):
        super(L1Norm, self).__init__()

    def forward(self, x):
        return torch.norm(x, 1, 1).sum()

def domain_encoding(loaders, args, encoder):
    ''' Compute the encoding of domains, each domain is represented as its mean vector
        Note: the covariance inverse matrix is learned
    '''
    statistics = []

    for loader in loaders:
        ind = 0
        labels = None
        S = []
        for batch, label in loader:
            if args.cuda:
                batch = Variable(batch.cuda())
            S.append(encoder(batch))
            if ind == 0:
                labels = label
            else:
                labels = torch.cat((labels, label), dim=0)
            ind += 1

        S = torch.cat(S, 0)
        neg_index = ((labels==0).nonzero())
        pos_index = ((labels==1).nonzero())
        neg_index = Variable(neg_index.expand(neg_index.size(0), S.size(1)))
        pos_index = Variable(pos_index.expand(pos_index.size(0), S.size(1)))
        if args.cuda:
            pos_index = pos_index.cuda()
            neg_index = neg_index.cuda()

        pos_S = torch.gather(S, 0, pos_index)
        neg_S = torch.gather(S, 0, neg_index)
        pos_mu_S = torch.mean(pos_S, dim=0, keepdim=True)
        neg_mu_S = torch.mean(neg_S, dim=0, keepdim=True) 
        mu_S = torch.mean(S, dim=0, keepdim=True)

        statistics.append((mu_S, pos_mu_S, neg_mu_S))

    return statistics

TEMPERATURE=4
def mahalanobis_metric_fast(p, mu, U, pos_mu, pos_U, neg_mu, neg_U):
    # covi = (cov + I).inverse()
    mahalanobis_distances = (p - mu).mm(U.mm(U.t())).mm((p - mu).t())
    pos_mahalanobis_distance = (p - pos_mu).mm(pos_U.mm(pos_U.t())).mm((p - pos_mu).t()).diag().sqrt().data
    neg_mahalanobis_distance = (p - neg_mu).mm(neg_U.mm(neg_U.t())).mm((p - neg_mu).t()).diag().sqrt().data
    mahalanobis_ratio1 = pos_mahalanobis_distance - neg_mahalanobis_distance
    mahalanobis_ratio2 = neg_mahalanobis_distance - pos_mahalanobis_distance
    max_ratio = torch.max(mahalanobis_ratio1, mahalanobis_ratio2)

    return max_ratio # / TEMPERATURE
    # return mahalanobis_distances.diag().sqrt().data

def mahalanobis_metric(p, S, L, U, pos_U, neg_U, args, encoder = None):
    r''' Compute the mahalanobis distance between the encoding of a sample (p) and a set (S).

    Args:
        p: tensor (batch_size, dim), a batch of samples
        S: tensor (size, dim), a domain which contains a set of samples
        encoder: a module used for encoding p and S

    Return:
        mahalanobis_distances: tensor (batch_size)
    '''

    if encoder is not None:
        p = encoder(p) # (batch_size, dim)
        S = encoder(S) # (size, dim)

    neg_index = ((L==0).nonzero())
    pos_index = ((L==1).nonzero())

    neg_index = neg_index.expand(neg_index.size(0), S.data.size(1))
    pos_index = pos_index.expand(pos_index.size(0), S.data.size(1))

    neg_S = torch.gather(S, 0, neg_index)
    pos_S = torch.gather(S, 0, pos_index)
    neg_mu = torch.mean(neg_S, dim=0, keepdim=True)
    pos_mu = torch.mean(pos_S, dim=0, keepdim=True)

    pos_mahalanobis_distance = (p - pos_mu).mm(pos_U.mm(pos_U.t())).mm((p - pos_mu).t()).diag().sqrt()
    neg_mahalanobis_distance = (p - neg_mu).mm(neg_U.mm(neg_U.t())).mm((p - neg_mu).t()).diag().sqrt()

    mahalanobis_ratio1 = pos_mahalanobis_distance - neg_mahalanobis_distance
    mahalanobis_ratio2 = neg_mahalanobis_distance - pos_mahalanobis_distance

    max_ratio = torch.max(mahalanobis_ratio1, mahalanobis_ratio2)

    return max_ratio.clamp(0.01, 2) # / TEMPERATURE # .clamp(0.001, 1)

    # mu_S = torch.mean(S, dim=0, keepdim=True) # (1, dim)
    # mahalanobis_distances = (p - mu_S).mm(U.mm(U.t())).mm((p - mu_S).t())
    # return mahalanobis_distances.diag().sqrt().clamp(0.01, 2)

def biaffine_metric_fast(p, mu, U):
    biaffine_distances = p.mm(U).mm(mu.t())
    return biaffine_distances.squeeze(1).data

def biaffine_metric(p, S, U, W, V, args, encoder = None):
    ''' Compute the biaffine distance between the encoding of a sample (p) and a set (S).

    Args:
        p: tensor (batch_size, dim), a batch of samples
        U: matrix (dim, dim)
        S: tensor (size, dim), a domain which contains a set of samples
        encoder: a module used for encoding p and S

    Return:
        biaffine_distance: tensor (batch_size)
    '''

    if encoder is not None:
        p = encoder(p)
        S = encoder(S)

    mu_S = torch.mean(S, dim=0, keepdim=True)
    biaffine_distances = p.mm(U).mm(mu_S.t()) + p.mm(W) + mu_S.mm(V) # extra components
    return biaffine_distances.squeeze(1).clamp(-10, 10)

DATA_DIR = "../../msda-data/amazon/chen12"

def train_epoch(iter_cnt, encoder, classifiers, critic, mats, data_loaders, args, optim_model):
    map(lambda m: m.train(), [encoder, critic] + classifiers)

    train_loaders, unl_loader, valid_loader = data_loaders
    dup_train_loaders = deepcopy(train_loaders)

    mtl_criterion = nn.CrossEntropyLoss()
    moe_criterion = nn.NLLLoss() # with log_softmax separated
    kl_criterion = nn.MSELoss()
    entropy_criterion = HLoss()

    if args.metric == "biaffine":
        metric = biaffine_metric
        Us, Ws, Vs = mats
    else:
        metric = mahalanobis_metric
        Us, Ps, Ns = mats

    for batches, unl_batch in zip(zip(*train_loaders), unl_loader):
        train_batches, train_labels = zip(*batches)
        unl_critic_batch, unl_critic_label = unl_batch

        iter_cnt += 1
        if args.cuda:
            train_batches = [ batch.cuda() for batch in train_batches ]
            train_labels = [ label.cuda() for label in train_labels ]

            unl_critic_batch = unl_critic_batch.cuda()
            unl_critic_label = unl_critic_label.cuda()

        train_batches = [ Variable(batch) for batch in train_batches ]
        train_labels = [ Variable(label) for label in train_labels ]
        unl_critic_batch = Variable(unl_critic_batch)
        unl_critic_label = Variable(unl_critic_label)

        optim_model.zero_grad()
        loss_mtl = []
        loss_moe = []
        loss_kl = []
        loss_entropy = []
        loss_dan = []

        ms_outputs = []  # (n_sources, n_classifiers)
        hiddens = []
        hidden_corresponding_labels = []
        # labels = []
        for i, (batch, label) in enumerate(zip(train_batches, train_labels)):
            hidden = encoder(batch)
            outputs = []
            # create output matrix:
            #     - (i, j) indicates the output of i'th source batch using j'th classifier
            hiddens.append(hidden)
            for classifier in classifiers:
                output = classifier(hidden)
                outputs.append(output)
            ms_outputs.append(outputs)
            hidden_corresponding_labels.append(label)
            # multi-task loss
            loss_mtl.append(mtl_criterion(ms_outputs[i][i], label))
            # labels.append(label)

            if args.lambda_critic > 0:
                # critic_batch = torch.cat([batch, unl_critic_batch])
                critic_label = torch.cat([1 - unl_critic_label, unl_critic_label])
                # critic_label = torch.cat([1 - unl_critic_label] * len(train_batches) + [unl_critic_label])

                if isinstance(critic, ClassificationD):
                    critic_output = critic(torch.cat(hidden, encoder(unl_critic_batch)))
                    loss_dan.append(critic.compute_loss(critic_output, critic_label))
                else:
                    critic_output = critic(hidden, encoder(unl_critic_batch))
                    loss_dan.append(critic_output)

                    # critic_output = critic(torch.cat(hiddens), encoder(unl_critic_batch))
                    # loss_dan = critic_output
            else:
                loss_dan = Variable(torch.FloatTensor([0]))

        # assert (len(outputs) == len(outputs[0]))
        source_ids = range(len(train_batches))
        for i in source_ids:

            support_ids = [ x for x in source_ids if x != i ] # experts

            # support_alphas = [ metric(
            #                      hiddens[i],
            #                      hiddens[j].detach(),
            #                      hidden_corresponding_labels[j],
            #                      Us[j], Ps[j], Ns[j],
            #                      args) for j in support_ids ]

            if args.metric == "biaffine":
                source_alphas = [ metric(hiddens[i],
                                         hiddens[j].detach(),
                                         Us[0], Ws[0], Vs[0], # for biaffine metric, we use a unified matrix
                                         args) for j in source_ids ]
            else:
                source_alphas = [ metric(hiddens[i],
                                         hiddens[j].detach(),
                                         hidden_corresponding_labels[j],
                                         Us[j], Ps[j], Ns[j],
                                         args) for j in source_ids ]

            support_alphas = [ source_alphas[x] for x in support_ids ]

            # print torch.cat([ x.unsqueeze(1) for x in support_alphas ], 1)
            support_alphas = softmax(support_alphas)

            # meta-supervision: KL loss over \alpha and real source
            source_alphas = softmax(source_alphas) # [ 32, 32, 32 ]
            source_labels = [ torch.FloatTensor([x==i]) for x in source_ids ] # one-hot
            if args.cuda:
                source_alphas = [ alpha.cuda() for alpha in source_alphas ]
                source_labels = [ label.cuda() for label in source_labels ]

            source_labels = Variable(torch.stack(source_labels, dim=0)) # 3*1
            source_alphas = torch.stack(source_alphas, dim=0)
            source_labels = source_labels.expand_as(source_alphas).permute(1,0)
            source_alphas = source_alphas.permute(1,0)
            loss_kl.append(kl_criterion(source_alphas, source_labels))

            # entropy loss over \alpha
            # entropy_loss = entropy_criterion(torch.stack(support_alphas, dim=0).permute(1, 0))
            # print source_alphas
            loss_entropy.append(entropy_criterion(source_alphas))

            output_moe_i = sum([ alpha.unsqueeze(1).repeat(1, 2) * F.softmax(ms_outputs[i][id], dim=1) \
                                    for alpha, id in zip(support_alphas, support_ids) ])
            # output_moe_full = sum([ alpha.unsqueeze(1).repeat(1, 2) * F.softmax(ms_outputs[i][id], dim=1) \
            #                         for alpha, id in zip(full_alphas, source_ids) ])

            loss_moe.append(moe_criterion(torch.log(output_moe_i), train_labels[i]))
            # loss_moe.append(moe_criterion(torch.log(output_moe_full), train_labels[i]))

        loss_mtl = sum(loss_mtl)
        loss_moe = sum(loss_moe)
        # if iter_cnt < 400:
        #     lambda_moe = 0
        #     lambda_entropy = 0
        # else:
        lambda_moe = args.lambda_moe
        lambda_entropy = args.lambda_entropy
        # loss = (1 - lambda_moe) * loss_mtl + lambda_moe * loss_moe
        loss = loss_mtl + lambda_moe * loss_moe
        loss_kl = sum(loss_kl)
        loss_entropy = sum(loss_entropy)
        loss += args.lambda_entropy * loss_entropy

        if args.lambda_critic > 0:
            loss_dan = sum(loss_dan)
            loss += args.lambda_critic * loss_dan

        loss.backward()
        optim_model.step()

        if iter_cnt % 30 == 0:
            # [(mu_i, covi_i), ...]
            # domain_encs = domain_encoding(dup_train_loaders, args, encoder)
            if args.metric == "biaffine":
                mats = [Us, Ws, Vs]
            else:
                mats = [Us, Ps, Ns]

            (curr_dev, oracle_curr_dev), confusion_mat = evaluate(
                    encoder, classifiers,
                    mats,
                    [dup_train_loaders, valid_loader],
                    args
                )

            # say("\r" + " " * 50)
            # TODO: print train acc as well
            say("{} MTL loss: {:.4f}, MOE loss: {:.4f}, DAN loss: {:.4f}, "
                "loss: {:.4f}, dev acc/oracle: {:.4f}/{:.4f}\n"
                .format(iter_cnt,
                        loss_mtl.data[0],
                        loss_moe.data[0],
                        loss_dan.data[0],
                        loss.data[0],
                        curr_dev,
                        oracle_curr_dev
            ))

    say("\n")
    return iter_cnt

def compute_oracle(outputs, label, args):
    ''' Compute the oracle accuracy given outputs from multiple classifiers
    '''
    oracle = torch.ByteTensor([0] * label.shape[0])
    if args.cuda:
        oracle = oracle.cuda()
    for i, output in enumerate(outputs):
        pred = output.data.max(dim=1)[1]
        oracle |= pred.eq(label)
    return oracle

def evaluate(encoder, classifiers, mats, loaders, args):
    ''' Evaluate model using MOE
    '''
    map(lambda m: m.eval(), [encoder] + classifiers)

    if args.metric == "biaffine":
        Us, Ws, Vs = mats
    else:
        Us, Ps, Ns = mats

    source_loaders, valid_loader = loaders
    domain_encs = domain_encoding(source_loaders, args, encoder)

    oracle_correct = 0
    correct = 0
    tot_cnt = 0
    y_true = []
    y_pred = []

    for batch, label in valid_loader:
        if args.cuda:
            batch = batch.cuda()
            label = label.cuda()

        batch = Variable(batch)
        hidden = encoder(batch)
        source_ids = range(len(domain_encs))
        if args.metric == "biaffine":
            alphas = [ biaffine_metric_fast(hidden, mu[0], Us[0]) \
                       for mu in domain_encs ]
        else:
            alphas = [ mahalanobis_metric_fast(hidden, mu[0], U, mu[1], P, mu[2], N) \
                       for (mu, U, P, N) in zip(domain_encs, Us, Ps, Ns) ]
        # alphas = [ (1 - x / sum(alphas)) for x in alphas ]
        alphas = softmax(alphas)
        if args.cuda:
            alphas = [ alpha.cuda() for alpha in alphas ]
        alphas = [ Variable(alpha) for alpha in alphas ]

        outputs = [ F.softmax(classifier(hidden), dim=1) for classifier in classifiers ]
        output = sum([ alpha.unsqueeze(1).repeat(1, 2) * output_i \
                        for (alpha, output_i) in zip(alphas, outputs) ])
        pred = output.data.max(dim=1)[1]
        oracle_eq = compute_oracle(outputs, label, args)

        if args.eval_only:
            for i in range(batch.shape[0]):
                for j in range(len(alphas)):
                    say("{:.4f}: [{:.4f}, {:.4f}], ".format(
                        alphas[j].data[i], outputs[j].data[i][0], outputs[j].data[i][1])
                    )
                oracle_TF = "T" if oracle_eq[i] == 1 else colored("F", 'red')
                say("gold: {}, pred: {}, oracle: {}\n".format(label[i], pred[i], oracle_TF))
            say("\n")
            # print torch.cat(
            #         [
            #             torch.cat([ x.unsqueeze(1) for x in alphas ], 1),
            #             torch.cat([ x for x in outputs ], 1)
            #         ], 1
            #     )

        y_true += label.tolist()
        y_pred += pred.tolist()
        correct += pred.eq(label).sum()
        oracle_correct += oracle_eq.sum()
        tot_cnt += output.size(0)

    acc = float(correct) / tot_cnt
    oracle_acc = float(oracle_correct) / tot_cnt
    return (acc, oracle_acc), confusion_matrix(y_true, y_pred)

def predict(args):
    encoder, classifiers, Us, Ps, Ns = torch.load(args.load_model)
    map(lambda m: m.eval(), [encoder] + classifiers)

    # args = argparser.parse_args()
    # say(args)
    if args.cuda:
        map(lambda m: m.cuda(), [encoder] + classifiers)
        Us = [ U.cuda() for U in Us ]
        Ps = [ P.cuda() for P in Ps ]
        Ns = [ N.cuda() for N in Ns ]

    say("\nTransferring from %s to %s\n" % (args.train, args.test))
    source_train_sets = args.train.split(',')
    train_loaders = []
    for source in source_train_sets:
        filepath = os.path.join(DATA_DIR, "%s_train.svmlight" % (source))
        train_dataset = AmazonDataset(filepath)
        train_loader = data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0
        )
        train_loaders.append(train_loader)

    test_filepath = os.path.join(DATA_DIR, "%s_test.svmlight" % (args.test))
    test_dataset = AmazonDataset(test_filepath)
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    say("Corpus loaded.\n")

    mats = [Us, Ps, Ns]
    (acc, oracle_acc), confusion_mat = evaluate(
            encoder, classifiers,
            mats,
            [train_loaders, test_loader],
            args
        )
    say(colored("Test accuracy/oracle {:.4f}/{:.4f}\n".format(acc, oracle_acc), 'red'))

def train(args):
    ''' Training Strategy

    Input: source = {S1, S2, ..., Sk}, target = {T}

    Train:
        Approach 1: fix metric and learn encoder only
        Approach 2: learn metric and encoder alternatively
    '''

    # test_mahalanobis_metric() and return

    encoder_class = get_model_class("mlp")
    encoder_class.add_config(argparser)
    critic_class = get_critic_class(args.critic)
    critic_class.add_config(argparser)

    args = argparser.parse_args()
    say(args)

    # encoder is shared across domains
    encoder = encoder_class(args)

    say("Transferring from %s to %s\n" % (args.train, args.test))
    source_train_sets = args.train.split(',')
    train_loaders = []
    Us = []
    Ps = []
    Ns = []
    Ws = []
    Vs = []
    # Ms = []
    for source in source_train_sets:
        filepath = os.path.join(DATA_DIR, "%s_train.svmlight" % (source))
        assert (os.path.exists(filepath))
        train_dataset = AmazonDataset(filepath)
        train_loader = data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0
        )
        train_loaders.append(train_loader)

        if args.metric == "biaffine":
            U = torch.FloatTensor(encoder.n_d, encoder.n_d)
            W = torch.FloatTensor(encoder.n_d, 1)
            nn.init.xavier_uniform(W)
            Ws.append(W)
            V = torch.FloatTensor(encoder.n_d, 1)
            nn.init.xavier_uniform(V)
            Vs.append(V)
        else:
            U = torch.FloatTensor(encoder.n_d, args.m_rank)

        nn.init.xavier_uniform(U)
        Us.append(U)
        P = torch.FloatTensor(encoder.n_d, args.m_rank)
        nn.init.xavier_uniform(P)
        Ps.append(P)
        N = torch.FloatTensor(encoder.n_d, args.m_rank)
        nn.init.xavier_uniform(N)
        Ns.append(N)
        # Ms.append(U.mm(U.t()))

    unl_filepath = os.path.join(DATA_DIR, "%s_train.svmlight" % (args.test))
    assert (os.path.exists(unl_filepath))
    unl_dataset = AmazonDomainDataset(unl_filepath)
    unl_loader = data.DataLoader(
        unl_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    valid_filepath = os.path.join(DATA_DIR, "%s_dev.svmlight" % (args.test))
    valid_dataset = AmazonDataset(valid_filepath)
    valid_loader = data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    test_filepath = os.path.join(DATA_DIR, "%s_test.svmlight" % (args.test))
    assert (os.path.exists(test_filepath))
    test_dataset = AmazonDataset(test_filepath)
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    say("Corpus loaded.\n")

    classifiers = []
    for source in source_train_sets:
        classifier = nn.Linear(encoder.n_out, 2) # binary classification
        nn.init.xavier_normal(classifier.weight)
        nn.init.constant(classifier.bias, 0.1)
        classifiers.append(classifier)

    critic = critic_class(encoder, args)

    # if args.save_model:
    #     say(colored("Save model to {}\n".format(args.save_model + ".init"), 'red'))
    #     torch.save([encoder, classifiers, Us, Ps, Ns], args.save_model + ".init")

    if args.cuda:
        map(lambda m: m.cuda(), [encoder, critic] + classifiers)
        Us = [ Variable(U.cuda(), requires_grad=True) for U in Us ]
        Ps = [ Variable(P.cuda(), requires_grad=True) for P in Ps ]
        Ns = [ Variable(N.cuda(), requires_grad=True) for N in Ns ]
        if args.metric == "biaffine":
            Ws = [ Variable(W.cuda(), requires_grad=True) for W in Ws ]
            Vs = [ Variable(V.cuda(), requires_grad=True) for V in Vs ]

    # Ms = [ U.mm(U.t()) for U in Us ]

    say("\nEncoder: {}\n".format(encoder))
    for i, classifier in enumerate(classifiers):
        say("Classifier-{}: {}\n".format(i, classifier))
    say("Critic: {}\n".format(critic))

    requires_grad = lambda x : x.requires_grad
    task_params = list(encoder.parameters())
    for classifier in classifiers:
        task_params += list(classifier.parameters())
    task_params += list(critic.parameters())
    task_params += Us
    task_params += Ps
    task_params += Ns
    if args.metric == "biaffine":
        task_params += Ws
        task_params += Vs

    optim_model = optim.Adam(
        filter(requires_grad, task_params),
        lr = args.lr,
        weight_decay = 1e-4
    )

    say("Training will begin from scratch\n")

    best_dev = 0
    best_test = 0
    iter_cnt = 0

    for epoch in range(args.max_epoch):
        if args.metric == "biaffine":
            mats = [Us, Ws, Vs]
        else:
            mats = [Us, Ps, Ns]

        iter_cnt = train_epoch(
                iter_cnt,
                encoder, classifiers, critic,
                mats,
                [train_loaders, unl_loader, valid_loader],
                args,
                optim_model
            )

        (curr_dev, oracle_curr_dev), confusion_mat = evaluate(
                encoder, classifiers,
                mats,
                [train_loaders, valid_loader],
                args
            )
        say("Dev accuracy/oracle: {:.4f}/{:.4f}\n".format(curr_dev, oracle_curr_dev))
        (curr_test, oracle_curr_test), confusion_mat = evaluate(
                encoder, classifiers,
                mats,
                [train_loaders, test_loader],
                args
            )
        say("Test accuracy/oracle: {:.4f}/{:.4f}\n".format(curr_test, oracle_curr_test))

        if curr_dev >= best_dev:
            best_dev = curr_dev
            best_test = curr_test
            print(confusion_mat)
            if args.save_model:
                say(colored("Save model to {}\n".format(args.save_model + ".best"), 'red'))
                torch.save([encoder, classifiers, Us, Ps, Ns], args.save_model + ".best")
        say("\n")

    say(colored("Best test accuracy {:.4f}\n".format(best_test), 'red'))

def test_mahalanobis_metric():
    p = torch.FloatTensor(1, 5).normal_()
    S = torch.FloatTensor(4, 5).normal_()
    p = Variable(p)# .cuda()
    S = Variable(S)# .cuda()
    print(p, S)
    encoder = nn.Sequential(nn.Linear(5, 5), nn.ReLU())
    encoder = encoder# .cuda()
    nn.init.xavier_normal(encoder[0].weight)
    nn.init.constant(encoder[0].bias, 0.1)
    print(encoder[0].weight)
    d = mahalanobis_metric(p, S, args, encoder)
    print(d)

import argparse
if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="Learning to Adapt from Multi-Source Domains")
    argparser.add_argument("--cuda", action="store_true")
    argparser.add_argument("--train", type=str, required=True,
                           help="multi-source domains for training, separated with (,)")
    argparser.add_argument("--test", type=str, required=True,
                           help="target domain for testing")
    argparser.add_argument("--eval_only", action="store_true")
    argparser.add_argument("--critic", type=str, default="mmd")
    argparser.add_argument("--batch_size", type=int, default=32)
    argparser.add_argument("--batch_size_d", type=int, default=32)
    argparser.add_argument("--max_epoch", type=int, default=100)
    argparser.add_argument("--lr", type=float, default=1e-4)
    argparser.add_argument("--lr_d", type=float, default=1e-4)
    argparser.add_argument("--lambda_critic", type=float, default=1)
    argparser.add_argument("--lambda_gp", type=float, default=10)
    argparser.add_argument("--lambda_moe", type=float, default=1)
    argparser.add_argument("--m_rank", type=int, default=10)
    argparser.add_argument("--lambda_entropy", type=float, default=0.1)
    argparser.add_argument("--load_model", type=str)
    argparser.add_argument("--save_model", type=str)
    argparser.add_argument("--metric", type=str, default="mahalanobis",
                           help="mahalanobis: mahalanobis distance; biaffine: biaffine distance")

    args, _ = argparser.parse_known_args()

    random.seed(0)
    torch.manual_seed(0)
    if args.cuda:
        torch.cuda.manual_seed(0)

    if args.eval_only: predict(args)
    else:              train(args)

