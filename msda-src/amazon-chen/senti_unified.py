import os
import sys
import argparse
import time
import random
from termcolor import colored, cprint

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data as data

from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE

sys.path.append('../')
from model_utils import get_model_class, get_critic_class
from model_utils.domain_critic import ClassificationD, MMD, CoralD, WassersteinD
from utils.io import AmazonDataset, AmazonDomainDataset
from utils.io import say, plot_embedding
from utils.op import one_hot

# from flip_gradient import flip_gradient

DATA_DIR = "../msda-data/amazon/chen12"

def train_epoch(iter_cnt, encoder, classifier, critic, train_loaders, target_d_loader, valid_loader, args, optimizer):
    encoder.train()
    classifier.train()
    critic.train()

    task_criterion = nn.CrossEntropyLoss()
    ae_criterion = nn.MSELoss()

    for source_batches, target_batch in zip(zip(*train_loaders), target_d_loader):

        all_task_batch, all_task_labels = zip(*source_batches)
        target_d_batch, target_d_labels = target_batch

        iter_cnt += 1

        if args.cuda:
            all_task_batch = [ task_batch.cuda() for task_batch in all_task_batch ]
            all_task_labels = [ task_labels.cuda() for task_labels in all_task_labels ]

            target_d_batch = target_d_batch.cuda()
            target_d_labels = target_d_labels.cuda()

        all_task_batch = [ Variable(task_batch) for task_batch in all_task_batch ]
        all_task_labels = [ Variable(task_labels) for task_labels in all_task_labels ]

        target_d_batch = Variable(target_d_batch)
        target_d_labels = Variable(target_d_labels)

        optimizer.zero_grad()

        loss_c = []
        loss_d = []
        for task_batch, task_labels in zip(all_task_batch, all_task_labels):

            ''' compute task loss '''
            hidden = encoder(task_batch)
            task_output = classifier(hidden)

            ### task accuracy on training batch
            task_pred = torch.squeeze(task_output.max(dim=1)[1])
            task_acc = (task_pred == task_labels).float().mean()

            ### task loss
            loss_c.append(task_criterion(task_output, task_labels))

            ''' domain-critic loss '''
            critic_batch = torch.cat([task_batch, target_d_batch])
            critic_labels = torch.cat([1 - target_d_labels, target_d_labels])

            # hidden_d = encoder(critic_batch)
            # move the grl layer to DANN
            # hidden_d_grl = flip_gradient(hidden_d)
            if isinstance(critic, ClassificationD):
                critic_output = critic(encoder(critic_batch))
                critic_pred = torch.squeeze(critic_output.max(dim=1)[1])
                critic_acc = (critic_pred == critic_labels).float().mean()
                loss_d.append(critic.compute_loss(critic_output, critic_labels))
            else: # mmd, coral, wd
                if args.cond is not None:
                    # outer(encoding, g) where g is the class distribution
                    target_d_g = F.softmax(classifier(encoder(target_d_batch)), dim=1).detach()
                    # task_g = F.softmax(task_output, dim=1).detach()
                    task_g = one_hot(task_labels, cuda=args.cuda)
                    # print torch.cat([task_g, task_labels.unsqueeze(1).float()], dim=1)
                    # print target_d_g

                    if args.cond == "concat":
                        task_encoding = torch.cat([hidden, task_g], dim=1)
                        target_d_encoding = torch.cat([encoder(target_d_batch), target_d_g], dim=1)
                    else: # "outer"
                        task_encoding = torch.bmm(hidden.unsqueeze(2), task_g.unsqueeze(1))
                        target_d_encoding = torch.bmm(encoder(target_d_batch).unsqueeze(2), target_d_g.unsqueeze(1))
                        task_encoding = task_encoding.view(task_encoding.shape[0], -1).contiguous()
                        target_d_encoding = target_d_encoding.view(target_d_encoding.shape[0], -1).contiguous()
                        # print task_encoding.shape, target_d_encoding.shape
                    critic_output = critic(task_encoding, target_d_encoding)
                else:
                    critic_output = critic(hidden, encoder(target_d_batch))
                loss_d.append(critic_output)

        loss_c = sum(loss_c)
        loss_d = sum(loss_d)
        loss = loss_c + args.lambda_critic * loss_d
        loss.backward()
        optimizer.step()

        if iter_cnt % 30 == 0:
            curr_test, confusion_mat, _  = evaluate(encoder, classifier, valid_loader, args)

            # say("\r" + " " * 50)
            say("{} task loss/acc: {:.4f}/{:.4f}, "
                "domain critic loss: {:.4f}, "
                # "adversarial loss/acc: {:.4f}/{:.4f}, "
                "loss: {:.4f}, "
                "test acc: {:.4f}\n"
                .format(iter_cnt,
                        loss_c.data[0], task_acc.data[0],
                        loss_d.data[0],
                        # loss_target_d.data[0], target_d_acc.data[0],
                        loss.data[0],
                        curr_test
            ))

    say("\n")
    return iter_cnt

def train_advreg_mmd(iter_cnt, encoder, gan_g, gan_d, corpus_loader, args, optimizer_reg):
    encoder.train()
    gan_g.train()
    gan_d.train()

    # train gan_disc
    for batch, labels in corpus_loader:
        optimizer_reg.zero_grad()

        batch = Variable(batch.cuda())
        z_real_hidden = encoder(batch)
        z_gauss = torch.normal(means=torch.zeros(batch.size()),
                               std=args.noise_radius)
        z_gauss = Variable(z_gauss.cuda())
        z_gauss_hidden = gan_g(z_gauss)

        loss_ar = gan_d(z_real_hidden, z_gauss_hidden)

        loss_ar.backward()
        optimizer_reg.step()

def evaluate(encoder, classifier, corpus_loader, args):
    encoder.eval()
    classifier.eval()

    correct = 0
    tot_cnt = 0
    y_true = []
    y_pred = []

    encoding_vecs = torch.FloatTensor()
    encoding_vecs = encoding_vecs.cuda()
    all_labels = torch.LongTensor()
    all_labels = all_labels.cuda()

    for batch, labels in corpus_loader:
        if args.cuda:
            batch = batch.cuda()
            labels = labels.cuda()

        batch = Variable(batch)
        hidden = encoder(batch)

        # print encoding_vecs, hidden, labels.view(-1, 1)
        if args.visualize:
            encoding_vecs = torch.cat([encoding_vecs, hidden.data])
            all_labels = torch.cat([all_labels, labels.view(-1, 1)])

        output = classifier(hidden)
        pred = output.data.max(dim = 1)[1]
        y_true += labels.tolist()
        y_pred += pred.tolist()
        correct += pred.eq(labels).sum()
        tot_cnt += output.size(0)

    acc = float(correct) / tot_cnt
    return acc, confusion_matrix(y_true, y_pred), (encoding_vecs, all_labels)

# todo: add support for train/test phase
def train(args):
    encoder_class = get_model_class(args.encoder)
    encoder_class.add_config(argparser)
    critic_class = get_critic_class(args.critic)
    critic_class.add_config(argparser)

    args = argparser.parse_args()
    say(args)

    say("Transferring from %s to %s\n" % (args.train, args.test))

    source_train_sets = args.train.split(',')
    train_loaders = []
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

    target_d_filepath = os.path.join(DATA_DIR, "%s_train.svmlight" % (args.test))
    assert (os.path.exists(target_d_filepath))
    train_target_d_dataset = AmazonDomainDataset(target_d_filepath, domain=1)
    train_target_d_loader = data.DataLoader(
        train_target_d_dataset,
        batch_size=args.batch_size_d,
        shuffle=True,
        num_workers=0
    )

    valid_filepath = os.path.join(DATA_DIR, "%s_dev.svmlight" % (args.test))
    assert (os.path.exists(valid_filepath))
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

    encoder = encoder_class(args)
    critic = critic_class(encoder, args)
    classifier = nn.Linear(encoder.n_out, 2) # binary classification
    nn.init.xavier_normal_(classifier.weight)
    nn.init.constant_(classifier.bias, 0.1)

    gan_gen = encoder_class(args)
    gan_disc = MMD(gan_gen, args)

    if args.cuda:
        encoder = encoder.cuda()
        critic = critic.cuda()
        classifier = classifier.cuda()
        gan_gen = gan_gen.cuda()
        gan_disc = gan_disc.cuda()

    say("\n{}\n\n".format(encoder))
    say("\n{}\n\n".format(critic))
    say("\n{}\n\n".format(classifier))
    say("\n{}\n\n".format(gan_gen))
    say("\n{}\n\n".format(gan_disc))

    print(encoder.state_dict().keys())
    print(critic.state_dict().keys())
    print(classifier.state_dict().keys())
    print(gan_gen.state_dict().keys())
    print(gan_disc.state_dict().keys())

    requires_grad = lambda x : x.requires_grad
    task_params = list(encoder.parameters()) + \
                  list(classifier.parameters()) + \
                  list(critic.parameters())
    optimizer = optim.Adam(
        filter(requires_grad, task_params),
        lr = args.lr,
        weight_decay = 1e-4
    )

    reg_params = list(encoder.parameters()) + \
                 list(gan_gen.parameters())
    optimizer_reg = optim.Adam(
        filter(requires_grad, reg_params),
        lr = args.lr,
        weight_decay = 1e-4
    )

    say("Training will begin from scratch\n")

    best_dev = 0
    best_test = 0
    iter_cnt = 0

    for epoch in range(args.max_epoch):
        iter_cnt = train_epoch(
            iter_cnt,
            encoder, classifier, critic,
            train_loaders, train_target_d_loader, valid_loader,
            args,
            optimizer
        )

        if args.advreg:
            for loader in train_loaders + [train_target_d_loader]:
                train_advreg_mmd(
                    iter_cnt,
                    encoder, gan_gen, gan_disc,
                    loader,
                    args,
                    optimizer_reg
                )

        curr_dev, confusion_mat, _ = evaluate(encoder, classifier, valid_loader, args)
        curr_test, confusion_mat, _ = evaluate(encoder, classifier, test_loader, args)
        say("Test accuracy: {:.4f}\n".format(curr_test))

        if curr_dev >= best_dev:
            best_dev = curr_dev
            best_test = curr_test
            print(confusion_mat)
            if args.save_model:
                say(colored("Save model to {}\n".format(args.save_model + ".best"), 'red'))
                torch.save([encoder, classifier], args.save_model + ".best")
        say("\n")

    say(colored("Best test accuracy {:.4f}\n".format(best_test), 'red'))
    # if args.save_model:
    #     say("Save final model to {}\n".format(args.save_model))
    #     torch.save(encoder.state_dict(), args.save_model)

def visualize(args):
    encoder, classifier = torch.load(args.load_model)

    if args.cuda:
        encoder = encoder.cuda()
        classifier = classifier.cuda()

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

    target_d_filepath = os.path.join(DATA_DIR, "%s_train.svmlight" % (args.test))
    train_target_d_dataset = AmazonDataset(target_d_filepath)
    train_target_d_loader = data.DataLoader(
        train_target_d_dataset,
        batch_size=args.batch_size_d,
        shuffle=False,
        num_workers=0
    )

    source_hs = []
    source_ys = []
    source_num = 0
    for loader in train_loaders:
        _, _, (hs, ys) = evaluate(encoder, classifier, loader, args)
        source_hs.append(hs)
        source_ys.append(ys)
        source_num += ys.shape[0]
    _, _, (ht, yt) = evaluate(encoder, classifier, train_target_d_loader, args)
    h_both = torch.cat(source_hs + [ht]).cpu().numpy()
    y_both = torch.cat(source_ys + [yt]).cpu().numpy()

    tsne = TSNE(perplexity=30, n_components=2, n_iter=3300)
    vdata = tsne.fit_transform(h_both)
    print(vdata.shape, source_num)
    torch.save([vdata, y_both, source_num], 'vis/%s-%s-mdan.vdata' % (args.train, args.test))
    plot_embedding(vdata, y_both, source_num, args.save_image)

def predict(args):
    encoder, classifier = torch.load(args.load_model)
    map(lambda m: m.eval(), [encoder, classifier])

    if args.cuda:
        # map(lambda m: m.cuda(), [encoder, classifier])
        encoder = encoder.cuda()
        classifier = classifier.cuda()

    test_filepath = os.path.join(DATA_DIR, "%s_train.svmlight" % (args.test))
    assert (os.path.exists(test_filepath))
    test_dataset = AmazonDataset(test_filepath)
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    acc, confusion_mat, _  = evaluate(encoder, classifier, test_loader, args)
    say(colored("Test accuracy {:.4f}\n".format(acc), 'red'))
    print(confusion_mat)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Unsupervised Domain Adaptation")
    argparser.add_argument("--cuda", action="store_true")
    argparser.add_argument("--encoder", "-m", type=str, required=True, help="encoder type")
    argparser.add_argument("--critic", type=str, default="dann")
    argparser.add_argument("--advreg", action="store_true")
    argparser.add_argument("--train", type=str, required=True,
                           help="multi-source training domains, separated with comma (,)")
    argparser.add_argument("--test", type=str, required=True, help="test domain")
    argparser.add_argument("--eval_only", action="store_true")
    argparser.add_argument("--batch_size", type=int, default=32)
    argparser.add_argument("--batch_size_d", type=int, default=32)
    argparser.add_argument("--max_epoch", type=int, default=100)
    argparser.add_argument("--optim", type=str, default="adam")
    argparser.add_argument("--lr", type=float, default=1e-4)
    argparser.add_argument("--lr_d", type=float, default=1e-4)
    argparser.add_argument("--lambda_critic", type=float, default=1)
    argparser.add_argument("--noise_radius", type=float, default=0.2)
    argparser.add_argument("--load_model", type=str, required=False,
                           help="load a pretrained model")
    argparser.add_argument("--save_model", type=str, required=False,
                           help="location to save a model")
    argparser.add_argument("--save_image", type=str, default="./tmp.pdf",
                           help="location to save the visualized output")
    argparser.add_argument("--visualize", action="store_true")
    argparser.add_argument("--cond", type=str,
                           help="Set to True when using conditional adversarial training")

    args, _ = argparser.parse_known_args()

    random.seed(0)
    torch.manual_seed(0)
    if args.cuda:
        torch.cuda.manual_seed(0)

    if args.eval_only:   predict(args)
    elif args.visualize: visualize(args)
    else:                train(args)

