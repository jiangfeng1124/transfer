import sys, os, glob
import argparse
import time
import random
from copy import copy, deepcopy

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
from utils.io import say, plot_embedding, ms_plot_embedding_sep, ms_plot_embedding_uni
from utils.op import one_hot

from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE

DATA_DIR = "../../msda-data/amazon/chen12"

def visualize(args):
    if args.mop == 3:
        encoder, classifiers, source_classifier = torch.load(args.load_model)
    elif args.mop == 2:
        encoder, classifiers, Us, Ps, Ns = torch.load(args.load_model)
    else:
        say("\nUndefined --mop\n")
        return

    map(lambda m: m.eval(), [encoder] + classifiers)
    if args.cuda:
        map(lambda m: m.cuda(), [encoder] + classifiers)

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

    test_filepath = os.path.join(DATA_DIR, "%s_train.svmlight" % (args.test))
    test_dataset = AmazonDataset(test_filepath)
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    say("Corpus loaded.\n")

    source_hs = []
    source_ys = []
    source_num = []
    for loader in train_loaders:
        encoding_vecs = torch.FloatTensor()
        labels = torch.LongTensor()
        if args.cuda:
            encoding_vecs = encoding_vecs.cuda()
            labels = labels.cuda()

        for batch, label in loader:
            if args.cuda:
                batch = batch.cuda()
                label = label.cuda()

            batch = Variable(batch)
            hidden = encoder(batch)
            encoding_vecs = torch.cat([encoding_vecs, hidden.data])
            labels = torch.cat([labels, label.view(-1, 1)])

        source_hs.append(encoding_vecs)
        source_ys.append(labels)
        source_num.append(labels.shape[0])

    ht = torch.FloatTensor()
    yt = torch.LongTensor()
    if args.cuda:
        ht = ht.cuda()
        yt = yt.cuda()

    for batch, label in test_loader:
        if args.cuda:
            batch = batch.cuda()
            label = label.cuda()

        batch = Variable(batch)
        hidden = encoder(batch)
        ht = torch.cat([ht, hidden.data])
        yt = torch.cat([yt, label.view(-1, 1)])

    h_both = torch.cat(source_hs + [ht]).cpu().numpy()
    y_both = torch.cat(source_ys + [yt]).cpu().numpy()

    say("Dimension reduction...\n")
    tsne = TSNE(perplexity=30, n_components=2, n_iter=3300)
    vdata = tsne.fit_transform(h_both)
    print vdata.shape, source_num
    torch.save([vdata, y_both, source_num], 'vis/%s-%s-mop%d.vdata' % (args.train, args.test, args.mop))
    ms_plot_embedding_sep(vdata, y_both, source_num, args.save_image)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="Learning to Adapt from Multi-Source Domains")
    argparser.add_argument("--cuda", action="store_true")
    argparser.add_argument("--train", type=str,
                           help="multi-source domains for training, separated with (,)")
    argparser.add_argument("--test", type=str,
                           help="target domain for testing")
    argparser.add_argument("--load_model", type=str)
    argparser.add_argument("--batch_size", type=int, default=32)
    argparser.add_argument("--vdata", type=str)
    argparser.add_argument("--save_image", type=str, default="./vis/tmp",
                           help="location to save the visualized output")
    argparser.add_argument("--ms", action="store_true")
    argparser.add_argument("--mop", type=int, default=3)

    args = argparser.parse_args()
    if args.vdata is not None:
        vdata, y_both, source_num = torch.load(args.vdata)
        ms_plot_embedding_sep(vdata, y_both, source_num, args.save_image)
    else:
        visualize(args)

# python visualize.py --cuda --train books,dvd,electronics --test kitchen --load_model $model --save_image $image

