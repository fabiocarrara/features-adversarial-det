import argparse
import copy
import itertools
import os
import sys
from collections import OrderedDict

import h5py
import numpy as np
import pandas as pd
import torch
import torchvision
from sklearn import metrics
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision.transforms import ToTensor, Compose, Resize, CenterCrop, Normalize
from tqdm import tqdm, trange

from utils import AdvDataset


class Detector(nn.Module):

    def __init__(self, centroids, centroid_dist='euclidean', h=100, bidir=False, **kwargs):
        super(Detector, self).__init__()

        with h5py.File(centroids, 'r') as f:
            self.centroid_dist = centroid_dist
            self.centroids = [torch.tensor(i[()]).to(kwargs['device']) for i in f.values()]

        self.lstm = nn.LSTM(1000, h, bidirectional=bidir)
        lstm_out_size = h * 2 if bidir else h
        self.classifier = nn.Linear(lstm_out_size, 1)

    def embed_one(self, x, c):
        if self.centroid_dist == 'euclidean':
            return torch.stack([(i - c).norm(dim=1) for i in x])
        elif self.centroid_dist == 'cosine':
            return torch.stack([F.cosine_similarity(i.unsqueeze(0), c) for i in x])

    def embed(self, x):
        embedded = [self.embed_one(x_i, c_i) for x_i, c_i in zip(x, self.centroids)]
        embedded = torch.stack(embedded)
        return embedded

    def forward(self, x):
        x = self.embed(x)
        output, _ = self.lstm(x)
        last_output = output[-1]
        return self.classifier(last_output)


def train(loader, detector, model, optimizer, args):
    global features_state
    detector.train()
    progress = tqdm(loader)
    for x, y in progress:
        # forward pass to collect features
        with torch.no_grad():
            model(x.to(args.device))
            features = list(features_state.values())

        y_hat = detector(features)
        y = y.reshape(-1, 1).float().to(args.device)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        progress.set_postfix({'loss': '{:6.4f}'.format(loss.tolist())})


def evaluate(loader, detector, model, args):
    with torch.no_grad():
        detector.eval()

        y = []
        y_hat = []
        for Xb, yb in tqdm(loader):
            y.append(yb.cpu().numpy())

            model(Xb.to(args.device))
            features = list(features_state.values())
            logits = detector(features)
            yb_hat = F.sigmoid(logits).squeeze().cpu().numpy()

            y_hat.append(yb_hat)

        y = np.concatenate(y)
        y_hat = np.concatenate(y_hat)

        # AUC
        fpr, tpr, thr = metrics.roc_curve(y, y_hat)
        auc = metrics.auc(fpr, tpr)

        # EER accuracy
        fnr = 1 - tpr
        eer_thr = thr[np.nanargmin(np.absolute(fnr - fpr))]
        eer_accuracy = metrics.accuracy_score(y, y_hat > eer_thr)
        eer = (eer_accuracy, eer_thr)

        # Best TPR-FPR
        dist = fpr ** 2 + (1 - tpr) ** 2
        best = np.argmin(dist)
        best = fpr[best], tpr[best], thr[best], auc

        tqdm.write('EER Accuracy: {:3.2%} ({:g})'.format(*eer))
        tqdm.write('BEST TPR-FPR: {:4.3%} {:4.3%} ({:g}) AUC: {:4.3%}'.format(*best))

        return ('auc', 'eer_accuracy'), torch.tensor((auc, eer_accuracy))


if __name__ == '__main__':

    global features_state
    # global var for forward hooks
    features_state = OrderedDict()

    parser = argparse.ArgumentParser(description='Train Adversarial Detector from Features in Dissimilarity Space')

    parser.add_argument('--eval', action='store_true', help='Load pre-trained model and evaluate it, then stop')

    # DATA PARAMS
    parser.add_argument('-o', '--orig-data', nargs='+', help='Folders containing original iamges')
    parser.add_argument('-a', '--adv-data', nargs='+', help='Folders containing adversarial images')

    # MODEL PARAMS
    parser.add_argument('-c', '--centroids', default='ilsvrc12_centroids.h5', help='Centroids for distance embedding')
    parser.add_argument('--bidir', action='store_true', help='Use Bidirectional LSTM')

    # TRAIN PARAMS
    parser.add_argument('-e', '--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('-b', '--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('-l', '--lr', type=float, default=0.003, help='Learning rate')
    parser.add_argument('--weight-decay', '--wd', type=float, default=0, help='L2 penalty weight decay')
    parser.add_argument('-s', '--seed', type=int, default=42, help='Random seed')

    # OTHER
    parser.add_argument('-r', '--run_dir', default='runs/debug', help='Base dir for run files')

    parser.set_defaults(bidir=False, eval=False)
    args = parser.parse_args()

    params = copy.deepcopy(vars(args))
    params['orig_data'] = ','.join(params['orig_data'])
    params['adv_data'] = ','.join(params['adv_data'])
    params = pd.DataFrame(params, index=[0])

    # CUDA?
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)

    # Data Loaders
    transform = Compose([Resize(256),
                         CenterCrop(224),
                         ToTensor(),
                         Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                         ])

    dataset = AdvDataset(args.orig_data, args.adv_data, orig_transform=transform)
    n_samples = len(dataset)
    split_len = ((n_samples // 3) + (n_samples % 3), n_samples // 3, n_samples // 3)
    train_data, val_data, test_data = random_split(dataset, split_len)

    train_loader = DataLoader(train_data, shuffle=True, pin_memory=True, num_workers=8, batch_size=args.batch_size)
    val_loader = DataLoader(val_data, shuffle=False, pin_memory=True, num_workers=8, batch_size=args.batch_size)

    # Models and optimizers
    model = torchvision.models.resnet50(pretrained=True).to(args.device)
    model.eval()


    def extract(self, input, output):
        if output.ndimension() > 2:
            features = F.avg_pool2d(output, output.shape[-2:]).squeeze(3).squeeze(2)

        features_state[self] = features


    blocks = itertools.chain(model.layer1, model.layer2, model.layer3, model.layer4, (model.avgpool,))
    for b in blocks:
        b.register_forward_hook(extract)

    detector = Detector(**vars(args)).to(args.device)

    if args.eval:
        test_loader = DataLoader(test_data, shuffle=False, pin_memory=True, num_workers=8, batch_size=args.batch_size)

        ckpt_path = os.path.join(args.run_dir, 'ckpt', 'best_model.pth')
        print('Loading:', ckpt_path)
        ckpt = torch.load(ckpt_path)
        detector.load_state_dict(ckpt['detector'])

        test_metrics_names, test_metrics = evaluate(test_loader, detector, model, args)
        test_metrics_dict = dict(zip(test_metrics_names, test_metrics.tolist()))
        print(pd.DataFrame(test_metrics_dict, index=[0]))
        sys.exit(0)

    # Setup folders
    if not os.path.exists(args.run_dir):
        os.makedirs(args.run_dir)

    ckpt_dir = os.path.join(args.run_dir, 'ckpt')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    params_file = os.path.join(args.run_dir, 'params.csv')
    params.to_csv(params_file, index=False)

    log_file = os.path.join(args.run_dir, 'log.csv')
    log = pd.DataFrame()

    optimizer = Adam(detector.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, 10)

    # Train loop
    best = torch.zeros(2)
    progress = trange(1, args.epochs + 1)
    for epoch in progress:
        progress.set_description('TRAIN')
        train(train_loader, detector, model, optimizer, args)
        progress.set_description('EVAL')
        val_metrics_names, val_metrics = evaluate(val_loader, detector, model, args)

        val_metrics_dict = dict(zip(val_metrics_names, val_metrics.tolist()))
        log = log.append(pd.DataFrame(val_metrics_dict, index=[pd.Timestamp('now')]))
        log.to_csv(log_file)

        if best[0] < val_metrics[0]:  # keep best AUC
            ckpt_path = os.path.join(ckpt_dir, 'best_model.pth')
            torch.save({
                'detector': detector.state_dict(),
                'optimizer': optimizer.state_dict(),
                'metrics': val_metrics_dict
            }, ckpt_path)

        best = torch.max(val_metrics, best)
        scheduler.step()

