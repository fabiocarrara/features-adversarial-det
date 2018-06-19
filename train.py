import argparse

import torch
import numpy as np

from torch import nn
from torch.optim import Adam
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from tqdm import tqdm, trange
from sklearn import metrics


class Detector(nn.Module):

    def __init__(self, h=100):
        super(Detector, self).__init__()

        self.lstm = nn.LSTM(1000, h, bidirectional=True)
        self.classifier = nn.Linear(2 * h, 1)

    def forward(self, x):
        output, _ = self.lstm(x)
        last_output = output[-1]
        return self.classifier(last_output)


def split_data(data, k, f, device):
    data = np.load(data)  # LAYER x IMAGE x FEATURES
    X, y = data['x'], data['y']
    n_samples = X.shape[1]
    test_mask = np.zeros(n_samples, dtype=bool)
    test_mask[(f % k)::k] = 1
    train_mask = ~test_mask  # boolean not

    X_train, y_train = X[:, train_mask, :], y[train_mask]
    X_test, y_test = X[:, test_mask, :], y[test_mask]

    n_train_samples = X_train.shape[1]
    shuffle = torch.randperm(n_train_samples)
    
    X_train = torch.from_numpy(X_train)[:, shuffle, :].to(device)
    y_train = torch.from_numpy(y_train)[shuffle].to(device)
    
    X_test = torch.from_numpy(X_test).to(device)
    y_test = torch.from_numpy(y_test)
    
    return (X_train, y_train), (X_test, y_test)


def train(data, model, optimizer, args):
    model.train()
    X, y = data
    n_samples = X.shape[1]
    
    # shuffle = torch.randperm(n_samples)
    # X = X[:, shuffle, :].to(args.device)
    # y = y[shuffle].to(args.device)

    batch_starts = range(0, n_samples, args.batch_size)
    batch_ends = range(args.batch_size, n_samples + args.batch_size, args.batch_size)
    batches = list(zip(batch_starts, batch_ends))
    progress = tqdm(batches)
    for start, end in progress:
        Xb, yb = X[:, start:end, :], y[start:end]
        y_hat = model(Xb).squeeze()
        loss = F.binary_cross_entropy_with_logits(y_hat, yb.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        progress.set_postfix({'loss': '{:6.4f}'.format(loss.tolist())})


def evaluate(data, model, args):
    with torch.no_grad():
        model.eval()
        X, y = data

        logits = model(X)
            
        y_hat = F.sigmoid(logits).squeeze().cpu().numpy()
        
        # AUC
        fpr, tpr, thr = metrics.roc_curve(y, y_hat)
        auc = metrics.auc(fpr, tpr)
        
        # EER accuracy
        fnr = 1 - tpr
        eer_thr = thr[np.nanargmin(np.absolute(fnr - fpr))]
        eer_accuracy = metrics.accuracy_score(y, y_hat > eer_thr)
        eer = (eer_accuracy, eer_thr)
        
        # Best TPR-FPR
        dist = fpr**2 + (1-tpr)**2
        best = np.argmin(dist)
        best = fpr[best], tpr[best], thr[best], auc
        
        tqdm.write('EER Accuracy: {:3.2%} ({:g})'.format(*eer))
        tqdm.write('BEST TPR-FPR: {:4.3%} {:4.3%} ({:g}) AUC: {:4.3%}'.format(*best))

        return torch.tensor((auc, eer_accuracy))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse dissimilarity CSVs')
    parser.add_argument('data', help='Dataset file')
    parser.add_argument('-k', type=int, default=5, help='Number of data folds')
    # parser.add_argument('-f', type=int, default=0, help='Fold index (starting from 0) to be used for test (the rest is for train)')

    parser.add_argument('-e', '--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('-b', '--batch-size', type=int, default=100, help='Number of training epochs')
    parser.add_argument('-s', '--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    torch.manual_seed(args.seed)

    folds_results = []
    for fold in trange(args.k):
        train_data, test_data = split_data(args.data, args.k, fold, args.device)
        model = Detector().to(args.device)
        optimizer = Adam(model.parameters())
        scheduler = CosineAnnealingLR(optimizer, 10)

        best = torch.zeros(2)
        for epoch in trange(1, args.epochs + 1):
            train(train_data, model, optimizer, args)
            val_metrics = evaluate(test_data, model, args)
            best = torch.max(val_metrics, best)
            scheduler.step()
        
        print('BEST OF FOLD {} | AUC={:3.2%} EER-Acc={:3.2%}'.format(fold, *best))
        folds_results.append(best)
    
    folds_results = torch.stack(folds_results)
    print('ALL FOLDS RESULTS | (AUC, EER-Acc):')
    print(folds_results)
    
    kfold_result = folds_results.mean(dim=0)
    print('K-FOLD RESULT | AUC={:3.2%} EER-Acc={:3.2%}'.format(*kfold_result))
