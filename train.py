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

    def __init__(self, h=100, bidir=False, **kwargs):
        super(Detector, self).__init__()

        self.lstm = nn.LSTM(1000, h, bidirectional=bidir)
        lstm_out_size = h*2 if bidir else h
        self.classifier = nn.Linear(lstm_out_size, 1)

    def forward(self, x):
        output, _ = self.lstm(x)
        last_output = output[-1]
        return self.classifier(last_output)


def pure_data(X, y, balance=True):
    keep = np.ones_like(y, dtype=bool)
    keep[0:1000] = y[0:1000] == 1  # adversarial, keep only successful (y = 1)
    keep[1000:2000] = y[1000:2000] == 0  # authentic, do not keep errors (y = 1)
    keep[2000:3000] = 0  # random_noise, do not keep it
    keep[3000:4000] = y[3000:4000] == 1  # adversarial, keep only successful (y = 1)
    keep[4000:5000] = y[4000:5000] == 1  # adversarial, keep only successful (y = 1)

    if balance:
        n_adv1 = keep[0:1000].sum()
        n_auth = keep[1000:2000].sum()
        # no random noise
        n_adv2 = keep[3000:4000].sum()
        n_adv3 = keep[4000:5000].sum()

        n_adv_per_category = n_auth // 3  # n. of adv per category to keep

        # discard adv1
        n_to_discard = n_adv1 - n_adv_per_category
        idx = np.where(keep[0:1000])[0]
        to_discard = np.random.choice(idx, n_to_discard, replace=False)
        keep[0:1000][to_discard] = 0

        # discard adv2
        n_to_discard = n_adv2 - n_adv_per_category
        idx = np.where(keep[3000:4000])[0]
        to_discard = np.random.choice(idx, n_to_discard, replace=False)
        keep[3000:4000][to_discard] = 0

        # discard adv3
        n_to_discard = n_adv3 - n_adv_per_category
        idx = np.where(keep[4000:5000])[0]
        to_discard = np.random.choice(idx, n_to_discard, replace=False)
        keep[4000:5000][to_discard] = 0

    X = X[:, keep, :]
    y = y[keep]

    return X, y, keep
    

def mtap_data(X, y):
    keep = np.ones_like(y, dtype=bool)
    keep[0:1000] = y[0:1000] == 1  # adversarial, keep only successful (y = 1)
    keep[1000:2000] = y[1000:2000] == 0  # authentic, do not keep errors (y = 1)
    keep[2000:3000] = y[2000:3000] == 0  # random_noise, keep only the ones remained authentic
    keep[3000:4000] = y[3000:4000] == 1  # adversarial, keep only successful (y = 1)
    keep[4000:5000] = y[4000:5000] == 1  # adversarial, keep only successful (y = 1)

    X = X[:, keep, :]
    y = y[keep]

    return X, y, keep


def load_data(args):
    data = np.load(args.data)  # LAYER x IMAGE x FEATURES
    X, y = data['x'], data['y']
    if args.data_selection == 'pure':
        X, y, keep = pure_data(X, y)
    elif args.data_selection == 'mtap':
        X, y, keep = mtap_data(X, y)
    
    print('Data kept (selection: {}) | {} (Adv. {:3.2%})'.format(args.data_selection, X.shape[1], y.sum() / len(y)))
    
    if args.preproc == 'sqrt':
        X = np.sqrt(X)
    elif args.preproc == 'squared':
        X = X**2
    elif args.preproc == 'repeat':
        X = np.tile(X, (2, 1, 1))  # repeat sequence twice    
    elif args.preproc == 'reverse':
        X = X[::-1]
    
    return X, y, keep
    

def split_data(data, f, args):    
    X, y = data
    n_samples = X.shape[1]
    test_mask = np.zeros(n_samples, dtype=bool)
    test_mask[(f % args.k)::args.k] = 1
    train_mask = ~test_mask  # boolean not

    X_train, y_train = X[:, train_mask, :], y[train_mask]
    X_test, y_test = X[:, test_mask, :], y[test_mask]

    n_train_samples = X_train.shape[1]
    shuffle = torch.randperm(n_train_samples)
    
    X_train = torch.from_numpy(X_train)[:, shuffle, :].to(args.device)
    y_train = torch.from_numpy(y_train)[shuffle].to(args.device)
    
    X_test = torch.from_numpy(X_test).to(args.device)
    y_test = torch.from_numpy(y_test)

    if args.preproc == 'add-last-softmax':
        X_train = torch.cat((X_train, F.softmax(X_train[-1:], dim=2)), dim=0)
        X_test = torch.cat((X_test, F.softmax(X_test[-1:], dim=2)), dim=0)
    elif args.preproc == 'softmax':
        X_train = F.softmax(X_train, dim=2)
        X_test = F.softmax(X_test, dim=2)
    
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

        return y_hat, torch.tensor((auc, eer_accuracy))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Adversarial Detector from Features in Dissimilarity Space')
    # DATA PARAMS
    parser.add_argument('data', help='Dataset file')
    parser.add_argument('-d', '--data-selection', choices=('pure', 'mtap'), default=None, help='How to select data')
    parser.add_argument('-p', '--preproc', choices=('sqrt', 'squared', 'repeat', 'reverse', 'softmax', 'add-last-softmax'), default=None, help='Data preprocessing')
    parser.add_argument('-k', type=int, default=5, help='Number of data folds')
    
    # parser.add_argument('-f', type=int, default=0, help='Fold index (starting from 0) to be used for test (the rest is for train)')

    # MODEL PARAMS
    parser.add_argument('--bidir', action='store_true', help='Use Bidirectional LSTM')

    # TRAIN PARAMS
    parser.add_argument('-e', '--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('-b', '--batch-size', type=int, default=100, help='Number of training epochs')
    parser.add_argument('-s', '--seed', type=int, default=42, help='Random seed')
    
    # OUTPUTS
    parser.add_argument('-o', '--output', default='kfold-predictions.npz', help='Where to save output predictions')

    parser.set_defaults(bidir=False)
    args = parser.parse_args()
    
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    torch.manual_seed(args.seed)

    folds_results = []
    
    data = load_data(args)
    data, keep = data[:2], data[2]
    
    predictions = np.empty_like(data[1], dtype=np.float32)
    targets = np.empty_like(data[1])
    
    for fold in trange(args.k):
        train_data, test_data = split_data(data, fold, args)
        model = Detector(**vars(args)).to(args.device)
        optimizer = Adam(model.parameters())
        scheduler = CosineAnnealingLR(optimizer, 10)

        best = torch.zeros(2)
        for epoch in trange(1, args.epochs + 1):
            train(train_data, model, optimizer, args)
            val_predictions, val_metrics = evaluate(test_data, model, args)
            if best[0] < val_metrics[0]:  # keep best AUC
                predictions[(fold % args.k)::args.k] = val_predictions
                targets[(fold % args.k)::args.k] = test_data[1]
                
            best = torch.max(val_metrics, best)
            scheduler.step()
        
        print('BEST OF FOLD {} | AUC={:3.2%} EER-Acc={:3.2%}'.format(fold, *best))
        folds_results.append(best)
    
    folds_results = torch.stack(folds_results)
    print('ALL FOLDS RESULTS | (AUC, EER-Acc):')
    print(folds_results)
    
    kfold_result = folds_results.mean(dim=0)
    print('K-FOLD RESULT | AUC={:3.2%} EER-Acc={:3.2%}'.format(*kfold_result))
    
    print('Saving K-FOLD predictions:', args.output)
    np.savez(args.output, y_hat=predictions, y=targets, keep=keep)
    
