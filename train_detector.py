import argparse
import copy
import itertools
import os
import sys
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torchvision
from sklearn import metrics
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision.transforms import ToTensor, Compose, Resize, CenterCrop, Normalize
from tqdm import tqdm, trange

from model import precompute_embeddings, Detector
from utils import split_adversarials, OrigAdvDataset, get_attack


def train(loader, detector, optimizer, args):
    global features_state
    detector.train()
    progress = tqdm(loader)
    for x, y in progress:
        y = y.reshape(-1, 1).float().to(args.device)
        y_hat = detector(x.to(args.device))
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        progress.set_postfix({'loss': '{:6.4f}'.format(loss.tolist())})


def evaluate(loader, paths, detector, args, return_predictions=False):

    with torch.no_grad():
        detector.eval()

        y = []
        y_hat = []
        for Xb, yb in tqdm(loader):

            y.append(yb.cpu().numpy())
            logits = detector(Xb.to(args.device))
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
        tqdm.write('EER Accuracy: {:3.2%} ({:g})'.format(*eer))

        # Best TPR-FPR
        dist = fpr ** 2 + (1 - tpr) ** 2
        best = np.argmin(dist)
        best = fpr[best], tpr[best], thr[best], auc
        tqdm.write('BEST TPR-FPR: {:4.3%} {:4.3%} ({:g}) AUC: {:4.3%}'.format(*best))
        
        # Macro-avg AUC
        a = [get_attack(i) for i in paths]
        data = pd.DataFrame({'pred': y_hat, 'target': y, 'attack': a})
        auths = data[data.attack == 'auth']
        print('Attack AUCs:')
        aucs = {}
        for attack, group in data.groupby('attack'):
            if attack == 'auth': continue
            pred = np.concatenate((group.pred.values, auths.pred.values))
            target = np.concatenate((group.target.values, auths.target.values))
            aucs[attack] = metrics.roc_auc_score(target, pred)
            print('{}: {:4.3%}'.format(attack, aucs[attack]))
        
        macro_auc = sum(aucs.values()) / len(aucs)

        print('Macro AUC: {:4.3%}'.format(macro_auc))

        if return_predictions:
            return y_hat, y, ('auc', 'eer_accuracy', 'macro_auc'), torch.tensor((auc, eer_accuracy, macro_auc))

        return ('auc', 'eer_accuracy', 'macro_auc'), torch.tensor((auc, eer_accuracy, macro_auc))


if __name__ == '__main__':

    global features_state
    # global var for forward hooks
    features_state = OrderedDict()

    parser = argparse.ArgumentParser(description='Train Adversarial Detector from Features in Dissimilarity Space')
    parser.add_argument('--eval', action='store_true', help='Load pre-trained model and evaluate it, then stop')

    # DATA PARAMS
    parser.add_argument('adversarials', help='CSV containing path to adversarial images')

    # MODEL PARAMS
    parser.add_argument('-c', '--centroids', default='ilsvrc12_centroids.h5', help='Centroids for distance embedding')
    parser.add_argument('-d', '--distance', choices=('euclidean', 'cosine'), default='euclidean', help='Distance to be used in embeddings')
    parser.add_argument('--hidden', type=int, default=100, help='Dimensionality of the hidden state for the LSTM')
    parser.add_argument('--bidir', action='store_true', help='Use Bidirectional LSTM')
    parser.add_argument('--mlp', action='store_true', help='Use MLP baseline')

    # TRAIN PARAMS
    parser.add_argument('-e', '--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('-b', '--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('-l', '--lr', type=float, default=0.0003, help='Learning rate')
    parser.add_argument('--weight-decay', '--wd', type=float, default=0, help='L2 penalty weight decay')
    parser.add_argument('-s', '--seed', type=int, default=23, help='Random seed')

    # OTHER
    parser.add_argument('-r', '--run_dir', default='runs/', help='Base dir for run directories')
    parser.add_argument('-f', '--force', action='store_true', help='Rerun already present experiments')

    parser.set_defaults(bidir=False, eval=False, force=False, mlp=False)
    args = parser.parse_args()
    
    # Run folder
    run_dir = 'detector_{}_h{}_{}_{}_b{}_lr{}_wd{}_e{}_s{}'.format(
        'mlp' if args.mlp else 'bi' if args.bidir else 'uni',
        args.hidden,
        'cos' if args.distance == 'cosine' else 'euc',
        'med' if 'medoids' in args.centroids else 'centr',
        args.batch_size,
        args.lr,
        args.weight_decay,
        args.epochs,
        args.seed
    )

    args.run_dir = os.path.join(args.run_dir, run_dir)
    
    if args.eval and not os.path.exists(args.run_dir):
        print('No run to evaluate:', args.run_dir)
        sys.exit(1)

    params = copy.deepcopy(vars(args))
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

    train_data, val_data, test_data = split_adversarials(args.adversarials)
    train_data = OrigAdvDataset(train_data, orig_transform=transform)
    val_data = OrigAdvDataset(val_data, orig_transform=transform, return_paths=True)    

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
        ckpt_path = os.path.join(args.run_dir, 'ckpt', 'best_model.pth')
        if not os.path.exists(ckpt_path):
            print('No pretrained model found:', ckpt_path)
            sys.exit(1)
            
        print('Loading:', ckpt_path)
        ckpt = torch.load(ckpt_path)
        detector.load_state_dict(ckpt['detector'])
        
        test_data = OrigAdvDataset(test_data, orig_transform=transform, return_paths=True)
        test_cache = 'cache/cache_test_{}_{}.pth'.format('cos' if args.distance == 'cosine' else 'euc', 'med' if 'medoids' in args.centroids else 'centr')
        test_paths, test_data = precompute_embeddings(features_state, test_data, model, args, return_paths=True, cache=test_cache)
        test_loader = DataLoader(test_data, shuffle=False, pin_memory=True, num_workers=8, batch_size=args.batch_size)

        test_preds, test_targets, test_metrics_names, test_metrics = evaluate(test_loader, test_paths, detector, args, return_predictions=True)
        
        test_metrics_dict = dict(zip(test_metrics_names, test_metrics.tolist()))
        test_metrics = pd.DataFrame(test_metrics_dict, index=[0])
        test_metrics_file = os.path.join(args.run_dir, 'test.csv')
        test_metrics.to_csv(test_metrics_file)
        
        test_preds = pd.DataFrame({'image': test_paths, 'pred': test_preds, 'target': test_targets})
        test_preds_file = os.path.join(args.run_dir, 'predictions.csv')
        print('Saving test predictions:', test_preds_file)
        test_preds.to_csv(test_preds_file)
        sys.exit(0)

    if not os.path.exists(args.run_dir):
        os.makedirs(args.run_dir)
        print('Running:', args.run_dir)
    elif not args.force:
        print('Skipping:', args.run_dir)
        sys.exit(0)

    ckpt_dir = os.path.join(args.run_dir, 'ckpt')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    params_file = os.path.join(args.run_dir, 'params.csv')
    params.to_csv(params_file, index=False)
    with pd.option_context('display.width', None, 'max_columns', None):
        print(params)

    log_file = os.path.join(args.run_dir, 'log.csv')
    log = pd.DataFrame()

    optimizer = Adam(detector.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    weights = train_data.weights
    sampler = WeightedRandomSampler(weights, len(weights))
    
    # Precompute Embeddings for train and val set once
    print('Precomputing embedded features: TRAIN')
    train_cache = 'cache/cache_train_{}_{}.pth'.format('cos' if args.distance == 'cosine' else 'euc',
                                     'med' if 'medoids' in args.centroids else 'centr')
    train_data = precompute_embeddings(features_state, train_data, model, args, cache=train_cache)
    print('Precomputing embedded features: VAL')
    val_cache = 'cache/cache_val_{}_{}.pth'.format('cos' if args.distance == 'cosine' else 'euc',
                                       'med' if 'medoids' in args.centroids else 'centr')
    val_paths, val_data = precompute_embeddings(features_state, val_data, model, args, return_paths=True, cache=val_cache)

    train_loader = DataLoader(train_data, sampler=sampler, pin_memory=True, batch_size=args.batch_size)
    val_loader = DataLoader(val_data, shuffle=False, pin_memory=True, batch_size=args.batch_size)

    # Train loop
    best = torch.zeros(3)
    progress = trange(1, args.epochs + 1)
    for epoch in progress:
        progress.set_description('TRAIN')
        train(train_loader, detector, optimizer, args)
        progress.set_description('EVAL')
        val_metrics_names, val_metrics = evaluate(val_loader, val_paths, detector, args)

        val_metrics_dict = dict(zip(val_metrics_names, val_metrics.tolist()))
        log = log.append(pd.DataFrame(val_metrics_dict, index=[pd.Timestamp('now')]))
        log.to_csv(log_file)

        if best[2] < val_metrics[2]:  # keep best macro-AUC
            ckpt_path = os.path.join(ckpt_dir, 'best_model.pth')
            torch.save({
                'detector': detector.state_dict(),
                'optimizer': optimizer.state_dict(),
                'metrics': val_metrics_dict
            }, ckpt_path)

        best = torch.max(val_metrics, best)

