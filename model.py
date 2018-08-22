import os

import h5py

import torch

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
from tqdm import tqdm


class Embedder(nn.Module):
    def __init__(self, centroids, distance='euclidean', **kwargs):
        super(Embedder, self).__init__()
        
        with h5py.File(centroids, 'r') as f:
            self.distance = distance
            self.centroids = [torch.tensor(i[()]).to(kwargs['device']) for i in f.values()]

    def embed(self, x, c):
        if self.distance == 'euclidean':
            return torch.stack([(i - c).norm(dim=1) for i in x])
        elif self.distance == 'cosine':
            return torch.stack([F.cosine_similarity(i.unsqueeze(0), c) for i in x])

    def forward(self, x):
        embedded = [self.embed(x_i, c_i) for x_i, c_i in zip(x, self.centroids)]
        embedded = torch.stack(embedded)
        return embedded
        

def precompute_embeddings(features_state, data, model, args, return_paths=False, cache=None):

    if cache and os.path.exists(cache):
        cache = torch.load(cache)
        X = cache['X']
        y = cache['y']
        p = cache['p']
    else:
        embed = Embedder(**vars(args))
        loader = DataLoader(data, shuffle=False, pin_memory=True, num_workers=8, batch_size=args.batch_size)
        p, X, y = [], [], []
        with torch.no_grad():
            for batch in tqdm(loader):
                Xb, yb = batch[-2:]
                if return_paths:
                    p.extend(batch[0])
                y.append(yb.cpu())
                model(Xb.to(args.device))
                features = list(features_state.values())
                embedded = embed(features).permute(1, 0, 2)
                X.append(embedded.cpu())

        del embed, loader
        X = torch.cat(X)
        y = torch.cat(y)
        if cache:
            torch.save({'X': X, 'y': y, 'p': p}, cache)
    
    dataset = TensorDataset(X, y)
    if return_paths:
        return p, dataset
        
    return dataset
    

class Detector(nn.Module):

    def __init__(self, hidden=100, bidir=False, mlp=False, **kwargs):
        super(Detector, self).__init__()

        self.mlp = mlp
        if mlp:
            self.fc = nn.Sequential( nn.Linear(17000, hidden), nn.ReLU(), nn.Dropout(0.5) )
        else:
            self.lstm = nn.LSTM(1000, hidden, bidirectional=bidir, batch_first=True)
            
        out_size = hidden * 2 if (bidir and not mlp) else hidden
        self.classifier = nn.Linear(out_size, 1)

    def forward(self, x):        
        if self.mlp:
            x = x.reshape(x.shape[0], -1)
            output = self.fc(x)
        else:
            output, _ = self.lstm(x)
            output = output[:, -1]
        return self.classifier(output)
        

