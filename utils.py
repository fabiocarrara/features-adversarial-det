import argparse
import os
from collections import Counter
from random import getrandbits

import glob2
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from torch.utils.data import Dataset, ConcatDataset
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS


def is_image_file(filename):
    return any(filename.lower().endswith(extension) for extension in IMG_EXTENSIONS)


def get_attack(path):
    if not path.endswith('.npz'):
        return 'auth'
    return os.path.basename(path).split('_')[1]
    

class OrigAdvDataset(Dataset):
    def __init__(self, paths, adv_transform=None, orig_transform=None, return_paths=False, return_label=True):
        self.paths = paths
        self.labels = [1 if i.endswith('.npz') else 0 for i in self.paths]
        self.attacks = [get_attack(i) for i in self.paths]
        self.adv_transform = adv_transform
        self.orig_transform = orig_transform
        self.return_paths = return_paths
        self.return_label = return_label
        
        a_counts = Counter(self.attacks)
        
        ntotal = len(self.paths)
        nadv = sum(self.labels)
        norig = ntotal - nadv
        
        self.weights = [(ntotal / a_counts[a]) for a in self.attacks]
        
    def __len__(self):
        return len(self.paths)
        
    def __getitem__(self, item):
        path = self.paths[item]
        label = self.labels[item]
        
        if label:  # adversarial
            image = torch.from_numpy(np.load(path)['img'])
            if self.adv_transform:
                image = self.adv_transform(image)
        else:
            image = default_loader(path)
            if self.orig_transform:
                image = self.orig_transform(image)        

        # path, image, label
        ret = [None, image, None]

        if self.return_paths:
            ret[0] = path
            
        if self.return_label:
            ret[2] = label
        
        ret = [r for r in ret if r is not None]
        ret = ret[0] if len(ret) == 1 else ret
        
        return ret
        

def split_adversarials(csv_filename):
    adv = pd.read_csv(csv_filename)
    ids = adv.ID.unique()
    
    adv['Split'] = adv.ID.map(lambda x: 'train' if x in ids[:700] else 'val' if x in ids[700:800] else 'test')    
    
    train = adv[adv['Split'] == 'train']
    val = adv[adv['Split'] == 'val']
    test = adv[adv['Split'] == 'test']

    print('TRAIN: {}'.format(len(train)))
    print('VAL  : {}'.format(len(val)))
    print('TEST : {}'.format(len(test)))
    
    # print(adv.pivot_table(index=['Attack','Eps'], columns='Split', values='ID', aggfunc='count'))
    
    train_files = train.Path.tolist() + train.OriginalPath.unique().tolist()
    val_files = val.Path.tolist() + val.OriginalPath.unique().tolist()
    test_files = test.Path.tolist() + test.OriginalPath.unique().tolist()
    
    return train_files, val_files, test_files
    

class NpzDataset(Dataset):
    def __init__(self, folder, transform=None, return_paths=False, return_label=None):
        self.folder = folder
        self.files = [i for i in os.listdir(folder) if i.endswith('.npz')]
        self.transform = transform
        self.return_paths = return_paths
        self.return_label = return_label

    def __getitem__(self, item):
        path = self.files[item]
        abspath = os.path.join(self.folder, path)
        image = torch.from_numpy(np.load(abspath)['img'])
        if self.transform:
            image = self.transform(image)

        if self.return_paths:
            ret = [path, image]
            if self.return_label is not None:
                ret.append(self.return_label)
            return ret

        if self.return_label is not None:
            return image, self.return_label

        return image

    def __len__(self):
        return len(self.files)


class ImageDataset(Dataset):
    def __init__(self, folder, transform=None, return_paths=False, return_label=None):
        self.folder = folder
        self.images = [i for i in os.listdir(folder) if is_image_file(i)]
        self.transform = transform
        self.return_paths = return_paths
        self.return_label = return_label

    def __getitem__(self, item):
        path = self.images[item]
        abspath = os.path.join(self.folder, path)
        image = default_loader(abspath)
        if self.transform:
            image = self.transform(image)

        if self.return_paths:
            ret = [path, image]
            if self.return_label is not None:
                ret.append(self.return_label)
            return ret

        if self.return_label is not None:
            return image, self.return_label

        return image

    def __len__(self):
        return len(self.images)


class AdvDataset(Dataset):

    def __init__(self, orig_folders, adv_folders, orig_transform=None, adv_transform=None, return_paths=False):
        adv_datasets = [NpzDataset(f, transform=adv_transform, return_paths=return_paths, return_label=1) for f in
                        adv_folders]
        adv_subset = ConcatDataset(adv_datasets)

        orig_datasets = [ImageDataset(f, transform=orig_transform, return_paths=return_paths, return_label=0) for f in
                         orig_folders]
        orig_subset = ConcatDataset(orig_datasets)
        # orig_idx = torch.randperm(len(orig_subset))[:len(adv_subset)]
        # orig_subset = Subset(orig_subset, orig_idx)

        n_orig = len(orig_subset)
        n_adv = len(adv_subset)
        n_total = n_orig + n_adv
        print('Orig / Adv: {} ({:3.2%}) / {} ({:3.2%})'.format(n_orig, (n_orig / n_total), n_adv, (n_adv / n_total)))

        self.combined = ConcatDataset([orig_subset, adv_subset])
        self.weights = (n_total / n_orig,) * n_orig + (n_total / n_adv,) * n_adv

    def __getitem__(self, item):
        return self.combined[item]

    def __len__(self):
        return len(self.combined)


# Global var for idx -> class mapping
classes = None


def _load_classes():
    global classes
    if classes is None:
        with open('ilsvrc12_synset_to_human_label_map.txt', 'r') as f:
            classes = [i.rstrip() for i in f.readlines()]
            classes.sort()
        classes.append('Adversarial example')


def idx2desc(idx):
    _load_classes()
    return classes[idx]


# https://gist.github.com/kingspp/3ec7d9958c13b94310c1a365759aa3f4
# Pyfunc Gradient Function
def _py_func_with_gradient(func, inp, Tout, stateful=True, name=None,
                           grad_func=None):
    """
    PyFunc defined as given by Tensorflow
    :param func: Custom Function
    :param inp: Function Inputs
    :param Tout: Ouput Type of out Custom Function
    :param stateful: Calculate Gradients when stateful is True
    :param name: Name of the PyFunction
    :param grad: Custom Gradient Function
    :return:
    """
    # Generate random name in order to avoid conflicts with inbuilt names
    rnd_name = 'PyFuncGrad-' + '%0x' % getrandbits(30 * 4)

    # Register Tensorflow Gradient
    tf.RegisterGradient(rnd_name)(grad_func)

    # Get current graph
    g = tf.get_default_graph()

    # Add gradient override map
    with g.gradient_override_map(
            {"PyFunc": rnd_name, "PyFuncStateless": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)


def convert_pytorch_model_to_tf(model, device, out_dims=None):
    """
    Convert a pytorch model into a tensorflow op that allows backprop
    :param model: A pytorch nn.Model object
    :param out_dims: The number of output dimensions (classes) for the model
    :return: A model function that maps an input (tf.Tensor) to the
    output of the model (tf.Tensor)
    """
    torch_state = {
        'logits': None,
        'x': None,
    }
    if not out_dims:
        out_dims = list(model.modules())[-1].out_features

    def _fprop_fn(x_np):
        x_tensor = torch.tensor(x_np, requires_grad=True)

        torch_state['x'] = x_tensor
        torch_state['logits'] = model(x_tensor.to(device))
        return torch_state['logits'].cpu().detach().numpy()

    def _bprop_fn(x_np, grads_in_np):
        _fprop_fn(x_np)
        grads_in_tensor = torch.tensor(grads_in_np).to(device)

        # Run our backprop through our logits to our xs
        loss = torch.sum(torch_state['logits'] * grads_in_tensor)
        loss.backward()
        return torch_state['x'].grad.cpu().numpy()

    def _tf_gradient_fn(op, grads_in):
        return tf.py_func(_bprop_fn, [op.inputs[0], grads_in],
                          Tout=[tf.float32])

    def tf_model_fn(x_op):
        out = _py_func_with_gradient(_fprop_fn, [x_op], Tout=[tf.float32],
                                     stateful=True,
                                     grad_func=_tf_gradient_fn)[0]
        out.set_shape([None, out_dims])
        return out

    return tf_model_fn
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Summarize Experimens')
    parser.add_argument('run_dir', default='runs/', help='Folder containing runs')
    args = parser.parse_args()
    
    logs = glob2.glob(os.path.join(args.run_dir, '**', 'log.csv'))
    runs = map(os.path.dirname, logs)
    
    results = []
    for run in runs:
        log = os.path.join(run, 'log.csv')
        log = pd.read_csv(log)
        idxmax = log.auc.idxmax()
        best = log.iloc[idxmax]
        
        params = os.path.join(run, 'params.csv')
        params = pd.read_csv(params)        
        params['auc'] = best.auc
        params['eer'] = best.eer_accuracy
        params['macro_auc'] = best.macro_auc
        
        results.append(params)
    
    results = pd.concat(results, axis=0)
    # conds = (~results.bidir) & (results.epochs == 250) & (results.seed == 23) & ~results.mlp
    # results = results[conds]
    
    unique_cols = results.apply(pd.Series.nunique) == 1
    non_unique_cols = ~unique_cols
    
    with pd.option_context('display.width', None, 'max_columns', None):
        # Print differences
        # print(results.loc[:, non_unique_cols].sort_values('auc', ascending=False))

        print(results.loc[:, non_unique_cols].pivot_table(index=['hidden', 'mlp', 'bidir'], columns=['centroids', 'distance'], values='auc'))

        # Print common
        print(results.loc[:, unique_cols].iloc[0])

        # print(results.pivot_table(index=['mlp', 'centroids', 'distance'], columns='macro_auc'))
