import argparse
import os
import re

import glob2 as glob
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from torchvision.datasets.folder import default_loader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

# Crazy regexp. Matches: ID, Attack, Conf, LNorm, Eps, EpsIter, Iters, Src, Dest
from utils import idx2desc

adv_regex = re.compile(r".*\/([a-z0-9]*)\.png_([a-z\-]+)(?:_conf([0-9\.]+))?(?:_L([a-z0-9]+))?(?:_eps([a-z0-9]+))?(?:_epsi([a-z0-9]+))?(?:_i([a-z0-9]+))?_src([0-9]+)_dst([0-9]+).npz")

transform = Compose([Resize(256),
                     CenterCrop(224),
                     ToTensor(),
                     Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                     ])


def parse_filename(a):
    match = adv_regex.match(a)
    if match is None:
        return (a,) + (None,) * 8

    return (a,) + match.groups()


def perturbation(a):
    orig = default_loader(a.OriginalPath)
    orig = transform(orig)
    adv = torch.from_numpy(np.load(a.Path)['img'])
#    L = 2 if a.LNorm is None else float(a.LNorm)
#    if np.isinf(L):
#        eps = (orig - adv).abs().max().item()
#    else:
#        eps = torch.norm(orig - adv, L).item()
    eps = (orig - adv).abs().max().item()

    return eps


def to01(i):
    i = i.numpy()
    i *= np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    i += np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    i = i.transpose((1, 2, 0))
    return i


def show(a):
    fig, axes = plt.subplots(1, 3, sharey=True)
    orig = transform(default_loader(a.OriginalPath))
    axes[0].imshow(to01(orig))
    adv = np.load(a.Path)['img']
    axes[1].imshow(to01(adv))
    diff = np.absolute(adv - orig)
    diff /= diff.max()
    axes[2].imshow(diff)
    print(idx2desc(a.Src), '-->', idx2desc(a.Dest))
    plt.pause(1)


if __name__ == '__main__':

    tqdm.pandas()

    parser = argparse.ArgumentParser(description='Compute stats of adversarial examples')
    parser.add_argument('--image-folder', default='images', help='Directory containing images (both original and adversarial)')
    args = parser.parse_args()

    # original = glob.glob(os.path.join(args.image_folder, '**', '*.png'))
    # original.sort()
    # original = pd.DataFrame(original, columns=['Path'])
    # original['ID'] = original['Path'].map(lambda x: os.path.basename(x)[:-4])
    # original = original.set_index('ID')
    # print(original)

    adversarial = glob.glob(os.path.join(args.image_folder, '**', '*.npz'))
    adversarial.sort()
    adversarial = map(parse_filename, adversarial)
    adv_cols = ('Path', 'ID', 'Attack', 'Conf', 'LNorm', 'Eps', 'EpsIter', 'Iters', 'Src', 'Dest')
    adversarial = list(adversarial)
    adversarial = pd.DataFrame.from_records(adversarial, columns=adv_cols)
    adversarial['Src'] = pd.to_numeric(adversarial['Src']).astype(int)
    adversarial['Dest'] = pd.to_numeric(adversarial['Dest']).astype(int)
    adversarial['OriginalPath'] = os.path.join(args.image_folder, 'original', '') + adversarial['ID'].astype(str) + '.png'
    adversarial = adversarial.set_index('ID')

    # print(adversarial)

    # a = adversarial[adversarial['Attack'] == 'cw'].head()

    adversarial['Perturbation'] = adversarial.progress_apply(perturbation, axis=1)
    adversarial.to_csv('adversarial.csv')

    # a.apply(show, axis=1)


