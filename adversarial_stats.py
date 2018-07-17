import argparse
import os
import re

import glob2 as glob
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from torchvision.datasets.folder import default_loader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

# Crazy regexp. Matches: ID, Attack, LNorm, Eps, EpsIter, Iters, Src, Dest
adv_regex = re.compile(r".*\/([a-z0-9]*)\.png_([a-z]+)(?:_L([a-z0-9]+))?(?:_eps([a-z0-9]+))?(?:_epsi([a-z0-9]+))?(?:_i([a-z0-9]+))?_src([0-9]+)_dst([0-9]+).npz")

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
    eps = torch.norm(orig - adv, 2)
    return eps


if __name__ == '__main__':

    tqdm.pandas()

    parser = argparse.ArgumentParser(description='Generate adversarial examples')
    parser.add_argument('--image_folder', default='images', help='Directory containing images (both original and adversarial)')
    args = parser.parse_args()

    original = glob.glob(os.path.join(args.image_folder, '**', '*.png'))
    original.sort()
    original = pd.DataFrame(original, columns=['Path'])
    original['ID'] = original['Path'].map(lambda x: os.path.basename(x)[:-4])
    original = original.set_index('ID')
    print(original)

    adversarial = glob.glob(os.path.join(args.image_folder, '**', '*.npz'))
    adversarial.sort()
    adversarial = map(parse_filename, adversarial)
    adv_cols = ('Path', 'ID', 'Attack', 'LNorm', 'Eps', 'EpsIter', 'Iters', 'Src', 'Dest')
    adversarial = list(adversarial)
    adversarial = pd.DataFrame.from_records(adversarial, columns=adv_cols)
    print(adversarial['ID'])
    adversarial['OriginalPath'] = os.path.join(args.image_folder, 'original', '') + adversarial['ID'].astype(str) + '.png'
    adversarial = adversarial.set_index('ID')

    print(adversarial)

    a = adversarial.head()
    print(a)
    print(a.progress_apply(perturbation, axis=1))

