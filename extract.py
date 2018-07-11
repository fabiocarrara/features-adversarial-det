import argparse
import h5py
import itertools

import torch

import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision.transforms import ToTensor, Compose, Resize, CenterCrop, Normalize
from tqdm import tqdm

from utils import ImageDataset


def main(args):

    transform = Compose([
                    Resize(256),
                    CenterCrop(224),
                    ToTensor(),
                    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
                    
    dataset = ImageDataset(args.image_folder, transform=transform)
    n_images = len(dataset)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=32, pin_memory=True, num_workers=8)

    resnet = torchvision.models.resnet50(pretrained=True).to(args.device)
    features = h5py.File(args.out)

    blocks = itertools.chain(resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4, (resnet.avgpool,))
    blocks = list(blocks)
    n_features = len(blocks)
    block_idx = dict(zip(blocks, map('{:02d}'.format, range(n_features))))

    n_processed = 0

    def extract(self, input, output):
        extracted = output
        if extracted.ndimension() > 2:
            extracted = F.avg_pool2d(extracted, extracted.shape[-2:]).squeeze(3).squeeze(2)

        block_num = block_idx[self]
        batch_size, feature_dims = extracted.shape
        dset = features.require_dataset(block_num, (n_images, feature_dims), dtype='float32')
        # extracted = extracted.to('cpu')
        dset[n_processed:n_processed + batch_size, :] = extracted.to('cpu')

    for b in blocks:
        b.register_forward_hook(extract)

    with torch.no_grad():
        for x in tqdm(dataloader):
            resnet(x.to(args.device))
            n_processed += x.shape[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract features from image folder with pretrained net')
    parser.add_argument('image_folder', help='Directory containing images')
    parser.add_argument('out', help='Extracted features output file')
    args = parser.parse_args()

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    main(args)
