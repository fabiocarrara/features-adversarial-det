import os
import argparse

import h5py
import numpy as np
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm


def compute_centroid(features, medoid=False):

    if medoid:
        distances = pdist(features, metric='euclidean')
        distances = squareform(distances)
        c_idx = distances.sum(axis=1).argmin()
        c = features[c_idx]
    else:
        c = np.mean(features, axis=0)

    return c


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute centroids from extracted features.')
    parser.add_argument('features_folder', type=str, help='Path to folder containing features.')
    parser.add_argument('output_file', type=str, help='Output file containing computed centroids.')
    parser.add_argument('-m', '--medoid', action='store_true', help='Compute medoids if set.')
    parser.add_argument('-d', '--distance', default='euclidean',
                        help='Distance metric for computing medoids (see scipy.spatial.distance.pdist for choices).')

    args = parser.parse_args()

    features_files = [i for i in os.listdir(args.features_folder) if i.endswith('.h5')]
    features_files.sort()
    features_files = [os.path.join(args.features_folder, i) for i in features_files]

    with h5py.File(features_files[0], 'r') as features:
        layer_names = sorted(features.keys())

    print('COMPUTING')
    class_centroids = []
    for class_features in tqdm(features_files):
        with h5py.File(class_features, 'r') as features:
            centroids = [compute_centroid(features[layer], medoid=args.medoid) for layer in tqdm(layer_names)]
            class_centroids.append(centroids)

    layer_centroids = zip(*class_centroids)

    print('SAVING:', args.output_file)
    with h5py.File(args.output_file, 'w') as out:
        for layer, centroids in tqdm(zip(layer_names, layer_centroids)):
            centroids = np.stack(centroids)
            out.create_dataset('{}'.format(layer), data=centroids)
