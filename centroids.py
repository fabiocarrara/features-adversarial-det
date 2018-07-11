import os
import argparse

import h5py
import numpy as np
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

def compute_centroid_medoid(features):
    c = np.mean(features, axis=0)
    distances = pdist(features, metric='euclidean')
    distances = squareform(distances)
    c_idx = distances.sum(axis=1).argmin()
    m = features[c_idx]
    return c, m

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute centroids from extracted features.')
    parser.add_argument('features_folder', type=str, help='Path to folder containing features.')
    parser.add_argument('output_file', type=str, help='Output file containing computed centroids.')
    parser.add_argument('-d', '--distance', default='euclidean',
                        help='Distance metric for computing medoids (see scipy.spatial.distance.pdist for choices).')

    args = parser.parse_args()

    features_files = [i for i in os.listdir(args.features_folder) if i.endswith('.h5')]
    features_files.sort()
    features_files = [os.path.join(args.features_folder, i) for i in features_files]
    
    with h5py.File(features_files[0], 'r') as features:
        layer_names = sorted(features.keys())
    
    centroids = []
    for class_features in tqdm(features_files):
        with h5py.File(class_features, 'r') as features:
            for layer in layer_names:
                centroids.append( compute_centroid_medoid(features[layer]) )
    
    centroids, medoids = zip(*centroids)
    with h5py.File(args.output_file, 'w') as out:
        for layer, centroid, medoid in zip(layer_names, centroids, medoids):
            out.create_dataset('centroids/{}'.format(layer), data=centroid)
            out.create_dataset('medoids/{}'.format(layer), data=medoid)
    
