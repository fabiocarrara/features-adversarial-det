import argparse
import os

import numpy as np


def main(args):
    csv = ('{:02d}.csv'.format(i) for i in range(1, 13))
    csv = (os.path.join(args.data_dir, i) for i in csv)
    data = (np.loadtxt(i, delimiter=',', dtype=np.float32) for i in csv)
    data = np.stack(data)

    labels = os.path.join(args.data_dir, 'successfulAttacks.csv')
    labels = np.loadtxt(labels, delimiter=',', dtype=int)

    if not args.out.endswith('.npz'):
        args.out.append('.npz')

    print ('Saving:', args.out)
    np.savez(args.out, x=data, y=labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse dissimilarity CSVs')
    parser.add_argument('data_dir', help='Directory containing dissimilarity CSVs')
    parser.add_argument('out', help='Parsed data output name')
    args = parser.parse_args()
    main(args)
