#!/bin/bash

IMAGENET=/home/fabio/SLOW/Datasets/ImageNet

# 1. EXTRACT FEATURES FROM ILSVRC12 TRAINING SET
mkdir -p features/train/
for I in `ls ${IMAGENET}/train/`; do # e.g. I = n12998815
    OUT="features/train/$I.h5"
    #if [ ! -f $OUT ]; then
        python extract.py ${IMAGENET}/train/$I $OUT
    #else
    #    echo "Skipping: $I"
    #fi
done

# 2. COMPUTE CENTROIDS
python centroids.py features/train ilsvrc12_centroids.h5
