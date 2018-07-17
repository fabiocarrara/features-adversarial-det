#!/bin/bash

IMAGENET=/home/fabio/SLOW/Datasets/ImageNet

# 1. EXTRACT FEATURES FROM ILSVRC12 TRAINING SET
extract_features() {
    mkdir -p features/train/
    for I in `ls ${IMAGENET}/train/`; do # e.g. I = n12998815
        OUT="features/train/$I.h5"
        if [ ! -f $OUT ]; then
            python extract.py ${IMAGENET}/train/$I $OUT
        else
            echo "Skipping: $I"
        fi
    done
}

# 2. COMPUTE CENTROIDS
compute_centroids() {
    python centroids.py features/train ilsvrc12_centroids.h5
    python centroids.py -m features/train ilsvrc12_medoids.h5
}

# 3. GENERATE ADVERSARIALS
generate_adversarials() {
    mkdir -p images/{fgsm,iter,m-iter,pgd}

    for L in -1 1 2; do
    for EPS in 20 40 60 80; do
    for ATTACK in 'fgsm' 'iter' 'm-iter' 'pgd'; do

        python attack.py -a $ATTACK -e $EPS -l $L images/original images/$ATTACK

    done
    done
    done
}

# extract_features
# compute_centroids
generate_adversarials