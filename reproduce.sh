#!/bin/bash

set -e

IMAGENET=/path/to/ImageNet

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
    mkdir -p images/{fgsm,iter,m-iter,pgd,lbfgs}

    for EPS in 20 40 60 80; do
    for ATTACK in 'fgsm' 'iter' 'm-iter' 'pgd'; do
        python attack.py -a $ATTACK -e $EPS -l -1 images/original images/$ATTACK
    done
    done

    python attack.py -a lbfgs images/original images/lbfgs
    python adversarial_stats.py --image-folder images/
}

# 4. TRAIN DETECTORS
train_detectors() {
    mkdir -p cache/

    for CENTROIDS in 'ilsvrc12_centroids.h5' 'ilsvrc12_medoids.h5'; do
    for DISTANCE in 'cosine' 'euclidean'; do
    for MODEL in '' '--mlp'; do
        python train_detector.py adversarials.csv $MODEL $BIDIR -c $CENTROIDS -d $DISTANCE --force
        python train_detector.py adversarials.csv $MODEL $BIDIR -c $CENTROIDS -d $DISTANCE --eval
    done
    done
    done
}

# 5. MAKE PLOTS
make_plots() {
    python plot_rocs.py -a adversarial.csv -o rocs.pdf runs/
    python plot_single_roc.py -a adversarial.csv -o roc_per_attack.pdf runs/detector_uni_h100_cos_med_b128_lr0.0003_wd0.0_e100_s23/predictions.csv
}

extract_features
compute_centroids
generate_adversarials
train_detectors
make_plots