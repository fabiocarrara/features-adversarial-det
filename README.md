## Adversarial Detection in Dissimilarity Space

This code trains a model for adversarial detection based on intermediate features mapped into a dissimilarity space.

### Requirements

 - [PyTorch 0.4](https://pytorch.org/)
 - sklearn
 - tqdm
 
### Training

 - Prepare data as specified in the README in the [data/](data/) folder.
 - Run training script:
 ```shell
 python train.py data/dissim.npz > log.txt
 ```
 
### Compare to MTAP article

 - Run training on MTAP data:
 ```shell
 python train.py --data-selection mtap -o predictions.npz data/dissim.npz > log.txt
 ```
 - Run analysis:
 ```shell
 python plot.py predictions.npz
 ```