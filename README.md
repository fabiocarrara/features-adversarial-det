## Adversarial examples detection in features distance spaces

This repo contains code to reproduce the experiments presented in "Adversarial examples detection in features distance spaces".
The code trains models for adversarial detection based on intermediate features of the attacked classifier embedded into dissimilarity spaces.

### Requirements
The main requirements are:
- Python 3
- [pytorch 0.4](https://pytorch.org/) + torchvision 
- tensorflow 1.8 + [cleverhans](https://github.com/tensorflow/cleverhans)

and can be installed with:
```sh
pip3 install -r requirements.txt
``` 
You will also need the following datasets to replicate the experiments:
- [ILSVRC'12 (ImageNet) TRAIN split](http://www.image-net.org/download-images)
- [NIPS 2017 Adversarial Competition DEV dataset](https://github.com/tensorflow/cleverhans/tree/master/examples/nips17_adversarial_competition/dataset)
 
### Steps to reproduce experiments

- Create the folder `images/original` in the project folder and put the NIPS DEV images in it
- Modify the `IMAGENET` variable in [reproduce.sh](reproduce.sh) to point to the folder containing the ILSVRC'12 dataset (the script will point to the `$IMAGENET/train/` folder)
- Run [reproduce.sh](reproduce.sh)
 ```shell
 ./reproduce.sh
 ```
 
The [reproduce.sh](reproduce.sh) bash script runs all the steps needed to reproduce the experiments presented in the paper, that is:
1. Features extraction from ILSVRC'12 TRAIN dataset
2. Class centroid / medoid computation
3. Generation of adversarial examples 
4. Training of multiple detectors
5. Reproducing ROC plots