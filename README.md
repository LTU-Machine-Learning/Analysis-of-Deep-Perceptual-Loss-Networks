# Systematic Analysis of Deep Percpetual Loss Networks

This repository contains the implementation for evaluating loss networks for deep perceptual loss and similarity for the work ["A Systematic Performance Analysis of Deep Perceptual Loss Networks: Breaking Transfer Learning Conventions"](https://arxiv.org/abs/2302.04032).
This work analyzes how different pretrained architectures and feature extraction layer affect performance for training with deep percpetual loss or when measuring perceptual similarity with deep perceptual similarity metrics.
To do this four previous works have been implemented and reevaluated with host of new loss networks.
The four implemented works and the folder with their implementation is detailed below:
- [Pretraining Image Encoders without Reconstruction via Feature Prediction Loss](https://ieeexplore.ieee.org/abstract/document/9412239) can be found in ```src/perceptual_autoencoders```
-  [Beyond the Pixel-Wise Loss for Topology-Aware Delineation](https://openaccess.thecvf.com/content_cvpr_2018/html/Mosinska_Beyond_the_Pixel-Wise_CVPR_2018_paper.html) can be found in ```src/topology_aware_delineation```
- [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://link.springer.com/chapter/10.1007/978-3-319-46475-6_43) can be found in ```src/super_res```
- [The Unreasonable Effectiveness of Deep Features as a Perceptual Metric](https://richzhang.github.io/PerceptualSimilarity) can be found in ```src/perceptual_metric```
Each implementation has its own README detailing how to run those experiments.

The repository also contains the implementation for the work [Deep Perceptual Similarity is Adaptable to Ambiguous Contexts](https://arxiv.org/abs/2304.02265) which is also located in ```src/perceptual_metric```.

## Structure
The experiment implementations can be found in ```src/```.
Datasets are automatically downloaded and placed in ```datasets/```.
Checkpoints (and some logs) downloaded and produced during experiments are stored in ```checkpoints/```.
Logs are primarily saved in ```logs/```.
The README files in each of the implementation folders have more detailed information about how each implementation uses these folders.

## Setup

This repository has been tested with the Python packages listed below.
The version details the version with which they have been tested, but it is likely that other versions will work as well.

Used by all implementations
```bash
pytorch 1.10.2
torchvision 0.11.3
numpy 1.12.2
```

Used by some implementations
```bash
pytorch-lightning 1.9.1
wandb 0.12.11
scikit-learn 1.2.2
scikit-image 0.20.0
scipy 1.8.0
opencv-python 4.7.0.72
kornia 0.6.10
```

Some implementations also use logging with Weights and Biases.
```super-res``` always uses WandB, while it is optional with ```percpetual_autoencoder``` and ```perceptual_metric``` (depending on which logger you use).
You will be prompted to connect to your WandB account first time this happens (thprugh their own API).

## Referencing

If you use this repository as part of a scientific work please cite the original works the implementations are based on as well as this work as detailed in the bibtex below:
```bash
@inproceedings{pihlgren2023systematic,
    doi = {10.48550/ARXIV.2302.04032},
    author = {Pihlgren, Gustav Grund and Nikolaidou, Konstantina and Chhipa, Prakash Chandra and Abid, Nosheen and Saini, Rajkumar and Sandin, Fredrik and Liwicki, Marcus},
    title = {A Systematic Performance Analysis of Deep Perceptual Loss Networks: Breaking Transfer Learning Conventions},
    publisher = {arXiv preprint},
    year = {2023},
}
```


If you are using the experiments in ```src/perceptual_metric/augmentations.py``` plese cite that work according to the bibtex below additionally/instead:
```bash
@inproceedings{pihlgren2023deep,
    doi = {10.48550/ARXIV.2304.02265},
    author = {Pihlgren, Gustav Grund and Sandin, Fredrik and Liwicki, Marcus},
    title = {Deep Perceptual Similarity is Adaptable to Ambiguous Contexts},
    publisher = {arXiv preprint},
    year = {2023},
}
```
