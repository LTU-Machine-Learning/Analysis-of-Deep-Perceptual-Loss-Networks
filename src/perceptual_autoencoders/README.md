# Dimensionality Reduction with Autoencoders

### Training and Testing
Running ```experiments.py``` will download the appropriate datasets (if necessary), train autoencoders with the specified loss network and parameters using auxilliary part of that dataset.
For each autoencoder a number of different MLP architectures will be trained to predict labels on the training set with the learned embeddings as input.
The performance of each MLP trained with autoencoder is then recorded and used to evalute the autoencoder.
The file will run an experiment for all combinations of the provided parameters and will reuse already trained autoencoders and skip over any experiment for which data has already been gathered.

The datasets will be downloaded into ```datasets/SVHN``` and ```datasets/STL-10```.
The trained autoencoders are saved in ```checkpoints/perceptual_autoencoders``` with one folder per autoencoder setup that has subfolders for each time that experiment has been repeated.
The results for each MLP is saved in a subfolder of the checkpoint folder with the autoencoder used.
To gather all the data and results run ```autoencoder_info_gatherer.py``` which will aggregate all data into a single file at ```checkpoints/perceptual_autoencoders/results.csv```.

```bash
python experiment.py --data <dataset to use (svhn or stl10)> --loss_nets <the loss networks to use> --loss_net_layers <the layers to use>
```

### Parameters

#### Datasets

The ```--data <svhn|stl10>``` parameter lets you select whether to use the SVHN or STL-10 dataset for evaluation.

#### Loss networks
The ```--loss_nets <loss networks to use>``` parameter determine which Torchivison model pretrained on ImageNet to use a loss network in the experiments.
While multiple loss networks can be specified it is recommended to only run with one at a time since you would likely use different layers for different architectures anyways.

#### Feature layers
The ```--loss_net_layers <first layer> <layer layers(optional)> ``` parameter indicate at which layers features should be extracted for deep perceptual loss.
For each layer a loss network is created that extracts from that layer.

#### Repetitions
The ```--repetitions <int>``` parameter decides how many times each non-deterministic experiment should be repeated.

#### Running experiments in paper
Using ```--use_experiment_setup``` will cause the experiments to be run using the same pretrained models and feature extraction layers as in the paper. Using this flag will cause the program to ignore the ```--loss_net``` and ```--loss_net_layers``` parameters.