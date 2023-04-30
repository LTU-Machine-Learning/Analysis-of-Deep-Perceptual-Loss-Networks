# Perceptual Similarity
This folder contains the implementation of the perceptual similarity experiments from [The Unreasonable Effectiveness of Deep Features as a Perceptual Metric](https://richzhang.github.io/PerceptualSimilarity) along with extensions to run experiments using different loss networks in the file ```experiments.py```.
The folder also contains the implementation of [Deep Perceptual Similarity is Adaptable to Ambiguous Contexts](https://arxiv.org/abs/2304.02265) in the file ```augmentations.py```.

## General
Running either experiments will automatically download the BAPPS dataset into ```datasets/BAPPS``` if it is not there already.
The files will run experiments with all combinations of the provided parameters and will skip over any experiments for which data has already been collected.
Logs and checkpoints for perceptual and adaptive experiments are stored in ```logs/perceptual_metric``` and ```logs/adaptive_metric``` respectively.
For each metric setup there is a folder with subfolders for each repetition of that containing their performance and any checkpoint that have been produced.
To gather all information into a single ````.csv``` file simply run ```python perceptual_info_gatherer.py``` or ```python adaptive_info_gatherer.py``` which will aggregate the results into ```logs/perceptual_metric/results.csv``` and ```logs/adaptive_metric/results.csv```.

### Parameters
These parameters are shared between ```experiment.py``` and ```augmentations.py``` 
Besides ```--loss_nets```, ```--extraction_layers```, ```--no_multilayer```, ```--variants```, and ```--repetitions``` all parameters are by default the same as those used in the experiments.
Using ```--use_experiment_setup``` will automatically use the loss networks and layers that were used in the paper (and overwrite those parameters if they are provided).
Running ```python experiments.py --help``` will give a list of all the available parameters.

#### Loss Networks
The ```--loss_nets <loss networks to use>``` parameter determine which Torchivison model pretrained on ImageNet to use a loss network in the experiments.
While multiple loss networks can be specified it is recommended to only run with one at a time since you would likely use different layers for different architectures anyways.

#### Feature layers
The ```--extraction_layers <first layer> <layer layers(optional)> ``` parameter indicate at which layers features should be extracted for deep perceptual loss.
Each loss network will have features extracted at these layers to use for similarity calculation.
If the ```--no_multilayer``` parameter is used instead one metric will be created per layer that only does extraction from that layer.

#### Variants
The ```--variants <variant> <more variants(optional)>``` parameter selects if and how the metric will be trained.
The available options are ```baseline```, ```lin```, ```fine-tune```, and ```scratch```.
```baseline``` creates a metric that is pretrained on ImageNet and applies it without additional training.
```lin``` creates a pretrainted metric and trains a layer of positive scalars which the extracted deep features are multiplied by.
```fine-tune``` creates a pretrained metric and trains both the loss network and the scalars.
```scratch``` creates a metric with a randomly initialized loss network which is trained with the scalars.

#### Comparison Function
The ```--metric_fun``` parameter selects how the deep features from the loss networks should be compared.
```spatial``` compares every extracted feature from one input with the corresponding features of another.
```mean``` averages the extracted features of each channel before comparison.
```sort``` ranks the extracted features of each channel so the first is the greatest activation and so on before comparison.
```spatial+mean``` and ```spatial+sort``` uses both comparisons and sums the result.


#### Repetitions
The ```--repetitions <int>``` parameter decides how many times each non-deterministic experiment should be repeated.


## Perceptual Similarity Experiments
Running ```experiments.py``` as shown below will create deep percpetual similarity metrics based on the given loss network, potentially train them on BAPPS, and then evalute them on the BAPPS dataset.

```bash
python experiments.py --loss_nets <the loss networks to use> --layers <the layers to use> --variants <training procedure>
```

## Adaptive Similarity Experiments
Running ```augmentations.py``` as shown below will create deep perceptual similarity metrics based on the given loss network, potentially train them to adapt to random distortion rankings on the images from the given dataset, evaluate the metric for the random distortion rankings on the given datasets as well as on BAPPS.
Running adaptive similarity experiments will automatically download the given datasets which can be SVHN, STL-10, or both.
These datasets will be downloaded into ````datasets/SVHN``` and ```datasets/STL-10```.

```bash
python augmentation.py --train_data <dataset for training> --test_data <one or more datasets for testing> --loss_nets <the loss networks to use> --layers <layers to use> --variants <training procedure> --rankings <the number of random rankings to train and test with>
```

### Adaptive Similarity Parameters
These parameters are specific to ```augmentation.py```.

#### Train and test data
Adaption training is performed by augmenting images and learning which augmentations should be seen as more similar.
The metrics are then evaluated on how well they have adapted to the ranking of augmentaitons.
Images from two datasets are available for this purpose ```svhn``` and ```stl10```.
```--train_data``` specifies which dataset to use for training
```--test_data``` specifies which dataset or both to use for testing.

#### Rankings and augmentations
To test adaptibility a metric should be tested on a number of different rankings of the augmentations.
The number to test is given by the ```--rankings``` parameter.
If not all augmentations should be used in rankings this can be specified with the ```--augmentations``` parameter followed by the augmentations to use.