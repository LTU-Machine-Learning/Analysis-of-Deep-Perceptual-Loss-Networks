# Super Resolution

### Conda Env
To create the conda environment:

```bash
conda create --name super_res_env --file super_res_env.txt
```

### Training with MSCOCO
Running the training file as shown below will train a super-resolution model with the COCO2014 dataset with the specified loss network.
The dataset will be automatically downloaded into ```datasets/COCO2014/``` if it has not been already.
Trained models will be stored in ```checkpoints/super_resolution/```.

```bash
python train.py --res 4 --loss_net <architecture> --feature_layers <layers as numbers separated by spaces>
```

### Test on Super Resolution Datasets
Running the testing file will evaluate trained models (if they exist) on the Set 5, Set 14, and BSD100 (test set of BSD300) datasets.
The datasets will be automatically downloaded into ```datasets/``` if they have not been already.
Currently the results (PSNR and SSIM scores) are only printed to the terminal.
The original image and their downscaled and then reupscaled versions for the latest model can be found in ```logs/super_resolution/```.

```bash
python test.py --res 4 --loss_net <architecture> --feature_layers <layers as numbers separated by spaces>
```

### Training and testing parameters

#### Help
To get an overview of all available parameters and flags run ```python test.py --help```

#### Loss network
The ```--loss_net <architecture>``` flag determine which Torchivison model pretrained on ImageNet to use a loss network in the experiments.
Available networks can be found in the ```src/loss_network.py``` file in the ```architecture_attributes``` dictionary.
Most Torchvision models are available.

#### Feature layers
The ```--feature_layers <first layer> <layer layers(optional)> ``` flag indicate at which layers features should be extracted for deep perceptual loss (one model trained per layer).
Skipping this flag will cause the program to use the same feature layers as in the paper, though for loss networks not evaluated in the paper this flag has to be included.

#### Resolution factor
The code can be used for resolution factor of 4 or 8 and can be adjusted with ```--res 4``` or ```--res 8```, respectively.

#### Running experiments in paper
Using the ```--use_experiment_setup``` flag will cause the experiments to be run using the same pretrained models and feature extraction layers as in the paper. Using this flag will cause the program to ignore the ```--loss_net``` and ```--feature_layers``` flags.
