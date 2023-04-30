# Delineation

### Training on Massachusetts Roads Dataset
Running the file as shown below will train a delineation model on the MRD dataset with the specified loss network.
The dataset will be automatically downloaded into ```datasets/MRD``` if it has not already.
Downloading can take a long time due to unoptimized requests (~1h), so consider manually collecting the dataset from ```http://www.cs.toronto.edu/~vmnih/data/``` and manually placing it into ```datasets/MRD/{train|val|test}/{input|output}```.
Trained models are stored in ```checkpoints/delineation``` though models trained with the same loss network and extraction layers will overwrite each other.

```bash
python main.py --loss_net <architecture> --layers <layer to extract from>
```

### Testing on Massachusetts Roads Dataset
Running the testing file will evaluate all trained model checkpoints stored in ```checkpoints/delineation``` on the MRD test set.
The Correctness, Completeness and Quality (delineation measurements) of each model on the test set will be recorded in ```logs/delineation/results.csv```.
The testing will use the provided arguments which may be different from those used when the models were trained which can lead to issues (though as long as the same arguments are used for training and testing it should be fine).

```bash
python test.py
```

### Training and testing parameters
Besides ```--loss_net``` and ```--layer``` all other parameters are by default the same as those used in the experiments.

#### Help
```python main.py --help``` and ```python test.py --help``` will give an overview of all the arguments available for training and testing.