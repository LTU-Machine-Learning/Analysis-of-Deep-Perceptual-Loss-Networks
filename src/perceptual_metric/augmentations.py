# Library imports
import random
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from PIL import Image
import argparse
import pathlib
import csv

# File imports
from dataset_collector import dataset_collector
from loss_networks import FeatureExtractor, extractor_collector, \
    ExtractionModifier, architecture_attributes
from perceptual_metric.experiment import DistanceMetric, Learner, score_jnd

from workspace_path import home_path
log_dir = home_path / 'logs/adaptive_metric'
checkpoint_dir = home_path / 'checkpoints/adaptive_metric'

rotate = transforms.RandomRotation(degrees=(30,330))
translate = transforms.RandomAffine(degrees=0, translate=(0.5,0.5))
darken = transforms.ColorJitter(brightness=(0.1,0.5))
color = transforms.ColorJitter(hue=0.5)
blur = transforms.GaussianBlur(kernel_size=(11,21), sigma=(4,10))
zoom_in = transforms.RandomAffine(degrees=0, scale=(1.1,2))
augments = {
    'rotate' : rotate,
    'translate' : translate,
    'darken' : darken,
    'color' : color,
    'blur' : blur,
    'zoom_in' : zoom_in,
} 


class AugmentRankingDataset(Dataset):
    '''
    A Dataset wrapper to return two random augmentations of a reference image
    along with an value of which augment is lower in the augmentation ranking
    Args:
        dataset (Dataset): Another Dataset to take images from
        transforms ([nn.Module]): Ranked transforms to sample from
        img_position (int/None): The position of imgs in dataset return tuple
        pre_transform (nn.Module): Transform to apply before random transform
        post_transform (nn.Module): Transform to apply after random transform
    '''

    def __init__(
        self, dataset, transforms, img_position=None, 
        pre_transfrom=None, post_transform=None
    ):
        self.dataset = dataset
        self.img_position = img_position
        self.transforms = transforms
        self.pre_transform = pre_transfrom
        self.post_transform = post_transform
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        items = self.dataset[idx]
        img = items if self.img_position is None else items[self.img_position]
        if not self.pre_transform is None:
            img = self.pre_transform(img)
        choices = torch.randperm(len(self.transforms))
        p0 = self.transforms[choices[0]](img)
        p1 = self.transforms[choices[1]](img)
        if not self.post_transform is None:
            p0 = self.post_transform(p0)
            p1 = self.post_transform(p1)

        return {
            'p0' : p0,
            'p1' : p1,
            'ref' : img,
            'judge' : torch.tensor(0.0 if choices[0]<choices[1] else 1.0)
        }


def augmentation_ranking_experiment(
    train_set, test_sets, augment_ranking, loss_network_path, variant,
    sync_loss=1, repetitions=1, lpips_normalize=False, channel_norm=True,
    batch_size=256, metric_fun='spatial', logger=CSVLogger, num_workers=1
):
    '''
    Given a loss network and whether or not to add calibration weights,
    creates a metric with that loss network. That metric is trained (if 
    applicable) to create similarity scores such that some specified
    augmentations are more similar than others according to the specified
    ranking. The metric is then tested on another given dataset with the same
    augmentations. Finally the metric is evaluated on the BAPPS validation
    splits to test how adapting it affects performance on percpetual similarity
    Args:
        train_set (dict): Kwargs for collecting training dataset
        test_sets ([dict]): List of Kwargs for collecting test dataset
        augment_ranking ([str]): Ranked list of augmentation names
        loss_network_path (str): Path to the loss network to use as a metric
        variant (str): Which variant to use (baseline, lin, scratch, tune)
        sync_loss (int): Sync loss and score, -1=when score<0.5, >0=# of epochs
        repetitions (int): How many times to repeat training
        lpips_normalize (bool): Whether to use torchvision or lpips normalize
        channel_norm (bool): Whether to normalize features in channel dimension
        batch_size (int): Size of training and validation batches
        metric_fun (str): How to compare features "spatial[+]", "sort", "mean"
        logger (f()->Logger): PyTorch Lightning Logger
        num_workers (int): Number of worker processes used to load data
    '''
    # Parameters
    batch_size = 1024
    epochs = 10
    validation_fraction = 0.2
    logit = 'large' #'small' #'none'
    modification = (
        metric_fun[metric_fun.index('+')+1:]
        if '+' in metric_fun 
        else metric_fun
    )
    modification_return = 'concat' if '+' in metric_fun else 'new'
    weights = variant != 'baseline'

    # Make project name for WandbLogger
    project_name = 'Augmentation-Adaptation-Perceptual-Metrics'

    # Create an experiment id
    norm_name = 'lpips-norm' if lpips_normalize else 'imagenet-norm'
    experiment_name = (
        f'{train_set["dataset"]}_{"-".join([t["dataset"]for t in test_sets])}_'
        f'{"-".join(augment_ranking)}_{pathlib.Path(loss_network_path).stem}_'
        f'{variant}_{norm_name}_{metric_fun}_sync-loss-{sync_loss}_'
        f'{"channel-norm" if channel_norm else "no-channel-norm"}'
    )

    # Whether to run on the GPU
    gpus = None
    if torch.cuda.is_available():
        gpus = 1

    # Create the image preprocessing
    pre_transform = transforms.Compose([
        transforms.Resize(size=(64,64)),
        transforms.ToTensor(),
    ])
    if lpips_normalize:
        post_transform = transforms.Normalize(
            (-0.030,-0.088,-0.188), (0.458,0.448,0.450)
        )
    else:
        post_transform= transforms.Normalize(
            (0.485,0.456,0.406), (0.229,0.224,0.225)
        )
    
    # Prepare augmentations
    transforms_rankings = [augments[augment] for augment in augment_ranking]

    # Collect the train and test data and set up augmentations
    data = AugmentRankingDataset(
        dataset=dataset_collector(**train_set),
        transforms=transforms_rankings,
        img_position=0,
        pre_transfrom=pre_transform,
        post_transform=post_transform
    )
    validation_length = int(len(data) * validation_fraction)
    train_data, val_data = random_split(
        data, [len(data) - validation_length, validation_length]
    )
    test_datas = [
        AugmentRankingDataset(
            dataset=dataset_collector(**test_set),
            transforms=transforms_rankings,
            img_position=0,
            pre_transfrom=pre_transform,
            post_transform=post_transform
        ) for test_set in test_sets
    ]

    # Collect the BAPPS evaluation data
    eval_2afc_datas = []
    subsplits_2afc = [
        'cnn', 'color', 'deblur', 'frameinterp', 'superres', 'traditional'
    ]
    for subsplit_2afc in subsplits_2afc:
        eval_2afc_data = dataset_collector(
            'BAPPS', 'val', subsplit=subsplit_2afc,
            image_transform=transforms.Compose([pre_transform,post_transform])
        )

        eval_2afc_datas.append(eval_2afc_data)
    eval_jnd_data = dataset_collector(
        'BAPPS', 'jnd/val',
        image_transform=transforms.Compose([pre_transform,post_transform])
    )
        
    # Train once per repetition
    for version in range(repetitions):
        
        # Create a path to this version
        filepath = log_dir / f'{experiment_name}/version_{version}'

        # Check if this experiment has already been done
        if (filepath / 'test_results.csv').exists():
            print(f'{experiment_name}/version_{version} is already tested')
            continue

        # Create the metric to use
        #TODO: Make factors possible
        loss_network = torch.load(loss_network_path)
        if metric_fun != 'spatial':
            loss_network = ExtractionModifier(
                extractor=loss_network,
                modification=modification,
                return_policy=modification_return,
                shape_policy='keep'
            )
        metric = DistanceMetric(
            loss_network, weights=weights, channel_norm=channel_norm
        )
        
        # Create the Learner to handle training
        learner = Learner(
            metric, logit=logit, lpips_normalize=lpips_normalize,
            sync_loss=sync_loss
        )

        # Create logger
        checkpoint_callback = ModelCheckpoint(
            dirpath = filepath
        )
        kw_log = {'project':project_name} if logger == WandbLogger else {}
        log = logger(
            save_dir=str(log_dir),
            name=f'{experiment_name}_version{version}',
            **kw_log
        )

        # Setting up the Pytorch-Lightning trainer
        trainer = pl.Trainer(
            logger=log,
            callbacks=[checkpoint_callback],
            max_epochs=10,
            gpus=gpus,
        )
        
        # Check if this experiment has already been done
        if (filepath / 'test_results.csv').exists():
            continue

        # Whether to train or not
        if weights:

            # Loading from checkpoint, if any
            ckpt_path = None
            if filepath.exists() and len([
                f for f in filepath.iterdir()
                if f.is_file() and f.suffix=='.ckpt'
            ]) > 0:
                ckpt_path = sorted([
                    f for f in filepath.iterdir()
                    if f.is_file() and f.suffix=='.ckpt'
                ])[-1]

            # Dataloaders for train and validation data
            train_loader = DataLoader(
                train_data, batch_size=batch_size, shuffle=True,
                num_workers=num_workers
            )
            val_loader = DataLoader(
                val_data, batch_size=batch_size, num_workers=num_workers
            )

            # Training using the trainer
            trainer.fit(
                model = learner,
                train_dataloaders = train_loader,
                val_dataloaders = val_loader,
                ckpt_path = ckpt_path
            )
        elif version > 0:
            # No need for multiple versions of deterministic experiment
            break

        # Create dataloaders for testing and BAPPS evaluation
        test_loaders = [
            DataLoader(
                test_data, batch_size=batch_size, num_workers=num_workers
            ) for test_data in test_datas
        ]
        jnd_loader = DataLoader(
            eval_jnd_data, batch_size=batch_size, num_workers=num_workers
        )
        eval_loaders_2afc = [
            DataLoader(
                ev, batch_size=batch_size, num_workers=num_workers
            ) for ev in eval_2afc_datas
        ]

        # Run the tests on all test sets, jnd, and all 2afc splits
        if gpus is not None:
            learner.cuda()
        test_results = trainer.test(
            model=learner, dataloaders=eval_loaders_2afc+test_loaders
        )
        jnd_result = score_jnd(jnd_loader, learner).item()

        # Save the test results to a .csv file
        row_names = subsplits_2afc + [t['dataset'] for t in test_sets]
        if not filepath.exists():
            filepath.mkdir(parents=True)
        with open(filepath / 'test_results.csv', 'w') as save_file:
            writer = csv.writer(save_file, delimiter=' ')
            for result in test_results:
                for row in result.items():
                    new_row = [
                        row[0][:-1] + row_names[int(row[0][-1])], row[1]
                    ]
                    writer.writerow(new_row)
            writer.writerow(['test_jnd/jnd', jnd_result])
        #Force logger to end experiment to stop WandB from using the same entry 
        if logger == WandbLogger:
            trainer.logger.experiment.finish()

def main():
    '''
    Given parameters, trains and tests loss networks as perceptual metrics of
    similarity as defined by random ordering of augmentation on the applicable
    datasets
    '''
    # Create parser and parse input
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_data',
        choices=['svhn', 'stl10'],
        type=str,
        help='Dataset to use for training'
    )
    parser.add_argument(
        '--test_data',
        choices=['svhn', 'stl10'],
        type=str,
        nargs='+',
        help='Datasets to use for testing'
    )
    parser.add_argument(
        '--augmentations',
        type=str,
        nargs='+',
        default=augments.keys(),
        choices=augments.keys(),
        help='Which augmentations to include in ranking to train for'
    )
    parser.add_argument(
        '--loss_nets',
        type=str,
        nargs='+',
        help='The pretrained loss networks to use for similarity calculations'
    )
    parser.add_argument(
        '--extraction_layers',
        type=int,
        default=[1,4,7,9,11],
        nargs='+',
        help='The different layers to feature extract from'
    )
    parser.add_argument(
        '--no_multilayer',
        action='store_true',
        help='Train one model per extraction layer instead of one with all'
    )
    parser.add_argument(
        '--variants',
        type=str,
        default=['lin'],
        nargs='+',
        choices=['baseline', 'lin', 'scratch', 'tune'],
        help='Which variant to use, baseline is pretrained, LPIPS otherwise'
    )
    parser.add_argument(
        '--repetitions',
        type=int,
        default=1,
        help='How many times to repeat each non-static experiment'
    )
    parser.add_argument(
        '--rankings',
        type=int,
        default=1,
        help='How many different rankings to evaluate'
    )
    parser.add_argument(
        '--sync_loss_epochs',
        type=int,
        default=-1,
        help='For how many epochs to sync loss and score (-1=when score<0.5)'
    )
    parser.add_argument(
        '--lpips_normalize',
        action='store_true',
        help='Use image normalization parameters from original paper'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1024,
        help='What batch size to use'
    )
    parser.add_argument(
        '--use_experiment_setup',
        action='store_true',
        help='Overrides loss network parameters to run predefined experiments'
    )
    parser.add_argument(
        #TODO: Enable this to use any factors other than 1.0 and None
        '--metric_fun',
        type=str,
        default=['spatial'],
        nargs='+',
        choices=['spatial', 'mean', 'sort', 'spatial+mean', 'spatial+sort'],
        help='Select which function(s) to calculate perceptual similarity with'
    )
    parser.add_argument(
        '--no_channel_norm',
        action='store_true',
        help='Experiments will not use unit normalization over channels'
    )
    parser.add_argument(
        '--logger',
        type=str,
        default='csv',
        help='Which logger to use'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=1,
        help='How many worker process to use to load data (number of threads)'
    )
    parser.add_argument(
        '--random_seed',
        action='store_true',
        help='Experiments will use a random seed instead of default one'
    )
    args = parser.parse_args()

    # Get the logger to use
    available_loggers = {
        'csv' : CSVLogger,
        'wandb' : WandbLogger,
        'tensorboard' : TensorBoardLogger
        # ADD MORE LOGGERS HERE
    }
    if args.logger in available_loggers:
        logger = available_loggers[args.logger]
    else:
        raise ValueError(
            f'{args.logger} does not match any available logger. '
            f'Select from {available_loggers.keys()} or add a new in the code'
        )

    # Collect the augmentations to use and generate random rankings
    if not args.random_seed:
        random.seed(0)
        torch.manual_seed(0)
    rankings = [
        random.sample(args.augmentations, k=len(augments))
        for _ in range(args.rankings)
    ]

    # Dictionary for collecting training and validation data
    train_sets = {
        'svhn' : {'dataset' : 'SVHN', 'split' : 'train'},
        'stl10' : {'dataset' : 'STL10', 'split' : 'train'},
    }
    if not args.train_data in train_sets:
        raise ValueError(
            f'{args.train_data} not an available dataset: {train_sets.keys()}'
        )
    train_set = train_sets[args.train_data]

    # Dictionary for collecting testing data
    test_sets = {
        'svhn' : {'dataset' : 'SVHN', 'split' : 'test'},
        'stl10' : {'dataset' : 'STL10', 'split' : 'test'},
    }
    tests_set = []
    for test_set in args.test_data:
        if not test_set in test_sets:
            raise ValueError(
                f'{args.test_data} not an available dataset: {test_sets.keys()}'
            )
        tests_set.append(test_sets[test_set])

    # Collect all loss networks to be tested
    for variant in args.variants:
        pretrained = variant != 'scratch'
        frozen = variant in ['baseline', 'lin']
        for fun in args.metric_fun:

            if args.use_experiment_setup:
                runs = sum([
                    (
                        [(net, [layer]) for layer in attributes['layers_in_experiments']]
                        if args.no_multilayer 
                        else [(net, attributes['layers_in_experiments'])]
                    )
                    for net, attributes 
                    in architecture_attributes.items() 
                    if attributes['used_in_experiments']
                ], [])
            else:
                runs = sum([
                    (
                        [(net, [layer]) for layer in args.extraction_layers]
                        if args.no_multilayer else [(net, args.extraction_layers)]
                    )
                    for net in args.loss_nets if net in architecture_attributes
                ], [])

            for net, layers in runs:
                loss_network_path = extractor_collector(
                    FeatureExtractor,
                    architecture=net,
                    layers=layers,
                    pretrained=pretrained,
                    frozen=frozen,
                    flatten_layer=False,
                    normalize_in=False
                )
                for ranking in rankings:
                    augmentation_ranking_experiment(
                        train_set=train_set,
                        test_sets=tests_set,
                        augment_ranking=ranking,
                        loss_network_path=loss_network_path,
                        variant=variant,
                        sync_loss=args.sync_loss_epochs,
                        repetitions=args.repetitions,
                        lpips_normalize=args.lpips_normalize,
                        channel_norm=not args.no_channel_norm,
                        batch_size=args.batch_size,
                        metric_fun=fun,
                        logger=logger,
                        num_workers=args.num_workers
                    )

# When this file is executed independently, execute the main function
if __name__ == "__main__":
    main()