# Library imports
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import os
import csv
import argparse
import pathlib
from itertools import combinations_with_replacement, product

# File imports
from perceptual_autoencoders.encoders import FourLayerCVAE, \
    FeaturePredictorCVAE, FeatureAutoencoder, PerceptualFeatureToImgCVAE, \
    FeatureToImgCVAE, encode_data
from perceptual_autoencoders.predictors import FullyConnected
from loss_networks import FeatureExtractor, architecture_attributes, \
    extractor_collector
from dataset_collector import dataset_collector, datasets

# Define log and checkpoint directory
from workspace_path import home_path
checkpoint_dir = home_path / 'checkpoints/perceptual_autoencoders'
log_dir = home_path / 'logs/perceptual_autoencoders'
log_dir.mkdir(parents=True, exist_ok=True)


def collect_autoencoder(
    experiment_name,
    dataset,
    validation_fraction=0.2,
    epochs=50,
    batch_size=256,
    network=FourLayerCVAE,
    z_dim=64,
    gamma=0,
    loss_net=None,
    version=0,
    gpu=True,
    beta_factor=None,
    logger=CSVLogger,
    num_workers=1,
    **kwargs
):
    '''
    Returns a trained autoencoder with the given parameters, training it if it
    doesn't already exist
    Args:
        experiment_name (str): Name of experiment and save folder
        dataset (dict): kwargs for collecting dataset
        validation_fraction (float): Fraction of dataset to use for validation
        epochs (int): Maximum number of epochs to train each autoencoder for
        batch_size (int): Size of the batches
        network (f()->nn.Module): Autoencoder architecture
        z_dim (int): The dimensionality of the embedding space (z)
        gamma (float): The gamma value for variational loss (0=non-variational)
        loss_net (str/None): Path to loss network
        version (int): Which version to collect if running multiple experiments
        gpu (bool): Whether to use GPUs if they are available
        beta_factor (float/None): beta value for channel sorted perceptual loss
        logger (f()->Logger): PyTorch Lightning Logger
        num_workers (int): Number of worker processes used to load data
        kwargs (dict): Additional parameters to be passed along to pl.Trainer
    Returns (str, bool): The autoencoder path and whether it was trained anew
    '''

    # Set checkpoint folder and create if it doesn't exist
    filepath = checkpoint_dir / f'{experiment_name}/version_{version}'
    filepath.mkdir(parents=True, exist_ok=True)

    # If this autoencoder has already been trained, return it
    if (filepath / 'best.ckpt').exists():
        return (f'{filepath}', False)

    # Collect datasets
    data = dataset_collector(**dataset)
    validation_length = int(len(data) * validation_fraction)
    train_data, val_data = random_split(
        data, [len(data) - validation_length, validation_length]
    )
    train_loader = DataLoader(
        train_data, batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_data, batch_size, num_workers=num_workers
    )

    # Calculate input size
    input_size = (train_data[0][0].size()[1], train_data[0][0].size()[2])

    # Create callback for checkpointing
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath = filepath
    )

    # Create callback for early stopping
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=max(10,epochs / 20)
    )

    # Create logger
    log = logger(
        save_dir=log_dir,
        name=f'{experiment_name}_version{version}'
    )
    
    # Loading from checkpoint, if one exists
    ckpt_path = None
    if filepath.exists() and len([
        f for f in filepath.iterdir()
        if f.is_file() and f.suffix=='.ckpt'
    ]) > 0:
        ckpt_path = sorted([
            f for f in filepath.iterdir()
            if f.is_file() and f.suffix=='.ckpt'
        ])[-1]

    # Initialize an autoencoder model with the given parameters
    model = network(
        input_size=input_size,
        z_dimensions=z_dim,
        variational=(gamma != 0),
        gamma=gamma,
        perceptual_net_path=loss_net,
        beta_factor=beta_factor
    )

    # Check whether to run on GPUs
    gpus = None
    if gpu and torch.cuda.is_available():
        gpus = 1

    # Initialize Trainer
    trainer = pl.Trainer(
        callbacks=[early_stop_callback, checkpoint_callback],
        logger=log,
        max_epochs=epochs,
        gpus=gpus,
        **kwargs
    )

    # Train the model
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path = ckpt_path
    )

    # Force logger to end experiment to stop WandB from using the same entry
    if logger == WandbLogger:
        trainer.logger.experiment.finish()

    # Save best model and checkpoint
    ckpts = [
        filepath/f'{f}' for f in os.listdir(filepath)
        if f.endswith('.ckpt')
    ]
    #TODO: Remove or raise error below?
    if len(ckpts) == 0:
        print('!')
        ckpts = [
            checkpoint_dir / f'{experiment_name}/{f}'
            for f in os.listdir(checkpoint_dir / experiment_name)
            if f.endswith('.ckpt')
        ]
    best_value = None
    best_ckpt = None
    for ckpt in ckpts:
        a = torch.load(ckpt, map_location='cpu')
        a_b = [b for b in a['callbacks'] if 'ModelCheckpoint' in b][0]
        a_value = a['callbacks'][a_b]['best_model_score']
        if (best_value is None or a_value < best_value):
            best_value = a_value
            best_ckpt = ckpt
    best = torch.load(best_ckpt, map_location='cpu')
    best['state_dict'].perceptual_net = None
    torch.save(best, f=filepath / 'best.ckpt')

    # Delete non-best checkpoints to save space
    for ckpt in ckpts:
        if not ckpt.name == 'best.ckpt':
            ckpt.unlink()

    # Return trained autoencoder
    return (f'{filepath}', True)


def run_experiments(
    ae_dataset,
    train_dataset,
    test_dataset,
    input_size,
    validation_fraction=0.2,
    ae_epochs=50,
    predictor_epochs=500,
    ae_batch_size=256,
    predictor_batch_size=256,
    ae_net=FourLayerCVAE,
    loss_net=None,
    z_dim=64,
    gamma=0,
    beta_factor=None,
    repetitions=1,
    logger=CSVLogger, 
    gpu=False,
    num_workers=1,
    clean_storage=False,
    **kwargs
):

    '''
    Collects (and trains if needed) an autoencoder with the given parameters
    and then evaluates that autoencoder by training and testing a number of
    predictor MLPs to make predictions based on the learned features.
    Args:
        ae_dataset (dict): kwargs for collecting dataset used for AE training
        train_dataset (dict): kwargs to collect dataset for predictor training
        test_dataset (dict): kwargs to collect dataset for predictor testing
        input_size (int, int): Height and width of the images in the dataset
        validation_fraction (float): Fraction of dataset to use for validation
        ae_epochs (int): Maximum number of epochs to train the autoencoder for
        predictor_epochs (int): Max number of epochs for predictor training
        ae_batch_size (int): Size of the batches during autoencoder training
        predictor_batch_size (int): Batch size for traingin the predictors
        ae_net (f()->nn.Module): Autoencoder architecture
        loss_net (str/None): Path to loss network for autoencoder training
        z_dim (int): The dimensionality of the embedding space (z)
        gamma (float): The gamma value for variational loss (0=non-variational)
        beta_factor (float/None): beta value for channel sorted perceptual loss
        repetitions (int): The number of times to repeat the experiment
        logger (f()->Logger): PyTorch Lightning Logger
        gpu (bool): Whether to use GPUs if they are available
        num_workers (int): Number of worker processes used to load data
        clean_storage (bool): Whether to delete the trained autoencoders after
        kwargs (dict): Additional parameters to be passed along to pl.Trainer
    Returns (str, bool): The autoencoder path and whether it was trained anew
    '''

    # Setup variables and losses
    train_loader = None
    val_loader = None
    test_loader = None
    onehot = None
    output_shape = (1, )
    if datasets[train_dataset['dataset']]['output_format'] == 'index':
        output_shape = datasets[train_dataset['dataset']]['nr_of_classes']
        onehot = output_shape

    loss_function = torch.nn.MSELoss()

    def losses(output, target):
        return {
            'loss': loss_function(output, target),
            'l1_distance': torch.mean(torch.norm(output - target, 1, dim=1)),
            'l2_distance': torch.mean(torch.norm(output - target, 2, dim=1)),
            'accuracy': torch.mean(
                torch.eq(
                    torch.max(output, 1)[1],
                    torch.max(target, 1)[1]
                ).float())
        }

    # Create architecture parameters
    architectures = [[], [32], [64], [32, 32], [64, 32], [64, 64], [128, 128]]
    [arch.append(output_shape) for arch in architectures] 
    hidden_func = nn.LeakyReLU
    out_func = None

    ae_parameters = {
        'ae_dataset' : ae_dataset['dataset'],
        'ae_input_size[0]' : str(input_size[0]),
        'ae_input_size[1]' : str(input_size[1]),
        'ae_epochs' : str(ae_epochs), 
        'ae_name' : ae_net.__name__,
        'ae_z' : str(z_dim),
        'ae_gamma' : str(gamma),
        'loss_net' : 'None' if loss_net is None else str(loss_net.stem)
    }
    if not beta_factor is None:
        ae_parameters['beta_factor'] = str(beta_factor)
    ae_name = '_'.join(ae_parameters.values())

    # Run experiment once per repetition
    for version in range(repetitions):

        # If this experiment has already been run completely, skip it
        skip = True
        for arch in architectures:
            ex = '_'.join([ae_name] + [
                str('-'.join([str(a) for a in [input_size]+arch])),
                hidden_func.__name__ if not hidden_func is None else 'None',
                out_func.__name__ if not out_func is None else 'None',
                str(predictor_epochs)
            ])
            path = checkpoint_dir/f'{ae_name}/version_{version}/{ex}/version_0'
            if not (path / 'results.csv').exists():
                skip = False
                break
        if skip:
            continue

        ae_dir, ae_new = collect_autoencoder(
            experiment_name = ae_name,
            dataset=ae_dataset,
            validation_fraction=validation_fraction,
            epochs=ae_epochs,
            batch_size=ae_batch_size,
            network=ae_net,
            z_dim=z_dim,
            gamma=gamma,
            loss_net=loss_net,
            version=version,
            gpu=gpu,
            beta_factor=beta_factor,
            logger=logger,
            num_workers=num_workers,
        )
        ae_path = pathlib.Path(f'{ae_dir}/best.ckpt')
        ae_ckpt = torch.load(ae_path, map_location='cpu')
        autoencoder = ae_net(
            input_size = input_size,
            z_dimensions = z_dim,
            variational = gamma != 0,
            gamma = gamma,
            perceptual_net_path = loss_net
        )
        autoencoder.load_state_dict(ae_ckpt['state_dict'])

        # Collecting the training, validation, and testing data at most once
        if train_loader is None or val_loader is None:
            data = dataset_collector(**train_dataset)
            validation_length = int(len(data) * validation_fraction)
            train_data, val_data = random_split(
                data, [len(data) - validation_length, validation_length]
            )
            train_loader = DataLoader(
                train_data, predictor_batch_size, num_workers=num_workers
            )
            val_loader = DataLoader(
                val_data, predictor_batch_size, num_workers=num_workers
            )
        if test_loader is None:
            test_data = dataset_collector(**test_dataset)
            test_loader = DataLoader(
                test_data, predictor_batch_size, num_workers=num_workers
            )

        # Encode the data with the autoencoder
        print(f'Encoding data using {ae_path}...')
        train_encoded = DataLoader(
            encode_data(
                autoencoder,
                train_loader,
                to_onehot=onehot
            ),
            batch_size=predictor_batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        val_encoded = DataLoader(
            encode_data(autoencoder, val_loader, to_onehot=onehot),
            batch_size=predictor_batch_size,
            num_workers=num_workers
        )
        test_encoded = DataLoader(
            encode_data(
                autoencoder,
                test_loader,
                to_onehot=onehot
            ),
            batch_size=predictor_batch_size,
            num_workers=num_workers
        )

        for architecture in architectures:
            
            # Create parameters and experiment fplder
            parameters = [
                str('-'.join([str(a) for a in [input_size]+architecture])),
                hidden_func.__name__ if not hidden_func is None else 'None',
                out_func.__name__ if not out_func is None else 'None',
                str(predictor_epochs)
            ]
            experiment_name = '_'.join([ae_name] + parameters)
            filepath = pathlib.Path(
                    f'{ae_dir}/{experiment_name}/version_0'
                )

            # Skip this predictor if results are already recorded
            if (filepath / 'results.csv').exists():
                continue

            # Initialize the predictor
            act_functs = [hidden_func] * (len(architecture) - 1) + [out_func]
            predictor = FullyConnected(
                input_size=z_dim,
                layers=architecture,
                activation_funcs=act_functs,
                loss_funcs=losses
            )


            # Create a checkpointing callbacl
            checkpoint_callback = ModelCheckpoint(
                monitor='val_loss',
                dirpath=filepath
            )

            # Create an early stopping callback
            early_stop_callback = EarlyStopping(
                monitor='val_loss',
                patience=max(10, predictor_epochs/20)
            )

            # Create a logger
            log = logger(
                save_dir=log_dir,
                name='_'.join(
                    [ae_name, str(version)] +
                    parameters +
                    [f'version0']
                )
            )
            log.log_hyperparams(ae_parameters)
            log.save()

            # Check whether to run on GPUs
            gpus = None
            if gpu and torch.cuda.is_available():
                gpus = 1

            # Initialize Trainer
            trainer = pl.Trainer(
                callbacks=[early_stop_callback,checkpoint_callback],
                logger=log,
                max_epochs=predictor_epochs,
                gpus=gpus,
                **kwargs
            )

            # Train the model
            trainer.fit(
                model=predictor,
                train_dataloaders=train_encoded,
                val_dataloaders=val_encoded,
                #ckpt_path=ckpt_path
            )

            # Run final validation and test the predictor
            val_results = trainer.validate(dataloaders=val_encoded)
            test_results = trainer.test(dataloaders=test_encoded)

            # Force logger to end experiment so WandB doesn't reuse entry
            if logger == WandbLogger:
                trainer.logger.experiment.finish()

            # Store results and log that this run has been evaluated
            with open(filepath / 'results.csv', 'w') as save_file:
                results = val_results + test_results
                writer = csv.DictWriter(
                    save_file,
                    delimiter=' ',
                    fieldnames=sum([list(r.keys()) for r in results], [])
                )
                res = {}
                writer.writeheader()
                for result in results:
                    res.update(result)
                writer.writerow(res)

            # Delete checkpoints to save space
            [
                (filepath/f'{f}').unlink() for f in os.listdir(filepath)
                if f.endswith('.ckpt')
            ]

        #Delete autoencoder checkpoint if desired
        if clean_storage and ae_new:
            ae_path.unlink()

def main():
    '''
    Given the autoencoder parameters and a dataset trains those autoencoders
    that are missing and then trains and tests the predictors specified by the
    predictor parameters for each autoencoer.
    '''
    # Create parser and parse input
    parser = argparse.ArgumentParser()
    parser.add_argument(
        # To add a dataset, append its name here and preprocessing later
        '--data',
        type=str,
        choices=['stl10', 'svhn'],
        required=True,
        help='The dataset to train and test on'
    )
    parser.add_argument(
        '--ae_epochs',
        type=int,
        default=50,
        help='Nr of epochs to train autoencoders for'
    )
    parser.add_argument(
        '--ae_batch_size',
        type=int,
        default=256,
        help='Size of autoencoder batches'
    )
    parser.add_argument(
        # To add an autoencoder, append its name here and preprocessing later
        '--ae_networks',
        type=str,
        default=['FourLayerCVAE'],
        nargs='+',
        choices=[
            'FourLayerCVAE', 'FeaturePredictorCVAE', 'FeatureAutoencoder',
            'PerceptualFeatureToImgCVAE', 'FeatureToImgCVAE'
        ],
        help='The different autoencoder networks to use'
    )
    parser.add_argument(
        '--ae_zs',
        type=int,
        default=[64],
        nargs='+',
        help='The different autoencoder z_dims to use'
    )
    parser.add_argument(
        '--ae_gammas',
        type=float,
        default=[0.0],
        nargs='+',
        help='The different autoencoder gammas to use'
    )
    parser.add_argument(
        '--ae_convergence',
        action='store_true',
        help='Whether to give a file with the validation losses per epoch'
    )
    parser.add_argument(
        '--loss_nets',
        type=str,
        default=['None'],
        nargs='+',
        help='The different loss networks to use for autoencoders'
    )
    parser.add_argument(
        '--loss_net_layers',
        type=int,
        default=[5],
        nargs='+',
        help='The different feature extraction layers to test'
    )
    parser.add_argument(
        '--predictor_epochs',
        type=int,
        default=500,
        help='Nr of epochs to train predictors for'
    )
    parser.add_argument(
        '--predictor_batch_size',
        type=int,
        default=256,
        help='Size of predictor batches'
    )
    parser.add_argument(
        '--ae_repetitions',
        type=int,
        default=1,
        help='How many AEs to train with each hyperparamter setting'
    )
    parser.add_argument(
        '--predictor_repetitions',
        type=int,
        default=1,
        help='How many predictors per AE and hyperparameter setting to train'
    )
    parser.add_argument(
        '--no_gpu',
        action='store_true',
        help='GPUs will not be used even if they are available'
    )
    parser.add_argument(
        '--use_experiment_setup',
        action='store_true',
        help='Overrides loss network parameters to run predefined experiments'
    )
    parser.add_argument(
        '--beta_factors',
        type=float,
        default=[None],
        nargs='+',
        help='Adds translation invariant term scaled by factor to the loss net'
    )
    parser.add_argument(
        '--logger',
        type=str,
        default='csv',
        help='Which logger to use'
    )
    parser.add_argument(
        '--clean_drive',
        action='store_true',
        help='Removes autoencoder models created during run to save space'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=1,
        help='How many worker process to use to load data (number of threads)'
    )
    # TODO: Implement
    # parser.add_argument(
    #    '--memory_wary', action='store_true',
    #    help='Will attempt to lower RAM usage (possibly at cost of speed)'
    # )
    # TODO: Add arguments to use non-default architectures and functions

    args = parser.parse_args()

    # Setup dicts for dataset collection, add code here to add new datasets
    print('Loading data for autoencoder training...')
    if args.data == 'stl10':
        input_size=(96, 96)
        transform = torchvision.transforms.ToTensor()
        ae_train_dict = {
            'dataset': 'STL10', 'split': 'unlabeled', 'transform': transform
        }
        predictor_train_dict = {
            'dataset': 'STL10', 'split': 'train', 'transform': transform
        }
        test_dict = {
            'dataset': 'STL10', 'split': 'test', 'transform': transform
        }
    elif args.data == 'svhn':
        input_size=(64, 64)
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Pad(16, 0, 'reflect'),
            torchvision.transforms.ToTensor()
        ])
        ae_train_dict = {
            'dataset': 'SVHN', 'split': 'extra', 'transform': transform
        }
        predictor_train_dict = {
            'dataset': 'SVHN', 'split': 'train', 'transform': transform
        }
        test_dict = {
            'dataset': 'SVHN', 'split': 'test', 'transform': transform
        }
    else:
        raise ValueError(
            f'Dataset {args.data} does not match any implemented dataset name'
        )

    # Get autoencoder networks, add code here to add new autoencoders
    networks = []
    for network in args.ae_networks:
        if network == 'FourLayerCVAE':
            networks.append(FourLayerCVAE)
        elif network == 'FeaturePredictorCVAE':
            networks.append(FeaturePredictorCVAE)
        elif network == 'FeatureAutoencoder':
            networks.append(FeatureAutoencoder)
        elif network == 'PerceptualFeatureToImgCVAE':
            networks.append(PerceptualFeatureToImgCVAE)
        elif network == 'FeatureToImgCVAE':
            networks.append(FeatureToImgCVAE)
        else:
            raise ValueError(
                f'{network} does not match any known autoencoder'
            )

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

    # Get loss networks
    loss_nets = []
    if args.use_experiment_setup:
        #If using the standard experiments, collect all appropriate loss nets
        loss_nets = [] 
        if all([ae == 'FourLayerCVAE' for ae in args.ae_networks]):
            loss_nets.append(None)
        for architecture, attributes in architecture_attributes.items():
            if not attributes['used_in_experiments']:
                continue
            for layer in attributes['layers_in_experiments']:
                loss_nets.append(
                    extractor_collector(
                        FeatureExtractor,
                        architecture=architecture,
                        layers=[layer],
                        normalize_in=False,
                        flatten_layer=False
                    )
                )
    else:
        #If not using the standard experiments, collect the defined loss nets
        for loss_net in args.loss_nets:
            if loss_net == 'None':
                loss_nets.append(None)
            elif loss_net in architecture_attributes:
                for layer in args.loss_net_layers:
                    loss_nets.append(
                        extractor_collector(
                            FeatureExtractor,
                            architecture=loss_net,
                            layers=[layer],
                            normalize_in=False,
                            flatten_layer=False
                        )
                    )
            else:
                raise ValueError(
                    f'{loss_net} does not match any known loss network\n'
                    'Select from: \n\t' +
                    '\n\t'.join(architecture_attributes.keys())
                )
    
    # For each parameter combination
    for network, z_dim, gamma, loss_net, beta_factor in product(
        networks, args.ae_zs, args.ae_gammas, loss_nets, args.beta_factors
    ):
        run_experiments(
            ae_dataset=ae_train_dict,
            train_dataset=predictor_train_dict,
            test_dataset=test_dict,
            input_size=input_size,
            validation_fraction=0.2,
            ae_epochs=args.ae_epochs,
            predictor_epochs=args.predictor_epochs,
            ae_batch_size=args.ae_batch_size,
            predictor_batch_size=args.predictor_batch_size,
            ae_net=network,
            loss_net=loss_net,
            z_dim=z_dim,
            gamma=gamma,
            beta_factor=beta_factor,
            repetitions=args.ae_repetitions,
            logger = logger,
            gpu=not args.no_gpu,
            num_workers=args.num_workers,
            clean_storage=args.clean_drive,
        )


# When this file is executed independently, execute the main function
if __name__ == "__main__":
    main()
