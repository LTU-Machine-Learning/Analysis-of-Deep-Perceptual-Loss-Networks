# Library imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import csv
import argparse
import pathlib

# File imports
from loss_networks import FeatureExtractor, extractor_collector, \
     architecture_attributes, ExtractionModifier
from dataset_collector import dataset_collector

from workspace_path import home_path
log_dir = home_path / 'logs/perceptual_metric'
checkpoint_dir = home_path / 'checkpoints/perceptual_metric'


class DistanceMetric(nn.Module):
    '''
    A distance metric module for two images as measured by loss networks.
    Works as the metric described in The Unreasonable Effectiveness of Deep
    Features as a Perceptual Metric and outputs a feature vector of distances
    Args:
        network (nn.Module): Loss network to use for distance calculations
        weights (bool): Whether to use trainable weights to calibrate features
        scale (bool): Whether to scale input from [0,1] to [-1,1]
        upsample (bool): Whether to Upsample each layer to the same size
        channel_norm (bool): Whether to normalize features in channel dimension
        use_dropout (bool): Whether, if using weights, to add a dropout layer
    '''

    def __init__(
        self, network, weights=False, scale=False, upsample=False,
        channel_norm=True, use_dropout=True
    ):
        super().__init__()
        self.network = network
        self.scale = scale
        self.upsample = upsample
        self.channel_norm = channel_norm
        self.use_dropout = use_dropout and weights

        if weights:
            dummy = self.network(torch.rand((1,3,64,64)))
            weights = [
                nn.Conv2d(z.size(1), 1, 1, stride=1, padding=0, bias=False) for 
                z in dummy
            ]
            if self.use_dropout:
                weights = [nn.Sequential(nn.Dropout(), w) for w in weights]
            self.weights = nn.ModuleList(weights)
        else:
            self.weights = None

    def forward(self, x, x0):

        if self.scale:
            x = 2*x - 1
            x0 = 2*x0 -1
        
        ys = self.network(x.clone())
        y0s = self.network(x0.clone())
        zs = []
        
        for i, (y, y0) in enumerate(zip(ys, y0s)):

            if self.channel_norm:
                y = F.normalize(y, p=2, dim=1)
                y0 = F.normalize(y0, p=2, dim=1)
                
            z = y - y0
            z = z**2

            if self.weights is None:
                z = z.sum(dim=1, keepdim=True)
            else:
                z = self.weights[i](z)

            if self.upsample:
                z = nn.Upsample(
                    size=x.size()[2:], mode='bilinear', align_corners=False
                )(z)
            else:
                z = z.mean(dim=(2,3))

            zs.append(z)
        
        layers = torch.cat(zs, dim=1)
        layer_sum = layers.sum(dim=1)
        return layer_sum


def score_2afc_batch(d0, d1, label):
    '''
    Takes 2 distance metrics and a label to compute the 2afc score and accuracy
    Args:
        
    '''
    #point_five = torch.tensor([0.5])
    #one = torch.tensor([1.0])
    p0_closer = d0 < d1
    p0_score =  p0_closer*(1.0-label)
    p1_score = (d1 < d0)*label
    same_score = (d0 == d1)*0.5

    score = p0_score + p1_score + same_score

    closest = label<0.5
    correct = p0_closer.eq(closest)
    
    return score, correct.float()


def score_2afc(data, metric):
    '''
    Takes a DataLoader for a 2afc dataset and a similarity metric and
    calculates the 2afc score and accuracy of the metric
    Args:
        data (utils.data.DataLoader): Loader of dict with ref, p0, p1, judge
        metric (nn.Module): Takes two images and gives a simiarity distance
    Returns ({tensor, tensor}): The 2afc score and accuracy
    '''
    gpu = next(metric.parameters()).is_cuda
    accuracies = []
    scores = []
    for d in data:
        point_five = torch.tensor([0.5])
        one = torch.tensor([1.0])
        ref = d['ref']
        p0 = d['p0']
        p1 = d['p1']
        judge = d['judge']
        if gpu:
            one = one.cuda()
            point_five = point_five.cuda()
            ref = ref.cuda()
            p0 = p0.cuda()
            p1 = p1.cuda()
            judge = judge.cuda()

        loss0 = metric(ref, p0).view(-1)
        loss1 = metric(ref, p1).view(-1)
        label = judge.view(-1)

        p0_closer = loss0 < loss1
        p0_score =  p0_closer*(one-label)
        p1_score = (loss1 < loss0)*label
        same_score = (loss0 == loss1)*point_five

        score = p0_score + p1_score + same_score
        scores.append(score)

        closest = label<point_five
        correct = p0_closer.eq(closest)
        accuracies.append(correct.float())
    return {
        'score': torch.cat(scores).mean(),
        'accuracy': torch.cat(accuracies).mean()
    }


def score_jnd(data, metric):
    '''
    Takes a Dataloader for a JND dataset and a similarity metric and
    calculates the JND mAP score for that metric on that datset
    Args:
        data (utils.data.DataLoader): Loader of dict with p0, p1, and same
        metric (nn.Module): Takes two images and gives a simiarity distance
    Returns (tensor): The JND mAP score
    '''
    gpu = next(metric.parameters()).is_cuda

    one = torch.tensor([1.0])
    zero = torch.tensor([0.0])
    
    if gpu:
        one = one.cuda()
        zero = zero.cuda()

    sames = []
    dists = []
    for batch in data:
        p0 = batch['p0']
        p1 = batch['p1']
        same = batch['same']

        if gpu:
            p0 = p0.cuda()
            p1 = p1.cuda()    
            same = same.cuda()

        dist = metric(p0, p1).view(-1)

        #if gpu:
        #    dist = dist.cpu()

        sames.append(same)
        dists.append(dist)
    
    sames = torch.cat(sames)
    dists = torch.cat(dists)

    sorted_dists, indicies = torch.sort(dists)
    ordered_sames = sames[indicies]

    tps = ordered_sames.cumsum(dim=0)
    fps = (1-ordered_sames).cumsum(dim=0)
    fns = ordered_sames.sum(dim=0) - tps

    precision = torch.cat((zero, tps/(tps+fps), zero))
    recall = torch.cat((zero, tps/(tps+fns), one))

    for i in range(recall.size(0)-1, 0, -1):
        precision[i-1] = torch.max(precision[i-1], precision[i])
    
    i = torch.where(recall[1:] != recall[:-1])[0]

    return torch.sum((recall[i+1]-recall[i]) * precision[i+1])


class Learner(pl.LightningModule):

    '''
    Module to handle loss and score calculation, and potential LPIPS training
    of image distance metrics
    Args:
        metric (f(a,b)->float): Distance metric of a and b, can be trainable
        logit ('large'/'small'/'none'): What type of logit to use for training
        lpips_normalize (bool): Is the input normalized as in the original work
        sync_loss (int): Use sync loss. -1=until 0.5 or better, >0=# of epochs
        optimizers ([optim.Optimizer]): Optimizers to use, None to use defaults
        scherdulers ([optim.lr_scheduler]): Schedulers to use, None=defaults
    '''

    def __init__(
        self, metric, logit='large', lpips_normalize=False, sync_loss=0,
        optimizers=None, schedulers=None
    ):
        super().__init__()
        self.metric = metric
        self.logit = logit
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.sync_loss = sync_loss

        params = {
            'loss_network': str(metric.network),
            'logit': logit,
            'scale': metric.scale,
            'upsample': metric.upsample,
            'use_dropout': metric.use_dropout,
            'lpips_normalize': lpips_normalize,
            'sync_loss' : sync_loss,
            'optimizers': str(optimizers),
            'schedulers': str(schedulers),
        }
        self.save_hyperparameters(params)

        self.use_sync_loss = self.sync_loss
        self.loss_function = nn.BCELoss()
        if self.logit == 'large':
            self.logit_net = nn.Sequential(
                nn.Conv2d(5, 32, 1, stride=1, padding=0, bias=True),
                nn.LeakyReLU(0.2,True),
                nn.Conv2d(32, 32, 1, stride=1, padding=0, bias=True),
                nn.LeakyReLU(0.2,True),
                nn.Conv2d(32, 1, 1, stride=1, padding=0, bias=True),
                nn.Sigmoid(),
            )
        elif self.logit == 'small':
            self.logit_net = nn.Sequential(
                nn.Linear(2,1),
                nn.Sigmoid()
            )
        else:
            self.logit_net = nn.Sigmoid()

    def on_before_zero_grad(self, optimizer):
        if not self.metric.weights is None and self.metric.use_dropout:
            for module in self.metric.weights:
                module[1].weight.data=torch.clamp(module[1].weight.data, min=0)
        elif not self.metric.weights is None:
            for module in self.metric.weights:
                module.weight.data = torch.clamp(module.weight.data, min=0)
            
    def on_train_start(self):
        self.use_sync_loss = self.sync_loss

    def on_train_epoch_end(self):
        if self.use_sync_loss > 0:
            self.use_sync_loss = self.use_sync_loss-1

    def loss_calculation(self, batch, batch_idx, prefix='test_'):
        p0 =  batch['p0']
        p1 = batch['p1']
        ref = batch['ref']
        judge = batch['judge']

        d0 = self.metric(ref, p0).view(-1)
        d1 = self.metric(ref, p1).view(-1)
        label = judge.view(-1)

        if self.logit == 'large':
            eps = 0.1
            prediction = self.logit_net(torch.stack(
                (d0,d1,d0-d1,d0/(d1+eps),d1/(d0+eps)), dim=1
            ).view(-1,5,1,1)).view(-1) 
        elif self.logit == 'small':
            prediction = self.logit_net(torch.stack((d0,d1),dim=1)).view(-1)
        else:
            prediction = self.logit_net(d0-d1)
        loss = self.loss_function(prediction, label)
        score, accuracy = score_2afc_batch(d0, d1, label)
        score, accuracy = score.mean(), accuracy.mean()
        
        if self.use_sync_loss > 0 or (self.use_sync_loss < 0 and score <= 0.5):
            loss = loss + 10*nn.ReLU()(
                self.loss_function(nn.Sigmoid()(d0-d1), label)
            )

        self.log(f'{prefix}loss', loss, on_epoch=True, logger=True)
        self.log(f'{prefix}score', score, on_epoch=True, logger=True)
        self.log(f'{prefix}accuracy', accuracy, on_epoch=True, logger=True)
        return loss

    def forward(self, x, x0):
        return self.metric(x, x0)
    
    def training_step(self, batch, batch_idx):
        return self.loss_calculation(batch, batch_idx, 'train_')

    def validation_step(self, batch, batch_idx):
        return self.loss_calculation(batch, batch_idx, 'validation_')

    def test_step(self, batch, batch_idx, dataloader_id=None):
        return self.loss_calculation(batch, batch_idx, 'test_')

    def configure_optimizers(self):
        if self.optimizers is None:
            optimizers = torch.optim.Adam(
                self.parameters(), lr=0.001, betas=(0.5, 0.999)
            )
        if self.schedulers is None and self.optimizers is None:
            schedulers = torch.optim.lr_scheduler.SequentialLR(
                optimizers,
                schedulers = [
                    torch.optim.lr_scheduler.ConstantLR(
                        optimizers, factor = 1.0, total_iters=5
                    ),
                    torch.optim.lr_scheduler.LinearLR(
                        optimizers, start_factor=1.0, end_factor=0.0
                    )
                ],
                milestones=[5]
            )
            
            schedulers.optimizer = optimizers #Needed due to a PyTorch 1.10 bug
        return ([optimizers],[schedulers])


def run_experiment(
    loss_network_path, variant, repetitions=1, lpips_normalize=False,
    channel_norm=True, batch_size=256, metric_fun='spatial', logger=CSVLogger,
    num_workers=1
):
    '''
    Given a loss network and whether or not to add calibration weights,
    creates a metric with that loss network, trains it (if applicable) on the
    BAPPS 2afc train set, and tests it on each of the BAPPS validation splits.
    Args:
        loss_network_path (str): Path to the loss network to use as a metric
        variant (str): Which variant to use (baseline, lin, scratch, tune)
        repetitions (int): How many times to repeat training
        lpips_normalize (bool): Whether to use torchvision or lpips normalize
        channel_norm (bool): Whether to normalize features in channel dimension
        batch_size (int): Size of training and validation batches
        metric_fun (str): How to compare features "spatial[+]", "sort", "mean"
        logger (f()->Logger): PyTorch Lightning Logger
        num_workers (int): Number of worker processes used to load data
    '''
    
    weights = variant != 'baseline'

    # Create an experiment id
    experiment_name = pathlib.Path(loss_network_path).stem + f'_{variant}'
    if lpips_normalize:
        experiment_name = experiment_name + '_lpips-norm'
    if metric_fun != 'spatial':
        experiment_name = experiment_name + f'_{metric_fun}'
    if not channel_norm:
        experiment_name = experiment_name + '_no-channel-norm'

    # Whether to run on the GPU
    gpus = None
    if torch.cuda.is_available():
        gpus = 1

    # Create the image preprocessing
    if lpips_normalize:
        norm = transforms.Normalize((-0.030,-0.088,-0.188),(0.458,0.448,0.450))
    else:
        norm = transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))

    image_transform = transforms.Compose([
        transforms.Resize(size=(64,64)),
        transforms.ToTensor(),
        norm
    ])

    # The metric(s) to test
    learners = []
    trainers = []
        
    # Train once per repetition
    for version in range(repetitions):
        
        # Create a path to this version
        filepath = log_dir / f'{experiment_name}/version_{version}'

        # Create the metric to use
        #TODO: Make factors possible
        loss_network = torch.load(loss_network_path)
        if metric_fun != 'spatial':
            loss_network = ExtractionModifier(
                extractor=loss_network,
                modification=metric_fun.replace('spatial+',''),
                return_policy='concat' if ('+' in metric_fun) else 'new',
                shape_policy='keep'
            )
        metric = DistanceMetric(
            loss_network, weights=weights, channel_norm=channel_norm
        )
        
        # Create the Learner to handle training
        learner = Learner(
            metric, logit='large', lpips_normalize=lpips_normalize
        )
        learners.append(learner)

        
        checkpoint_callback = ModelCheckpoint(
            dirpath = filepath
        )
        log = logger(
            save_dir=str(log_dir),
            name=f'{experiment_name}_version{version}'
        )

        # Setting up the Pytorch-Lightning trainer
        trainer = pl.Trainer(
            logger=log,
            callbacks=[checkpoint_callback],
            max_epochs=10,
            gpus=gpus,
        )
        trainers.append(trainer)
        
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

            # Collect the training set
            train_data = dataset_collector(
                'BAPPS', 'train',
                image_transform=image_transform, subsplit='all'
            )
            train_loader = DataLoader(
                train_data, batch_size=batch_size, shuffle=True,
                num_workers=num_workers
            )

            # Training using the trainer
            trainer.fit(
                model = learner,
                train_dataloaders = train_loader,
                ckpt_path = ckpt_path
            )
        else:
            # No need for multiple versions of deterministic experiment
            break
    
    # Create Dataloaders of 2afc splits and jnd for testing
    val_loaders_2afc = []
    subsplits_2afc = [
        'cnn', 'color', 'deblur', 'frameinterp', 'superres', 'traditional'
    ]
    for subsplit_2afc in subsplits_2afc:
        val_data = dataset_collector(
            'BAPPS', 'val', image_transform=image_transform,
            subsplit=subsplit_2afc
        )
        val_loader = DataLoader(
            val_data, batch_size=batch_size, shuffle=False,
            num_workers=num_workers
        )
        val_loaders_2afc.append(val_loader)
    jnd_data = dataset_collector(
        'BAPPS', 'jnd/val', image_transform=image_transform
    )
    jnd_loader = DataLoader(
        jnd_data, batch_size=batch_size, shuffle=False,
        num_workers=num_workers
    )

    for version, (trainer, learner) in enumerate(zip(trainers, learners)):
        filepath = log_dir / f'{experiment_name}/version_{version}'
        
        # Check if this experiment has already been done
        if (filepath / 'test_results.csv').exists():
            print(f'{experiment_name}/version_{version} is already tested')
            continue

        if gpus is not None:
            learner.cuda()
        jnd_result = score_jnd(jnd_loader, learner).item()
        results = trainer.test(
            model = learner,
            dataloaders = val_loaders_2afc
        )

        if not filepath.exists():
            filepath.mkdir(parents=True)
        with open(filepath / 'test_results.csv', 'w') as save_file:
            writer = csv.writer(save_file, delimiter=' ')
            for result in results:
                for row in result.items():
                    new_row = [
                        row[0][:-1] + subsplits_2afc[int(row[0][-1])], row[1]
                    ]
                    writer.writerow(new_row)
            writer.writerow(['test_jnd/jnd', jnd_result])

        #Force logger to end experiment to stop WandB from using the same entry    
        if logger == WandbLogger:
            trainer.logger.experiment.finish()


def main():
    '''
    Given parameters, trains (if applicable) and tests loss networks as
    perceptual metrics on the applicable datasets
    '''
    # Create parser and parse input
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--loss_nets',
        type=str,
        default=['alexnet'],
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
        default=['baseline'],
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
        '--lpips_normalize',
        action='store_true',
        help='Use image normalization parameters from original paper'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=256,
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
        help='Select which function to calculate perceptual similarity with'
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
                run_experiment(
                    loss_network_path,
                    variant,
                    args.repetitions,
                    args.lpips_normalize,
                    batch_size=args.batch_size,
                    metric_fun=fun,
                    num_workers=args.num_workers,
                    channel_norm=not args.no_channel_norm,
                    logger=logger
                )


# When this file is executed independently, execute the main function
if __name__ == "__main__":
    main()
