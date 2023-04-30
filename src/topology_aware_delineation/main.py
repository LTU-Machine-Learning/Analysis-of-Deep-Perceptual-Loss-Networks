import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

import numpy as np
import argparse

from train import *
from dataset_collector import dataset_collector, JointTransformDataset
from loss_networks import architecture_attributes

# Path to workspace
from workspace_path import home_path
checkpoint_dir = home_path / 'checkpoints/delineation'
checkpoint_dir.mkdir(parents=True, exist_ok=True)

log_dir = home_path / 'logs/delineation'
log_dir.mkdir(parents=True, exist_ok=True)

data_dir = home_path / 'datasets/MRD'
data_dir.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type = int, default = 100, help = 'number of epochs to train for')
    parser.add_argument('--loss_net', type = str, default = 'vgg19',  help = f'pretrained loss net {[arch for (arch, vals) in architecture_attributes.items() if vals["used_in_experiments"]]}')
    parser.add_argument('--layer', default = 3, type = int, help = 'the extracted features from feature extractor network')
    parser.add_argument('--K', type = int, default = 1, help = 'number of iterative steps')
    parser.add_argument('--batch_size', type = int, default = 8, help = 'batch size')
    parser.add_argument('--lr', type = float, default = 1e-3, help = 'learning rate')
    parser.add_argument('--prelud', type = float, default = 0.2, help = 'prelu coeff for down-sampling block')
    parser.add_argument('--preluu', type = float, default = 0., help = 'prelu coeff for up-sampling block')
    parser.add_argument('--dropout', type = float, default = 0., help = 'dropout coeff for all blocks')
    parser.add_argument('--bn', type = bool, default = True, help = 'batch-normalization coeff for all blocks')
    parser.add_argument('--color', type = bool, default = True, help = 'True if input is RGB, False otherwise')
    parser.add_argument('--mu', type = float, default = 0.01, help = 'loss coeff for vgg features')
    parser.add_argument('--model', type = str, default = 'None', help = 'path for pretrained model')
    parser.add_argument('--no_gpu', action='store_true', help = 'GPUs will not be used even if they are available')
    parser.add_argument('--no_download', action='store_true', help='Prevents automatic downloading of datasets')
    parser.add_argument('--num_workers', type=int, default=8, help='The number of workers to use for data management')

    option, unknown = parser.parse_known_args()

    # Use GPU if it is available unless given --no-gpu flag
    gpu = (not option.no_gpu) and torch.cuda.is_available()

    # Data Loader
    transform_trainset = JointTransform2D(crop = (224, 224))
    transform_testset = JointTransform2D(resize = 224, crop = False, p_flip = 0)

    trainset = dataset_collector(dataset='MRD', split='train', to_rgb=False, download = not option.no_download)
    valset = dataset_collector(dataset='MRD', split='val', to_rgb=False, download = not option.no_download)
    trainset = JointTransformDataset(trainset, transform_trainset)
    valset = JointTransformDataset(valset, transform_testset)
    training_data_loader = DataLoader(trainset, batch_size = option.batch_size, shuffle = True, num_workers = int(option.num_workers))
    val_data_loader = DataLoader(valset, batch_size = option.batch_size, shuffle = True, num_workers = int(option.num_workers))

    print('training_set:', len(trainset))
    print('val_set     :', len(valset))
    import time
    logname = log_dir / 'log.txt'
    
    # Set up model 
    inp_ch = 4 if option.color else 2

    unet = UNet_Model(inp_ch, batchNorm_down = option.bn, batchNorm_up = True, leakyReLU_factor_down = option.prelud, dropout_downward = option.dropout, leakyReLU_factor_up = option.preluu, dropout_upward = option.dropout)
    f_net = FNN_CNN_Net(option.loss_net, [option.layer])
    topology_aware_model = Topology_Aware_Delineation_Net(unet, f_net, option.K)

    if option.model != 'None':
        topology_aware_model.unet.load_state_dict(torch.load(option.model)["t_net"])

    # Criterion
    BCE_loss = nn.BCELoss()
    MSE_loss = nn.MSELoss()

    # Optimizer
    model_optimizer = optim.Adam(topology_aware_model.parameters(), lr = option.lr)
    best_loss = np.inf
    train_util = Training_Utility()
    print("Training TopologyNet.......")

    for epoch in range(option.epochs):
        train_util.epoch_in_train(topology_aware_model, model_optimizer, training_data_loader, MSE_loss, BCE_loss, option.mu, epoch, gpu)

        if epoch % 1 == 0:
            print("Begin validation...")
            current_loss = train_util.epoch_in_val(topology_aware_model, model_optimizer, val_data_loader, MSE_loss, BCE_loss, option.mu, epoch, gpu)
            if current_loss < best_loss:
                best_loss = current_loss
                nw_name = option.FNN
                nw_name = nw_name + "_" + str(option.layer)
                train_util.update_model(topology_aware_model, checkpoint_dir / f'model_{nw_name}.t7')
