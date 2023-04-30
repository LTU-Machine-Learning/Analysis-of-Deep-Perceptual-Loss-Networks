import torch
import torch.nn as nn
import pytorch_lightning as pl

'''
This file contains implementations of the predictor networks from
https://github.com/guspih/Perceptual-Autoencoders
'''


class FullyConnected(pl.LightningModule):
    '''
    A simple fully connected network
    Args:
        input_size (int): Input size to the network
        layers ([int]): Layer sizes
        activation_funcs ([f()->nn.Module]): Class of activation functions
        loss_funcs (f()->dict): Function to get loss and other things to log
    '''

    def __init__(self, input_size, layers, activation_funcs, loss_funcs):
        super().__init__()
        self.loss_funcs = loss_funcs
        if not isinstance(activation_funcs, list):
            activation_funcs = [
                activation_funcs for _ in range(len(layers)+1)
            ]

        self.network = nn.Sequential()
        layers = [input_size] + layers
        for layer_id in range(len(layers)-1):
            self.network.add_module(
                'linear{}'.format(layer_id),
                nn.Linear(layers[layer_id], layers[layer_id+1])
            )
            if not activation_funcs[layer_id] is None:
                self.network.add_module(
                    'activation{}'.format(layer_id),
                    activation_funcs[layer_id]()
                )

    def forward(self, x):
        return self.network(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def any_step(self, batch, batch_idx, prefix):
        x, y = batch
        y_hat = self(x)
        losses = self.loss_funcs(y_hat, y)
        
        for name, value in losses.items():
            self.log(f'{prefix}{name}', value, on_epoch=True, logger=True)

        return losses['loss']

    def training_step(self, batch, batch_idx):
        return self.any_step(batch, batch_idx, prefix='train_')

    def validation_step(self, batch, batch_idx):
        return self.any_step(batch, batch_idx, prefix='val_')

    def test_step(self, batch, batch_idx):
        return self.any_step(batch, batch_idx, prefix='test_')
