import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torch.utils.data import TensorDataset
from argparse import ArgumentParser

'''
This file contains implementations of the encoders and autoencoders from
https://github.com/guspih/Perceptual-Autoencoders
'''


class TemplateVAE(pl.LightningModule):
    '''
    A template class for Variational Autoencoders to minimize code duplication
    hparams:
        input_size (int,int): The height and width of the input image
        z_dimensions (int): The number of latent dimensions in the encoding
        variational (bool): Whether the model is variational or not
        gamma (float): The weight of the KLD loss
        perceptual_net_path (str/None): Path to the loss network to use
        beta_factor (float/None): Factor for optional loss term
    '''

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--input_size', type=tuple, default=(64, 64))
        parser.add_argument('--z_dimensions', type=int, default=32)
        parser.add_argument('--variational', type=bool, default=True)
        parser.add_argument('--gamma', type=float, default=20.0)
        parser.add_argument('--perceptual_net_path', type=str, default=None)
        parser.add_argument('--beta_factor', type=float, default=None)
        return parser

    def __init__(
        self, input_size=(64, 64), z_dimensions=32, variational=True,
        gamma=20.0, perceptual_net_path=None, beta_factor=None
    ):
        super().__init__()
        self.save_hyperparameters()
        self.perceptual_loss = not self.hparams.perceptual_net_path is None

        # Parameter check
        if (
            (self.hparams.input_size[0] - 64) % 16 != 0 or
            (self.hparams.input_size[1] - 64) % 16 != 0
        ):
            raise ValueError(
                f'Input_size is {self.input_size}, but must be 64+16*N'
            )

        # Collecting perceptual net
        self.perceptual_net = None
        if self.perceptual_loss:
            self.perceptual_net = torch.load(self.hparams.perceptual_net_path)

            # Checking that the perceptual_net can handle beta_factor
            if (
                not self.hparams.beta_factor is None and 
                self.perceptual_net.flatten_layer
            ):
                raise ValueError(
                    f'beta_factor incompatible with flatten_layer in extractor'
                )


    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar

    def sample(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.autograd.Variable(std.data.new(std.size()).normal_())
        out = eps.mul(std).add_(mu)
        return out

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        if self.hparams.variational:
            z = self.sample(mu, logvar)
        else:
            z = mu
        rec_x = self.decode(z)
        return rec_x, z, mu, logvar

    def reconstruction_loss(self, x, rec):
        if self.perceptual_loss:
            xs = self.perceptual_net(x)
            recs = self.perceptual_net(rec)
            xs_flat = torch.cat([x.view(x.size(0), -1) for x in xs])
            recs_flat = torch.cat([rec.view(rec.size(0), -1) for rec in recs])
            ret = F.mse_loss(recs_flat, xs_flat, reduction='mean')
            if not self.hparams.beta_factor is None:
                ret += self.hparams.beta_factor*self.beta_factor_loss(xs, recs)
            return ret
        else:
            x = x.view(x.size(0), -1)
            rec = rec.view(x.size(0), -1)
            return F.mse_loss(rec, x, reduction='mean')

    def beta_factor_loss(self, xs, recs):
        xs = torch.cat(
            [x.view(x.size(0), x.size(1), -1).sort()[0] for x in xs]
        )
        recs = torch.cat(
            [rec.view(rec.size(0), rec.size(1), -1).sort()[0] for rec in recs]
        )

        return max(torch.log(F.mse_loss(xs, recs, reduction='mean')),0.0)

    def loss_calculation(self, batch, batch_idx, prefix='train_'):
        x, y = batch
        rec_x, z, mu, logvar = self(x)

        rec = self.reconstruction_loss(x, rec_x)
        loss = rec
        if self.hparams.variational:
            kld = -1 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = rec + self.hparams.gamma*kld

        self.log(f'{prefix}loss', loss)
        if self.hparams.variational:
            self.log(f'{prefix}rec', rec)
            self.log(f'{prefix}kld', kld)

        return loss

    def training_step(self, batch, batch_idx):
        return self.loss_calculation(batch, batch_idx, prefix='train_')

    def validation_step(self, batch, batch_idx):
        return self.loss_calculation(batch, batch_idx, prefix='val_')

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())


def _create_coder(channels, kernel_sizes, strides, conv_types,
                  activation_types, paddings=(0, 0), batch_norms=False
                  ):
    '''
    Function that creates en- or decoders based on parameters
    Args:
        channels ([int]): Channel sizes per layer. 1 more than layers
        kernel_sizes ([int]): Kernel sizes per layer
        strides ([int]): Strides per layer
        conv_types ([f()->type]): Type of the convoultion module per layer
        activation_types ([f()->type]): Type of activation function per layer
        paddings ([(int, int)]): The padding per layer
        batch_norms ([bool]): Whether to use batchnorm on each layer
    Returns (nn.Sequential): The created coder
    '''
    if not isinstance(conv_types, list):
        conv_types = [conv_types for _ in range(len(kernel_sizes))]

    if not isinstance(activation_types, list):
        activation_types = [activation_types for _ in range(len(kernel_sizes))]

    if not isinstance(paddings, list):
        paddings = [paddings for _ in range(len(kernel_sizes))]

    if not isinstance(batch_norms, list):
        batch_norms = [batch_norms for _ in range(len(kernel_sizes))]

    coder = nn.Sequential()
    for layer in range(len(channels)-1):
        coder.add_module(
            'conv' + str(layer),
            conv_types[layer](
                in_channels=channels[layer],
                out_channels=channels[layer+1],
                kernel_size=kernel_sizes[layer],
                stride=strides[layer]
            )
        )
        if batch_norms[layer]:
            coder.add_module(
                'norm'+str(layer),
                nn.BatchNorm2d(channels[layer+1])
            )
        if not activation_types[layer] is None:
            coder.add_module('acti'+str(layer), activation_types[layer]())

    return coder


def _create_image_encoder(input_size, z_dimensions):
    encoder_channels = [3, 32, 64, 128, 256]
    encoder = _create_coder(
        encoder_channels, [4, 4, 4, 4], [2, 2, 2, 2],
        nn.Conv2d, nn.ReLU,
        batch_norms=[True, True, True, True]
    )

    def f(x): return int((x - 2)/2)
    conv_size_0 = f(f(f(f(input_size[0]))))
    conv_size_1 = f(f(f(f(input_size[1]))))
    conv_flat_size = int(encoder_channels[-1]*conv_size_0*conv_size_1)
    mu = nn.Linear(conv_flat_size, z_dimensions)
    logvar = nn.Linear(conv_flat_size, z_dimensions)

    return encoder, mu, logvar


def _create_image_decoder(input_size, z_dimensions):

    def g(x): return int((x-64)/16)+1
    deconv_flat_size = g(input_size[0]) * g(input_size[1]) * 1024
    dense = nn.Linear(z_dimensions, deconv_flat_size)

    decoder_channels = [1024, 128, 64, 32, 3]
    decoder = _create_coder(
        decoder_channels, [5, 5, 6, 6], [2, 2, 2, 2],
        nn.ConvTranspose2d,
        [nn.ReLU, nn.ReLU, nn.ReLU, nn.Sigmoid],
        batch_norms=[True, True, True, False]
    )

    return dense, decoder


def _create_feature_encoder(input_size, z_dimensions, perceptual_net):
    inp = torch.rand((1, 3, input_size[0], input_size[1]))
    outs = perceptual_net(
        inp.to(next(perceptual_net.parameters()).device)
    )
    perceptual_size = torch.cat([out.view(-1) for out in outs]).numel()

    hidden_layer_size = int(min(perceptual_size/2, 2048))
    encoder = nn.Sequential(
        nn.Linear(perceptual_size, hidden_layer_size),
        nn.ReLU(),
    )
    mu = nn.Linear(hidden_layer_size, z_dimensions)
    logvar = nn.Linear(hidden_layer_size, z_dimensions)
    return encoder, mu, logvar


def _create_feature_decoder(input_size, z_dimensions, perceptual_net):
    inp = torch.rand((1, 3, input_size[0], input_size[1]))
    outs = perceptual_net(
        inp.to(next(perceptual_net.parameters()).device)
    )
    perceptual_size = torch.cat([out.view(-1) for out in outs]).numel()

    hidden_layer_size = int(min(perceptual_size/2, 2048))
    decoder = nn.Sequential(
        nn.Linear(z_dimensions, hidden_layer_size),
        nn.ReLU(),
        nn.Linear(hidden_layer_size, perceptual_size)
    )
    return decoder


class FourLayerCVAE(TemplateVAE):
    '''
    A Convolutional Variational Autoencoder for images
    hparams:
        input_size (int,int): The height and width of the input image
            acceptable sizes are 64+16*n
        z_dimensions (int): The number of latent dimensions in the encoding
        variational (bool): Whether the model is variational or not
        gamma (float): The weight of the KLD loss
        perceptual_net: Which perceptual network to use (None for pixel-wise)
    '''

    def __init__(self, **hparams):
        super().__init__(**hparams)

        # Image encoding parts
        self.encoder, self.mu, self.logvar = _create_image_encoder(
            self.hparams.input_size, self.hparams.z_dimensions
        )

        # Image decoding parts
        self.dense, self.decoder = _create_image_decoder(
            self.hparams.input_size, self.hparams.z_dimensions
        )

        self.relu = nn.ReLU()

    def decode(self, z):
        y = self.dense(z)
        y = self.relu(y)
        y = y.view(
            y.size(0), 1024,
            int((self.hparams.input_size[0]-64)/16)+1,
            int((self.hparams.input_size[1]-64)/16)+1
        )
        y = self.decoder(y)
        return y


class FeaturePredictorCVAE(TemplateVAE):
    '''
    A Convolutional Variational autoencoder trained with feature prediction
    I-F-FP procedure in the paper
    Args:
        input_size (int,int): The height and width of the input image
            acceptable sizes are 64+16*n
        z_dimensions (int): The number of latent dimensions in the encoding
        variational (bool): Whether the model is variational or not
        gamma (float): The weight of the KLD loss
        perceptual_net: Which perceptual network to use
    '''

    def __init__(self, **parameters):
        super().__init__(**parameters)

        # Ensure that perceptual_loss is not None
        assert self.perceptual_net != None, \
            'For FeaturePredictorCVAE, perceptual_net cannot be None'

        # Image encoding parts
        self.encoder, self.mu, self.logvar = _create_image_encoder(
            self.hparams.input_size, self.hparams.z_dimensions
        )

        # Feature decoding parts
        self.decoder = _create_feature_decoder(
            self.hparams.input_size,
            self.hparams.z_dimensions,
            self.perceptual_net
        )

    def reconstruction_loss(self, x, rec):
        ys = self.perceptual_net(x)
        ys = torch.cat([y.view(y.size(0),-1) for y in ys])
        return F.mse_loss(rec, ys, reduction='mean')


class FeatureAutoencoder(TemplateVAE):
    '''
    An fc autoencoder that autoencodes the features of a perceptual network
    F-F-FP procedure in the paper
    Args:
        input_size (int,int): The height and width of the input image
            acceptable sizes are 64+16*n
        z_dimensions (int): The number of latent dimensions in the encoding
        variational (bool): Whether the model is variational or not
        gamma (float): The weight of the KLD loss
        perceptual_net: Which perceptual network to use
    '''

    def __init__(self, **parameters):
        super().__init__(**parameters)

        # Ensure that perceptual_loss is not None
        assert self.perceptual_net != None, \
            'For FeatureAutoencoder, perceptual_net cannot be None'

        # Feature encoding parts
        self.encoder, self.mu, self.logvar = _create_feature_encoder(
            self.hparams.input_size,
            self.hparams.z_dimensions,
            self.perceptual_net
        )

        # Feature decoding parts
        self.decoder = _create_feature_decoder(
            self.hparams.input_size,
            self.hparams.z_dimensions,
            self.perceptual_net
        )

    def encode(self, x):
        ys = self.perceptual_net(x)
        ys = torch.cat([y.view(y.size(0), -1) for y in ys])
        ys = self.encoder(ys)
        mu = self.mu(ys)
        logvar = self.logvar(ys)
        return mu, logvar

    def reconstruction_loss(self, x, rec):
        ys = self.perceptual_net(x)
        ys = torch.cat([y.view(y.size(0), -1) for y in ys])
        return F.mse_loss(rec, ys, reduction='mean')


class PerceptualFeatureToImgCVAE(TemplateVAE):
    '''
    A CVAE that encodes perceptual features and reconstructs the images
    Trained with perceptual loss
    F-I-PS in the paper
    Args:
        input_size (int,int): The height and width of the input image
            acceptable sizes are 64+16*n
        z_dimensions (int): The number of latent dimensions in the encoding
        variational (bool): Whether the model is variational or not
        gamma (float): The weight of the KLD loss
        perceptual_net: Which feature extraction and perceptual net to use
    '''

    def __init__(self, **parameters):
        super().__init__(**parameters)

        # Ensure that perceptual_loss is not None
        assert self.perceptual_net != None, \
            'For PerceptualFeatureToImgCVAE, perceptual_net cannot be None'

        # Feature encoding parts
        self.encoder, self.mu, self.logvar = _create_feature_encoder(
            self.hparams.input_size,
            self.hparams.z_dimensions,
            self.perceptual_net
        )

        # Image decoding parts
        self.dense, self.decoder = _create_image_decoder(
            self.hparams.input_size, self.hparams.z_dimensions
        )

        self.relu = nn.ReLU()

    def encode(self, x):
        ys = self.perceptual_net(x)
        ys = torch.cat([y.view(y.size(0), -1) for y in ys])
        ys = self.encoder(ys)
        mu = self.mu(ys)
        logvar = self.logvar(ys)
        return mu, logvar

    def decode(self, z):
        y = self.dense(z)
        y = self.relu(y)
        y = y.view(
            y.size(0), 1024,
            int((self.hparams.input_size[0]-64)/16)+1,
            int((self.hparams.input_size[1]-64)/16)+1
        )
        y = self.decoder(y)
        return y


class FeatureToImgCVAE(PerceptualFeatureToImgCVAE):
    '''
    A CVAE that encodes perceptual features and reconstructs the images
    Trained with pixel-wise loss
    F-I-PW in the paper
    Args:
        input_size (int,int): The height and width of the input image
            acceptable sizes are 64+16*n
        z_dimensions (int): The number of latent dimensions in the encoding
        variational (bool): Whether the model is variational or not
        gamma (float): The weight of the KLD loss
        perceptual_net: Which feature extraction net to use
    '''

    def reconstruction_loss(self, x, rec):
        x = x.reshape(x.size(0), -1)
        rec = rec.view(x.size(0), -1)
        return F.mse_loss(rec, x, reduction='mean')


def encode_data(autoencoder, dataloader, batch_size=512, to_onehot=False):
    '''
    Takes a autoencoder and a DataLoader and encodes that data
    Args:
        autoencoder (nn.Module): Autoencoder to use for encoding the data
        dataloader (utils.data.DataLoader): DataLoader with data to encode
        batch_size (int): Batches for encoding
        to_onehot (None/int): Size if the output should be turned into onehot
    Returns (utils.data.Dataset): The encoded dataset
    '''
    gpu = next(autoencoder.parameters()).is_cuda
    coded_xs = []
    coded_ys = []
    autoencoder.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            if gpu:
                x = x.cuda()
            coded_x = autoencoder.encode(x)[0]
            coded_y = y
            if not to_onehot is None:
                coded_y = torch.eye(to_onehot)
                coded_y = coded_y[y]
            if gpu:
                x = x.cpu()
                coded_x = coded_x.cpu()
            coded_xs.append(coded_x)
            coded_ys.append(coded_y)
    autoencoder.train()

    coded_xs = torch.cat(coded_xs, dim=0)
    coded_ys = torch.cat(coded_ys, dim=0)
    return TensorDataset(coded_xs, coded_ys)
