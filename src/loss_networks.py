from math import prod
import os
import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# Folder for extractor checkpoints
from workspace_path import home_path
checkpoint_folder = home_path / 'checkpoints/extractors'

# Set the directory for torchvision models
torch.hub.set_dir(home_path/'checkpoints/hub')

# Dictionary of torchvision models and the attribute 'paths' to their features
architecture_attributes = {
    'alexnet': {'features': ['features'], 'max_layer_nr': 13, 'used_in_experiments': True, 'layers_in_experiments': [1,4,7,11]},
    'vgg11': {'features': ['features'], 'max_layer_nr': 21, 'used_in_experiments': True, 'layers_in_experiments': [1,4,9,19]},
    'vgg11_bn': {'features': ['features'], 'max_layer_nr': 29, 'used_in_experiments': False},
    'vgg13': {'features': ['features'], 'max_layer_nr': 25, 'used_in_experiments': False},
    'vgg13_bn': {'features': ['features'], 'max_layer_nr': 35, 'used_in_experiments': False},
    'vgg16': {'features': ['features'], 'max_layer_nr': 31, 'used_in_experiments': True, 'layers_in_experiments': [3,8,15,29]},
    'vgg16_bn': {'features': ['features'], 'max_layer_nr': 44, 'used_in_experiments': True, 'layers_in_experiments': [5,12,22,42]},
    'vgg19': {'features': ['features'], 'max_layer_nr': 37, 'used_in_experiments': True, 'layers_in_experiments': [3,8,17,35]},
    'vgg19_bn': {'features': ['features'], 'max_layer_nr': 53, 'used_in_experiments': False},
    'densenet121': {'features': ['features'], 'max_layer_nr': 12, 'used_in_experiments': True, 'layers_in_experiments': [2,4,6,10]},
    'densenet161': {'features': ['features'], 'max_layer_nr': 12, 'used_in_experiments': False},
    'densenet169': {'features': ['features'], 'max_layer_nr': 12, 'used_in_experiments': False},
    'densenet201': {'features': ['features'], 'max_layer_nr': 12, 'used_in_experiments': False},
    'resnet18': {'features': [None], 'max_layer_nr': 9, 'used_in_experiments': True, 'layers_in_experiments': [2,4,5,7]},
    'resnet34': {'features': [None], 'max_layer_nr': 9, 'used_in_experiments': False},
    'resnet50': {'features': [None], 'max_layer_nr': 9, 'used_in_experiments': True, 'layers_in_experiments': [2,4,5,7]},
    'resnet101': {'features': [None], 'max_layer_nr': 9, 'used_in_experiments': False},
    'resnet152': {'features': [None], 'max_layer_nr': 9, 'used_in_experiments': False},
    'wide_resnet50_2': {'features': [None], 'max_layer_nr': 9, 'used_in_experiments': False},
    'wide_resnet101_2': {'features': [None], 'max_layer_nr': 9, 'used_in_experiments': False},
    'shufflenet_v2_x1_0': {'features': [None], 'max_layer_nr': 6, 'used_in_experiments': False},
    'shufflenet_v2_x2_0': {'features': [None], 'max_layer_nr': 6, 'used_in_experiments': False},
    'mobilenet_v2': {'features': ['features'], 'max_layer_nr': 19, 'used_in_experiments': False},
    'googlenet': {'features': [None], 'max_layer_nr': 16, 'used_in_experiments': True, 'layers_in_experiments': [0,5,8,15]},
    'inception_v3': {'features': [None], 'max_layer_nr': 15, 'used_in_experiments': True, 'layers_in_experiments': [2,7,9,14]},
    'squeezenet1_0': {'features': ['features'], 'max_layer_nr': 13, 'used_in_experiments': False},
    'squeezenet1_1': {'features': ['features'], 'max_layer_nr': 13, 'used_in_experiments': True, 'layers_in_experiments': [1,3,7,12]},
    'mobilenet_v3_large': {'features': ['features'], 'max_layer_nr': 17, 'used_in_experiments': False},
    'mobilenet_v3_small': {'features': ['features'], 'max_layer_nr': 13, 'used_in_experiments': False},
    'resnext50_32x4d': {'features': [None], 'max_layer_nr': 9, 'used_in_experiments': True, 'layers_in_experiments': [2,4,5,7]},
    'resnext101_32x8d': {'features': [None], 'max_layer_nr': 9, 'used_in_experiments': False},
    'mnasnet0_5': {'features': ['layers'], 'max_layer_nr': 17, 'used_in_experiments': False},
    'mnasnet0_75': {'features': ['layers'], 'max_layer_nr': 17, 'used_in_experiments': False},
    'mnasnet1_0': {'features': ['layers'], 'max_layer_nr': 17, 'used_in_experiments': False},
    'mnasnet1_3': {'features': ['layers'], 'max_layer_nr': 17, 'used_in_experiments': False},
    'efficientnet_b0': {'features': ['features'], 'max_layer_nr': 9, 'used_in_experiments': True, 'layers_in_experiments': [0,1,4,7]},
    'efficientnet_b1': {'features': ['features'], 'max_layer_nr': 9, 'used_in_experiments': False},
    'efficientnet_b2': {'features': ['features'], 'max_layer_nr': 9, 'used_in_experiments': False},
    'efficientnet_b3': {'features': ['features'], 'max_layer_nr': 9, 'used_in_experiments': False},
    'efficientnet_b4': {'features': ['features'], 'max_layer_nr': 9, 'used_in_experiments': False},
    'efficientnet_b5': {'features': ['features'], 'max_layer_nr': 9, 'used_in_experiments': False},
    'efficientnet_b6': {'features': ['features'], 'max_layer_nr': 9, 'used_in_experiments': False},
    'efficientnet_b7': {'features': ['features'], 'max_layer_nr': 9, 'used_in_experiments': True, 'layers_in_experiments': [0,1,4,7]},
    'regnet_y_400mf': {'features': ['stem', 'trunk_output'], 'max_layer_nr': 7, 'used_in_experiments': False},
    'regnet_y_800mf': {'features': ['stem', 'trunk_output'], 'max_layer_nr': 7, 'used_in_experiments': False},
    'regnet_y_1_6gf': {'features': ['stem', 'trunk_output'], 'max_layer_nr': 7, 'used_in_experiments': False},
    'regnet_y_3_2gf': {'features': ['stem', 'trunk_output'], 'max_layer_nr': 7, 'used_in_experiments': False},
    'regnet_y_8gf': {'features': ['stem', 'trunk_output'], 'max_layer_nr': 7, 'used_in_experiments': False},
    'regnet_y_16gf': {'features': ['stem', 'trunk_output'], 'max_layer_nr': 7, 'used_in_experiments': False},
    'regnet_y_32gf': {'features': ['stem', 'trunk_output'], 'max_layer_nr': 7, 'used_in_experiments': False},
    'regnet_x_400mf': {'features': ['stem', 'trunk_output'], 'max_layer_nr': 7, 'used_in_experiments': False},
    'regnet_x_800mf': {'features': ['stem', 'trunk_output'], 'max_layer_nr': 7, 'used_in_experiments': False},
    'regnet_x_1_6gf': {'features': ['stem', 'trunk_output'], 'max_layer_nr': 7, 'used_in_experiments': False},
    'regnet_x_3_2gf': {'features': ['stem', 'trunk_output'], 'max_layer_nr': 7, 'used_in_experiments': False},
    'regnet_x_8gf': {'features': ['stem', 'trunk_output'], 'max_layer_nr': 7, 'used_in_experiments': False},
    'regnet_x_16gf': {'features': ['stem', 'trunk_output'], 'max_layer_nr': 7, 'used_in_experiments': False},
    'regnet_x_32gf': {'features': ['stem', 'trunk_output'], 'max_layer_nr': 7, 'used_in_experiments': False},
}


class FeatureExtractor(nn.Module):
    '''
    A feature extractor for torchvision models
    Args:
        architecture (str): The architecture to extract from
        layers ([int]): The sub-modules in 'features' to extract at (0-index)
        pretrained (bool): Whether to load a pretrained model or not
        frozen (bool): Whether the network can be trained
        normalize_in (bool): Whether to normalize as torchvision suggests
        flatten_layer (bool): Whether to flatten the extractions per layer
    '''

    def __init__(
        self, architecture, layers, pretrained=True, frozen=True,
        normalize_in=True, flatten_layer=True
    ):
        super().__init__()
        self.architecture = architecture
        self.layers = layers
        self.final_layer = max(layers)+1
        self.pretrained = pretrained
        self.frozen = frozen
        self.normalize_in = normalize_in
        self.flatten_layer = flatten_layer

        if self.normalize_in:
            self.mean = torch.nn.Parameter(
                torch.as_tensor([0.485, 0.456, 0.406])[None, :, None, None],
                requires_grad=False
            )
            self.std = torch.nn.Parameter(
                torch.as_tensor([0.229, 0.224, 0.225])[None, :, None, None],
                requires_grad=False
            )

        original_model = models.__dict__[architecture](
            pretrained=self.pretrained
        )
        features=[]
        for attr in architecture_attributes[architecture]['features']:
            if attr is None:
                original_features = original_model
            else:
                original_features = getattr(original_model, attr)
            
            features.extend(list(original_features.children()))

            if len(features) >= self.final_layer:
                self.features = nn.ModuleList(features[:self.final_layer])
                break
        
        if not hasattr(self, 'features'):
            raise ValueError(
                f'Could not extract features from all specified layers. '
                f'{architecture} only has {len(features)} layers'
            )

        if frozen:
            self.eval()
            for feature in self.features:
                for param in feature.parameters():
                    param.requires_grad = False

    def forward(self, x):
        y = []
        if self.normalize_in:
            x = x.sub_(self.mean).div_(self.std)
        for layer in range(self.final_layer):
            x = self.features[layer](x)
            if layer in self.layers:
                if self.flatten_layer:    
                    y.append(x.view(x.size(0), -1))
                else:
                    y.append(x)
        return y

    def __str__(self):
        return (
            f'{self.architecture}({self.layers}-{self.pretrained}-'
            f'{self.frozen}-{self.normalize_in}-{self.flatten_layer})'
        )


def SimpleExtractor(architecture, layer, frozen=True, sigmoid_out=True):
    '''
    A simple feature extractor from a single layer for torchvision models
    This is the same as SimpleExtractor from guspih/Perceptual-Autoencoders
    Args:
        architecture (str): The architecture to extract from
        layer (int): The sub-module in 'features' to extract from
        frozen (bool): Whether the network can be trained
        sigmoid_out (bool): Whether to normalize the output with a sigmoid
    Returns (nn.Module): A module that performs feature extraction
    '''
    #return FeatureExtractor(architecture, [layer], True, frozen, False, False)
    # TODO: Somehow add sigmoid_in back
    return ExtractionFlattener(
        FeatureExtractor(architecture, [layer], True, frozen, False)
    )


class ExtractionFlattener(nn.Module):
    '''
    A wrapper for feature extractors that flattens the returned values into a
    single tensor
    Args:
        extactor (nn.Module): The feature extractor to flatten
    '''

    def __init__(self, extractor):
        super().__init__()
        self.extractor = extractor

    def forward(self, x):
        y = self.extractor(x)
        return torch.cat(y)

    def __str__(self):
        return str(self.extractor)


class ExtractionModifier(nn.Module):
    '''
    A wrapper for FeatureExtractor that modifies extracted features, through
    averaging, sorting, etc
    Args:
        extractor (nn.Module): The feature extractor to sort
        modification (str): Modification to apply "mean", "sort"
        return_policy (str): How to return features "new", "tuple", "concat"
        shape_policy (str): Shape of return features "default", "keep"
    '''

    def __init__(
        self, extractor, modification='sort', return_policy='new',
        shape_policy='default'
    ):
        super().__init__()
        self.extractor = extractor
        self.modification = modification
        self.return_policy = return_policy
        self.shape_policy = shape_policy
        mods = ['mean', 'sort']
        returns = ['new', 'tuple', 'concat']
        shapes = ['default', 'keep']
        if not modification in mods:
            raise ValueError(
                f'modification has to be in {mods}, not {modification}'
            )
        if not return_policy in returns:
            raise ValueError(
                f'return_policy has to be in {returns}, not {return_policy}'
            )
        if not shape_policy in shapes:
            raise ValueError(
                f'shape_policy has to be in {shapes}, not {shape_policy}'
            )
    
    def forward(self, x):
        keep = self.shape_policy == 'keep'

        ys = self.extractor(x)
        
        if self.modification == 'mean':
            moded_ys = [
                y.view(y.size(0), y.size(1), -1).mean(-1,keep) for y in ys
            ]
        elif self.modification == 'sort':
            moded_ys = [y.view(y.size(0), y.size(1), -1).sort()[0] for y in ys]
        
        if keep:
            size = [(y.size(0), y.size(1), prod(y.size()[2:]))  for y in ys]
            moded_ys = [y.expand(size[i]) for i, y in enumerate(moded_ys)]
            moded_ys = [y.view(ys[i].size()) for i, y in enumerate(moded_ys)]
        
        if self.return_policy == 'concat':
            return ys + moded_ys
        elif self.return_policy == 'tuple':
            return ys, moded_ys
        return moded_ys

    def __str__(self):
        return (
            f'ExtractionModifier({self.modification},{self.return_policy},'
            f'{str(self.extractor)})'
        )


#TODO: Fix naming convention so it doesn't matter if params are default or not
def extractor_collector(extractor, **parameters):
    '''
    Creates an extractor or collects it from a file if it already exists
    Args:
        extractor (f()->nn.Module): Callable to create a extractor
        parameters (dict): parameters for the callable
    Returns (str): 
    '''
    parameter_string = '-'.join([str(param) for param in parameters.values()])
    extractor_file = f'{extractor.__name__}_{parameter_string}'
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)
    if not extractor_file in os.listdir(checkpoint_folder):
        extractor_model = extractor(**parameters)
        torch.save(extractor_model, checkpoint_folder/extractor_file)
    return checkpoint_folder/extractor_file
