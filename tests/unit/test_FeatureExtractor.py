import loss_networks
from loss_networks import FeatureExtractor
import torchvision.models as models
import os
import torch


def test_correct_architecture_features_extracted():
    '''
    Tests whether the features output from pretrained alexnet is the same
    as the features of the max_layer_nr layer of FeatureExtractor of alexnet
    '''

    # setup
    os.environ['TORCH_HOME'] = '/workspace'
    real_alexnet = models.alexnet(pretrained=True)

    # given
    nr_feature_layers = len(list(real_alexnet.features.children()))
    assert nr_feature_layers == loss_networks.architecture_attributes['alexnet']['max_layer_nr']
    x = torch.rand((5, 3, 64, 64))
    feature_extractor = FeatureExtractor(
        'alexnet', [nr_feature_layers-1], normalize_in=False, pretrained=True
    )

    # when
    real_output = real_alexnet.features(x)
    real_output = real_output.view(real_output.size(0), -1)
    test_output = feature_extractor(x)[0]

    # then
    assert torch.all(torch.eq(test_output, real_output))


def test_feature_extraction_max_layer():
    '''
    Tests whether the max_layer_nr attritbute of the architecture_attributes
    table is set correctly. A model with the max_layer_nr as final extraction
    layer should work while a model with one layer later as extraction layer
    should give an error.
    '''

    os.environ['TORCH_HOME'] = '/workspace'
    x = torch.rand((5, 3, 64, 64))

    for architecture in loss_networks.architecture_attributes:
        max_layer = loss_networks.architecture_attributes[architecture]['max_layer_nr']
        feature_extractor_good = FeatureExtractor(
            architecture, list(range(max_layer)),
            normalize_in=False, pretrained=False
        )
        feature_extractor_bad = FeatureExtractor(
            architecture, list(range(max_layer+1)),
            normalize_in=False, pretrained=False
        )
        try:
            feature_extractor_good(x)
        except Exception:
            assert False
        try:
            feature_extractor_bad(x)
        except Exception:
            pass
        else:
            assert False
