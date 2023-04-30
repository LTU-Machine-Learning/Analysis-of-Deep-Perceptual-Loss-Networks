import torch

import loss_networks as ln


def test_loss_network_layer_number():
    '''
    Tests how many extraction layers each network has by creating them all with
    every layer from 0 until they return an index error
    '''
    test_img = torch.rand(1,3,224,224)

    for architecture in ln.architecture_attributes:
        for layer in range(100):
            try:
                extractor = ln.FeatureExtractor(
                    architecture, layers=[layer], pretrained=False
                )
                extractor(test_img)
            except Exception as error:
                print(
                    f'Architecture {architecture} produced an error '
                    f'when extracting from layer nr {layer}:\n  {error}'
                )
                break