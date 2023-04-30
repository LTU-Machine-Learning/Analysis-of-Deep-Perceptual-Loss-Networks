import dataset_collector
from torchvision.transforms import ToTensor
from PIL import Image
import os
import torch
import random

'''
These tests focus on the ISBI12 and Cracks datasets as they are reasonably
small and together cover much of what needs to be tested.
'''


def test_isbi12_collection():
    '''
    Tests the collection and potential downloading of the ISBI12 dataset
    '''

    dataset_collector.dataset_collector(dataset='ISBI12', split='train')
    dataset_collector.dataset_collector(dataset='ISBI12', split='unlabeled')
    assert True


def test_cracks_collection():
    '''
    Tests the collection and potential downloading of the Cracks dataset
    '''

    dataset_collector.dataset_collector(dataset='Cracks', split='all')
    assert True


def test_isbi12_correct_labelling():
    '''
    Tests that the correct images and labels gets paired up when collecting
    the train split of ISBI12 dataset
    '''

    isbi = dataset_collector.dataset_collector(dataset='ISBI12', split='train')
    images = ToTensor()(
        Image.open('/workspace/datasets/ISBI12/train-volume.tif')
    )
    labels = ToTensor()(
        Image.open('/workspace/datasets/ISBI12/train-labels.tif')
    )
    for i in range(len(isbi)):
        dataset_image, dataset_label = isbi[i]
        image_found = False
        for j, img in enumerate(images):
            if torch.all(torch.eq(img, dataset_image)):
                label = labels[j]
                assert torch.all(torch.eq(label, dataset_label))
                image_found = True
                break
        if not image_found:
            assert False


def test_cracks_correct_labelling():
    '''
    Tests that the correct images and labels gets paired up when collecting
    the Cracks dataset
    '''

    cracks = dataset_collector.dataset_collector(dataset='Cracks', split='all')

    for i in range(len(cracks)):
        dataset_image, dataset_label = cracks[i]
        image_found = False
        for img in os.listdir('/workspace/datasets/Cracks/input'):
            actual_image = ToTensor()(
                Image.open(f'/workspace/datasets/Cracks/input/{img}')
            )
            if torch.all(torch.eq(actual_image, dataset_image)):
                label = img[:img.index('.')] + '.bmp'
                actual_label = ToTensor()(
                    Image.open(f'/workspace/datasets/Cracks/output/{label}')
                )
                assert torch.all(torch.eq(actual_label, dataset_label))
                image_found = True
                break
        if not image_found:
            assert False
