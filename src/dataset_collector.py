import os
import numpy
import torch
from torch.utils.data import Dataset, TensorDataset, ConcatDataset
import torchvision.datasets
from torchvision.transforms import ToTensor, Resize
from PIL import Image
import urllib.request 
import requests
import zipfile
import tarfile
import shutil
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool

import numpy as np
import pickle
import os, sys
from PIL import Image
import random
import argparse
import time, datetime

from workspace_path import home_path

'''
This file contains functions for loading (and if necessary downloading) the
relevant datasets.
'''

# Path to root folder for datasets
from workspace_path import home_path
root_folder = home_path / 'datasets'


def dataset_collector(dataset, split, **kwargs):
    '''
    Wrapper for all collectors, will use the datasets dict to see if a given
    dataset and split is available and return it using its specified collector
    Args:
        dataset (str): Key for the dataset in the datasets dict
        split (str): An available split for the given dataset
        **kwargs (dict): Any additional parameters
    Returns (torch.utils.data.Dataset)
    '''
    if dataset not in datasets:
        available_datasets = ', '.join(datasets.keys())
        raise ValueError(f'Unexpected value of dataset: {dataset}. '
                         f'Available datasets are {available_datasets}')
    dataset_info = datasets[dataset]
    if split not in dataset_info['split']:
        available_splits = ', '.join(dataset_info['split'])
        raise ValueError(
            f'Unexpected value of split: {split}. '
            f'Available splits for this dataset are {available_splits}')
    kwargs['dataset'] = dataset
    kwargs['split'] = split
    return dataset_info['source'](**kwargs)


def torchvision_collector(dataset, **kwargs):
    '''
    Function for collecting and, if necessary, downloading torchvision datasets
    Args:
        dataset (str): The dataset to collect
        **kwargs (dict): Any additional parameters for collection
    Returns (torch.utils.data.Dataset)
    '''
    if datasets[dataset]['downloadable'] and 'download' not in kwargs:
        kwargs['download'] = True
    return torchvision.datasets.__dict__[dataset](root_folder/dataset,**kwargs)


def mrd_collector(split='train', show=True, **kwargs):
    '''
    Function for collecting and, if necessary, downloading the Massachusetts
    Roads Dataset
    Args:
        split (str): Which split to collect ('train', 'val', or 'test')
        show (bool): Whether to print download progress
        **kwargs (dict): Any additional parameters for collection
    Returns (torch.utils.data.Dataset)
    '''
    folder = root_folder/'MRD'
    folder.mkdir(parents=True, exist_ok=True)

    list_urls = [
        'http://www.cs.toronto.edu/~vmnih/data/mass_roads/train/sat/index.html',
        'https://www.cs.toronto.edu/~vmnih/data/mass_roads/train/map/index.html',
        'https://www.cs.toronto.edu/~vmnih/data/mass_roads/valid/sat/index.html',
        'https://www.cs.toronto.edu/~vmnih/data/mass_roads/valid/map/index.html',
        'https://www.cs.toronto.edu/~vmnih/data/mass_roads/test/sat/index.html',
        'https://www.cs.toronto.edu/~vmnih/data/mass_roads/test/map/index.html'
    ]

    if len(os.listdir(folder/'test/output')) < 49:
        if 'download' in kwargs and not kwargs['download']:
            raise FileNotFoundError(
                'Files are missing. Set \'download\' to True to automatically '
                'download them')
        
        subfolders = [
            folder/'train/input', folder/'train/output',
            folder/'val/input', folder/'val/output',
            folder/'test/input', folder/'test/output']

        for i, (url, subfolder) in enumerate(zip(list_urls, subfolders)):
            subfolder.mkdir(parents=True, exist_ok=True)

            with urllib.request.urlopen(url) as response: 
                html = response.read().decode('utf-8')
            html = html.split("\n")[:-1]

            for j, link in enumerate(html):
                if show:
                    print(
                        f'Downloading MRD dataset part {i+1}/{6} '
                        f'item {j+1}/{len(html)}       ', end='\r'
                    )
                address = link[9:-10].split(">")[0][:-1]
                name = link[9:-10].split(">")[1]

                with open(subfolder/name, "wb") as fp:
                    r = requests.get(address)  
                    fp.write(r.content)
        if show:
            print('Downloading MRD dataset complete                 ')
    folder = folder/split
    if not folder.exists():
        raise ValueError(
            f'Unexpected value of split: {split}. '
            f'\'train\', \'val\' or \'test\' expected'
        )
    return MultipleFolderDataset(
        folder/'input', folder/'output',
        image_transform=kwargs.get('image_transform'),
        to_rgb=kwargs.get('to_rgb')
    )


def coco2014_collector(split='train', **kwargs):
    '''
    Function for collecting and, if necessary, downloading the COCO 2014
    dataset
    Args:
        split (str): Which split to collect ('train', 'val', or 'test')
        **kwargs (dict): Any additional parameters for collection
    Returns (torch.utils.data.Dataset)
    '''
    (root_folder/'COCO2014/train2014').mkdir(parents=True, exist_ok=True)
    (root_folder/'COCO2014/val2014').mkdir(parents=True, exist_ok=True)
    (root_folder/'COCO2014/test2014').mkdir(parents=True, exist_ok=True)
    if (
        len(os.listdir(root_folder/'COCO2014/train2014')) < 2
        or len(os.listdir(root_folder/'COCO2014/val2014')) < 2
        or len(os.listdir(root_folder/'COCO2014/test2014')) < 2
    ):
        if 'download' in kwargs and not kwargs['download']:
            raise FileNotFoundError(
                'Files are missing. Set \'download\' to True to automatically '
                'download them')
        dataset_parts = ['train', 'val', 'test']
        for part in dataset_parts:
            filename = root_folder/f'COCO2014/{part}2014.zip'
            if not os.path.isfile(filename):
                download_raw_url(
                    url=f'http://images.cocodataset.org/zips/{part}2014.zip',
                    save_path=filename)
            with zipfile.ZipFile(filename, 'r') as zip_ref:
                zip_ref.extractall(root_folder/'COCO2014')
    return MultipleFolderDataset(
        root_folder/f'COCO2014/{split}2014',
        image_transform=kwargs.get('image_transform')
    )

def bsd300_collector(split='test', **kwargs):
    '''
    Function for collecting and, if necessary, downloadingt the BDS300 dataset
    Args:
        split (str): Whuch split to collect. 'test'=BSD100 ('train', 'test')
        **kwargs (dict): Any additional parameters for collection
    '''
    bsd300_folder = root_folder/'BSD300'
    (bsd300_folder/'train').mkdir(parents=True, exist_ok=True)
    (bsd300_folder/'test').mkdir(parents=True, exist_ok=True)
    if (
        len(os.listdir(bsd300_folder/'train')) < 200
        or len(os.listdir(bsd300_folder/'test')) < 100
    ):
        if 'download' in kwargs and not kwargs['download']:
            raise FileNotFoundError(
                'Files are missing. Set \'download\' to True to automatically '
                'download them'
            )
        filename=bsd300_folder/'BSDS300-images.tgz'
        if not filename.exists():
            download_raw_url(
                'https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz',
                save_path=filename
            )
        with tarfile.TarFile(filename, 'r') as tar_ref:
            train_members = [
                tarinfo for tarinfo in tar_ref.getmembers()
                if 'train/' in tarinfo.name and tarinfo.isfile()
            ]
            test_members = [
                tarinfo for tarinfo in tar_ref.getmembers()
                if 'test/' in tarinfo.name and tarinfo.isfile()
            ]
            for member in train_members:
                member.path = member.name.split('/')[-1]
            for member in test_members:
                member.path = member.name.split('/')[-1]
            tar_ref.extractall(bsd300_folder/'train', train_members)
            tar_ref.extractall(bsd300_folder/'test', test_members)
    return MultipleFolderDataset(
        bsd300_folder/split,
        image_transform=kwargs.get('image_transform')
    )

def set_collector(dataset='SET5', **kwargs):
    '''
    Function for collecting and, if necessary, downloadingt the Set 5 dataset
    Args:
        split (str): Which "set" dataset to collect. ('all')
        **kwargs (dict): Any additional parameters for collection
    '''
    set_folder = root_folder/dataset
    (set_folder).mkdir(parents=True, exist_ok=True)
    
    if dataset == 'SET5':
        set_len=5
        set_name='Set5'
    elif dataset == 'SET14':
        set_len=14
        set_name='Set14'
    else:
        raise ValueError(f'Nonexistant dataset {dataset}')
    
    if len(os.listdir(set_folder)) < set_len:
        if 'download' in kwargs and not kwargs['download']:
            raise FileNotFoundError(
                'Files are missing. Set \'download\' to True to automatically '
                'download them'
            )
        for i in [f'0{x}' if x<10 else x for x in range(1,set_len+1)]:
            download_raw_url(
                f'https://github.com/jbhuang0604/SelfExSR/blob/master/data/{set_name}/image_SRF_2/img_0{i}_SRF_2_HR.png?raw=true',
                save_path=set_folder/f'0{i}.png',
                chunk_size=0
            )
    return MultipleFolderDataset(
        set_folder,
        image_transform=kwargs.get('image_transform')
    )

def bapps_collector(split='train', subsplit='all', **kwargs):
    '''
    Function for collecting and, if necessary, downloading the BAPPS dataset
    Args:
        split (str): Which split to collect ('train', 'val', 'jnd/val')
        subsplit (str): Which subsplit to collect ('all' collects all)
        **kwargs (dict): Any additional parameters for collection
    Returns (torch.utils.data.Dataset)
    '''
    (root_folder/'BAPPS/train').mkdir(parents=True, exist_ok=True)
    (root_folder/'BAPPS/val').mkdir(parents=True, exist_ok=True)
    (root_folder/'BAPPS/jnd/val').mkdir(parents=True, exist_ok=True)
    if (
        len(os.listdir(root_folder/'BAPPS/train')) < 3
        or len(os.listdir(root_folder/'BAPPS/val')) < 6
        or len(os.listdir(root_folder/'BAPPS/jnd/val')) < 2
    ):
        if 'download' in kwargs and not kwargs['download']:
            raise FileNotFoundError(
                'Files are missing. Set \'download\' to True to automatically '
                'download them')
        print('Downloading BAPPS dataset...')
        dataset_link = 'https://people.eecs.berkeley.edu/~rich.zhang/projects/2018_perceptual/dataset'
        dataset_parts = ['twoafc_train', 'twoafc_val', 'jnd']
        for part in dataset_parts:
            filename = root_folder/f'BAPPS/{part}.tar.gz'
            if not os.path.isfile(filename):
                download_raw_url(url=f'{dataset_link}/{part}.tar.gz',
                                 save_path=filename)
            with tarfile.TarFile(filename, 'r') as tar_ref:
                tar_ref.extractall(root_folder/'BAPPS')
    subsplits = os.listdir(root_folder/f'BAPPS/{split}')
    if subsplit == 'all':
        ret = []
        for subsplit in subsplits:
            dirs = os.listdir(root_folder/f'BAPPS/{split}/{subsplit}')
            paths = [root_folder/f'BAPPS/{split}/{subsplit}/{d}' for d in dirs]
            ret.append(MultipleFolderDataset(
                *paths, name=subsplit,
                image_transform=kwargs.get('image_transform'))
            )
        return ConcatDataset(ret)
    elif subsplit in subsplits:
        dirs = os.listdir(root_folder/f'BAPPS/{split}/{subsplit}')
        paths = [root_folder/f'BAPPS/{split}/{subsplit}/{d}' for d in dirs]
        return MultipleFolderDataset(
            *paths, name=subsplit,
            image_transform=kwargs.get('image_transform')
        )
    else:
        raise ValueError(
            f'Unexpected value of subsplit: {subsplit}. '
            f'Expected any of: all, {", ".join(subsplits+["all"])}'
        )


# Dictionary of available datasets and their attributes and parameters
datasets = {
    'STL10': {
        'full_name': 'STL-10 dataset',
        'source': torchvision_collector,
        'downloadable': True,
        'split': ['train', 'test', 'unlabeled', 'train+unlabeled'],
        'folds': [None, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'output_format': 'index',
        'nr_of_classes': 10
    },
    'SVHN': {
        'full_name': 'Street View House Numbers dataset',
        'source': torchvision_collector,
        'downloadable': True,
        'split': ['train', 'test', 'extra'],
        'output_format': 'index',
        'nr_of_classes': 10
    },
    'MRD': {
        'full_name': 'Massachusetts Roads Dataset',
        'source': mrd_collector,
        'downloadable': True,
        'split': ['train', 'val', 'test'],
        'output_format': None,  # TODO: annotate the format (eg 'onehot'),
        #'transforms': None
    },
    'COCO2014': {
        'full_name': 'Common Objects in Context 2014',
        'source': coco2014_collector,
        'downloadable': True,
        'split': ['train', 'val', 'test'],
        'output_format': 'none'
    },
    'BSD300': {
        'full_name': 'Berkeley Segmentation Dataset 300',
        'source': bsd300_collector,
        'downloadable': True,
        'split': ['train', 'test'],
        'output_format': 'none'
    },
    'SET5': {
        'full_name': 'Set 5',
        'source': set_collector,
        'downloadable': True,
        'split': ['all'],
        'output_format': 'none'
    },
    'SET14': {
        'full_name': 'Set 14',
        'source': set_collector,
        'downloadable': True,
        'split': ['all'],
        'output_format': 'none'
    },
    'BAPPS': {
        'full_name': 'Berkeley Adobe Perceptual Patch Similarity',
        'source': bapps_collector,
        'downloadable': True,
        'split': ['train', 'val', 'jnd/val'],
        'output_format': None  # TODO: annotate the format (eg 'onehot')
    }
}


def download_raw_url(url, save_path, show=False, chunk_size=128, decode=False):
    '''
    Downloads raw data from url. Reworked from:
    https://stackoverflow.com/questions/9419162/download-returned-zip-file-from-url
    Args:
        url (str): Url of raw data
        save_path (str): File name to store data under
        show (bool): Whether to print what is being downloaded
        chunk_size (int): How large chunks of data to collect at a time
    '''
    if show:
        print(f'\rDownloading URL: {url}', end='')
    r = requests.get(url, stream=True)
    if decode:
        r.raw.decode_content = True
    with open(save_path, 'wb') as fd:
        if chunk_size > 0:
            for chunk in r.iter_content(chunk_size=chunk_size):
                fd.write(chunk)
        else:
            shutil.copyfileobj(r.raw, fd)


def download_single_file(
    address, folder, name, max_items=None, show=False, show_string=''
):
    with open(folder/name, "wb") as fp:
        r = requests.get(address)  
        fp.write(r.content)
    if show:
        print(
            f'{show_string} item '
            f'{len(list(folder.glob("*")))}/{max_items}'
        )
    return


class MultipleFolderDataset(Dataset):
    '''
    A dataset for loading data where data is contained in folders where each
    matching group of data has the same name (with possibly different
    file-endings)
    Allowed file types: .png, .tif, .tiff, .jpg, .jpeg, .bmp, .npy 
    Args:
        *args (str): Paths to the folders to extract from
        name (str): A name to be returned together with each datapoint
        image_transform (nn.Module): Transform applied to images individually
        rgb_convert (bool): Whether to format all images as RGB
    '''
    def __init__(self, *args, name=None, image_transform=None, to_rgb=None):
        super().__init__()
        if len(args) < 1:
            raise RuntimeError('Must be given at least one path')
        self.name = name
        acceptable_endings = [
            'png', 'tif', 'tiff', 'jpg', 'jpeg', 'bmp', 'npy'
        ]
        folder_files = []
        for folder in args:
            files = os.listdir(folder)
            folder_files.append({
                f[:f.index('.')]: f[f.index('.') + 1:]
                for f in files if f[f.index('.') + 1:] in acceptable_endings
            })
        self.data_paths = []
        for filename, ending in folder_files[0].items(): 
            paths = [f'{args[0]}/{filename}.{ending}']
            for folder, arg in zip(folder_files[1:], args[1:]):
                if filename in folder:
                    paths.append(f'{arg}/{filename}.{folder[filename]}')
                else:
                    break
            if len(paths) != len(args):
                continue
            self.data_paths.append(paths)
        self.image_transform = image_transform
        if self.image_transform is None:
            self.image_transform = ToTensor()
        self.to_rbg = to_rgb


    def __getitem__(self, index):
        image_endings = ['png', 'tif', 'tiff', 'jpg', 'jpeg', 'bmp']
        npy_endings = ['npy']

        ret = {} #= []
        folders, data = [], []
        for path in self.data_paths[index]:
            ending = path[path.index('.') + 1:]
            folder = path.split('/')[-2]
            if ending in image_endings:
                image = Image.open(path)
                if self.to_rbg is None or self.to_rbg:
                    image = image.convert(mode='RGB')
                ret[folder] = self.image_transform(image)
            elif ending in npy_endings:
                ret[folder] = torch.from_numpy(numpy.load(path))
            else:
                raise RuntimeError('Loading from unsupported file type')
        if not self.name is None:
            ret['name'] = self.name
        return ret

    def __len__(self):
        return len(self.data_paths)
    
class JointTransformDataset(Dataset):
    '''
    A wrapper dataset that applies a given joint transform to the \'input\' and
    \'output\' entries extracted from another dataset.
    Args:
        dataset (utils.data.Dataset): A torch dataset with the actual data
        joint_transform (nn.Module): A transform that is applied to two inputs
    '''

    def __init__(self, dataset, joint_transform):
        super().__init__()
        self.dataset = dataset
        self.joint_transform = joint_transform

    def __getitem__(self, index):
        item = self.dataset[index]
        inp, out = self.joint_transform(item['input'], item['output'])
        item['input'] = inp
        item['output'] = out
        return item
    
    def __len__(self):
        return len(self.dataset)
