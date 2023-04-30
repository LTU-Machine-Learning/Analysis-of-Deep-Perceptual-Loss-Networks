import os
from PIL import Image, ImageFilter
import math
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import cv2
import random
import torchvision.transforms.functional as FT
from torch.autograd import Variable

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# mean and std of ImageNet to use pre-trained VGG
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
mse_criterion = torch.nn.MSELoss(reduction='mean')
normalize = transforms.Normalize(mean=IMAGENET_MEAN,
                                 std=IMAGENET_STD)

denormalize = transforms.Normalize(mean=[-mean/std for mean, std in zip(IMAGENET_MEAN, IMAGENET_STD)],
        std=[1/std for std in IMAGENET_STD])

unloader = transforms.ToPILImage()

class ImageFolder(torch.utils.data.Dataset):
    def __init__(self, root_path, transform):
        super(ImageFolder, self).__init__()
        
        self.file_names = sorted(os.listdir(root_path))
        self.root_path = root_path        
        self.transform = transform
        
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, index):
        image = Image.open(os.path.join(self.root_path + self.file_names[index])).convert("RGB")
        return self.transform(image)


def get_transformer(imsize=None):
    transform = transforms.Compose([
        transforms.Resize(imsize),
        transforms.ToTensor()])
    return transform


def calc_psnr(sr_path, hr_path, scale=2, rgb_range=255.0):
    
    sr = cv2.imread(sr_path)
    hr = cv2.imread(hr_path)
    
    sr = np.flip(sr, 2)
    hr = np.flip(hr, 2)
    
    diff = (1.0*sr - 1.0*hr)/rgb_range

    shave = scale
    
    diff[:,:,0] = diff[:,:,0] * 65.738/255
    diff[:,:,1] = diff[:,:,1] * 129.057/255
    diff[:,:,2] = diff[:,:,2] * 25.064/255 #256

    diff = np.sum(diff, axis=2)

    valid = diff[shave:-shave, shave:-shave]
    mse = np.mean(valid**2)
    return -10 * math.log10(mse)

rgb_weights = torch.FloatTensor([65.481, 128.553, 24.966])
imagenet_mean = torch.FloatTensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(2)

def convert_image(img, source, target):
    """
    Convert an image from a source format to a target format.
    :param img: image
    :param source: source format, one of 'pil' (PIL image), '[0, 1]' or '[-1, 1]' (pixel value ranges)
    :param target: target format, one of 'pil' (PIL image), '[0, 255]', '[0, 1]', '[-1, 1]' (pixel value ranges),
                   'imagenet-norm' (pixel values standardized by imagenet mean and std.),
                   'y-channel' (luminance channel Y in the YCbCr color format, used to calculate PSNR and SSIM)
    :return: converted image
    """
    assert source in {'pil', '[0, 1]', '[-1, 1]'}, "Cannot convert from source format %s!" % source
    assert target in {'pil', '[0, 255]', '[0, 1]', '[-1, 1]', 'imagenet-norm',
                      'y-channel'}, "Cannot convert to target format %s!" % target

    # Convert from source to [0, 1]
    if source == 'pil':
        img = FT.to_tensor(img)

    elif source == '[0, 1]':
        pass  # already in [0, 1]

    elif source == '[-1, 1]':
        img = (img + 1.) / 2.

    # Convert from [0, 1] to target
    if target == 'pil':
        img = FT.to_pil_image(img)

    elif target == '[0, 255]':
        img = 255. * img

    elif target == '[0, 1]':
        pass  # already in [0, 1]

    elif target == '[-1, 1]':
        img = 2. * img - 1.

    elif target == 'y-channel':
        # Based on definitions at https://github.com/xinntao/BasicSR/wiki/Color-conversion-in-SR
        # torch.dot() does not work the same way as numpy.dot()
        # So, use torch.matmul() to find the dot product between the last dimension of an 4-D tensor and a 1-D tensor
        img = torch.matmul(255. * img.permute(0, 2, 3, 1)[:, 4:-4, 4:-4, :], rgb_weights) / 255. + 16.

    
    '''
    elif target == 'imagenet-norm':
        if img.ndimension() == 3:
            img = (img - imagenet_mean) / imagenet_std
        elif img.ndimension() == 4:
            img = (img - imagenet_mean_cuda) / imagenet_std_cuda
    '''
    
    return img

def imsave(tensor, path):
    if tensor.is_cuda:
        tensor = tensor.cpu()
    tensor = torchvision.utils.make_grid(tensor)    
    torchvision.utils.save_image(denormalize(tensor).clamp_(0.0, 1.0), path)
    return None
    
def imload(path, imsize=None, cropsize=None):
    transformer = get_transformer(imsize)
    return transformer(Image.open(path).convert("RGB")).unsqueeze(0)


class Super_Resolution_Dataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, input_transform=None, target_transform=None):
        super(Super_Resolution_Dataset, self).__init__()
        self.image_dir = image_dir
        self.image_filenames = [os.path.join(image_dir, x) for x in os.listdir(image_dir)]
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index): 
        image_name = os.path.join(self.image_dir, self.image_filenames[index])
        
        image = Image.open(image_name)
        if image.mode != 'RGB': 
            image = image.convert('RGB')
        
        target = image.copy()
        target = target.filter(ImageFilter.GaussianBlur(radius=1))
        
        if self.input_transform:
            input = self.input_transform(image)
        if self.target_transform:
            target = self.target_transform(target)
        
        return input, target

    def __len__(self):
        return len(self.image_filenames)

def img_transform(crop_size, upscale_factor=1):
    if upscale_factor == 1:
        transform = transforms.Compose([
            transforms.Resize(crop_size // upscale_factor),
            transforms.CenterCrop(crop_size // upscale_factor),
            transforms.ToTensor(),
            ])
    else:
        transform = transforms.Compose([
            transforms.Resize(crop_size // upscale_factor, interpolation = transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(crop_size // upscale_factor),
            transforms.ToTensor(),
            ])
    return transform

def extract_features(model, x, layers):
    features = list()
    for index, layer in enumerate(model):
        x = layer(x)
        if index in layers:
            features.append(x)
    return features

def calc_Content_Loss(features, targets, weights=None):
    if weights is None:
        weights = [1/len(features)] * len(features)
    
    content_loss = 0
    for f, t, w in zip(features, targets, weights):
        content_loss += mse_criterion(f, t) * w
    return content_loss


def cal_feature_loss(output, target, strength=None):
    if strength is None:
        strength = [1/len(output)] * len(output)  
    feature_loss = 0
    for out, tar, st in zip(output, target, strength):
        feature_loss += st * mse_criterion(out, tar) 
    return feature_loss


def psnr(label, outputs, max_val=255.):
    """
    Compute Peak Signal to Noise Ratio (the higher the better).
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE).
    https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio#Definition
    First we need to convert torch tensors to NumPy operable.
    """
    label = label.cpu().detach().numpy()
    outputs = outputs.cpu().detach().numpy()
    img_diff = outputs - label
    rmse = math.sqrt(np.mean((img_diff) ** 2))
    if rmse == 0:
        return 100
    else:
        PSNR = 20 * math.log10(max_val / rmse)
        return PSNR


def rgb_to_ycbcr(input):
    #code from https://discuss.pytorch.org/t/how-to-change-a-batch-rgb-images-to-ycbcr-images-during-training/3799
    # input is mini-batch N x 3 x H x W of an RGB image
    output = Variable(input.data.new(*input.size()))
    output[:, 0, :, :] = input[:, 0, :, :] * 65.481 + input[:, 1, :, :] * 128.553 + input[:, 2, :, :] * 24.966 + 16
    # similarly write output[:, 1, :, :] and output[:, 2, :, :] using formulas from https://en.wikipedia.org/wiki/YCbCr
    #keep Y channel to compute metrics
    output = output[:,0:1,:,:] # keep only Y channel
    return output


def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    
    batch = batch.div_(255.0)
    return (batch - mean) / std