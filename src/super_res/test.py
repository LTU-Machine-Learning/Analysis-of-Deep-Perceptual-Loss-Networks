import cv2
import torch
import torch.nn as nn
from PIL import Image
from network import Transform_Network_Super_x4, Transform_Network_Super_x8
from skimage.metrics import structural_similarity,  peak_signal_noise_ratio
import torchvision.utils as vutils
import math
import numpy as np
import os
import argparse
from tqdm import tqdm
from main_utils import Super_Resolution_Dataset, img_transform, calc_Content_Loss
from torch.utils.data import DataLoader
from loss_networks import *
#from train import calc_Content_Loss 
from torch.autograd import Variable
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, MultiScaleStructuralSimilarityIndexMeasure
import wandb
from skimage import exposure
from torchmetrics.functional import peak_signal_noise_ratio as psnr_ten
import kornia
from dataset_collector import dataset_collector

#from openpyxl import load_workbook
wandb.init(project="Super_Resolution_x8")


# Define log and checkpoint directory
from workspace_path import home_path
checkpoint_dir = home_path / 'checkpoints/super_resolution'
checkpoint_dir.mkdir(parents=True, exist_ok=True)
log_dir = home_path / 'logs/super_resolution'
log_dir.mkdir(parents=True, exist_ok=True)
data_dir = home_path/'datasets'


def calc_psnr(sr_path, hr_path, scale=2, rgb_range=255.0):
    
    sr = cv2.imread(sr_path)
    hr = cv2.imread(hr_path)
    
    sr = np.flip(sr, 2)
    hr = np.flip(hr, 2)
    
    diff = (1.0*sr - 1.0*hr)/rgb_range

    shave = scale
    
    diff[:,:,0] = diff[:,:,0] * 65.738/255
    diff[:,:,1] = diff[:,:,1] * 129.057/255
    diff[:,:,2] = diff[:,:,2] * 25.064/255

    diff = np.sum(diff, axis=2)

    valid = diff[shave:-shave, shave:-shave]
    mse = np.mean(valid**2)
    return -10 * math.log10(mse)

def calc_ssim(sr_path, hr_path):
    ssim = structural_similarity(sr_path, hr_path) # one may obtain a slightly higher output than original setting
    return ssim

def rgb_to_ycbcr(input):
    #code from https://discuss.pytorch.org/t/how-to-change-a-batch-rgb-images-to-ycbcr-images-during-training/3799
    # input is mini-batch N x 3 x H x W of an RGB image
    output = Variable(input.data.new(*input.size()))
    output[:, 0, :, :] = input[:, 0, :, :] * 65.481 + input[:, 1, :, :] * 128.553 + input[:, 2, :, :] * 24.966 + 16
    # similarly write output[:, 1, :, :] and output[:, 2, :, :] using formulas from https://en.wikipedia.org/wiki/YCbCr
    #keep Y channel to compute metrics
    output = output[:,0:1,:,:] # keep only Y channel
    return output

def convert_image(img, rgb_weights):
    sr = cv2.imread(img)
    sr = np.flip(sr, 2)
    sr = 1.0*sr/255.

    shave = 2
    
    sr[:,:,0] = sr[:,:,0] * 65.738/256
    sr[:,:,1] = sr[:,:,1] * 129.057/256
    sr[:,:,2] = sr[:,:,2] * 25.064/256
    
    sr = np.sum(sr, axis=2)
    valid = sr[shave:-shave, shave:-shave]
    return valid


def test(model, test_data, loss_network, device, args, layers, images):
    '''Epoch operation in training phase'''
    
    architecture = loss_network.architecture
    model.eval()
    total_loss = 0
    n_corrects = 0 
    total = 0
    psnrs = 0
    psnr_list = []
    psnr_list_2 = []
    mssim_list = []
    mssim_tensor_list = []
    iteration = 0
    image_index = 1
    total_psnr = 0
    total_psnr_2 = 0
    psnr_tensor_list = []
    rgb_weights = torch.FloatTensor([65.738, 129.057, 25.064]).to(device)

    (log_dir/'inference_image_results').mkdir(parents=True, exist_ok=True)
    (log_dir/f'final_blur/images/x{args.res}').mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_data)):
            
            image = data[0].to(device)
            ref = vutils.save_image(image.data, log_dir/f'inference_image_results/v_utils_original_{image_index}.png', scale_each=True, normalize=False, nrow=1)

            blur_image = data[1].to(device)

            output_image = model(blur_image)
            image_ycbcr = rgb_to_ycbcr(image)
            
            output_ycbcr = rgb_to_ycbcr(output_image)
            psnr_tensor = PeakSignalNoiseRatio().to(device)
            ssim_tensor = StructuralSimilarityIndexMeasure().to(device)
            
            image_korn = kornia.color.rgb_to_ycbcr(image)
            y_i, cr, cb = image_korn.chunk(dim=-3, chunks=3)
            
            output_ycbcr = rgb_to_ycbcr(output_image)
            output_korn = kornia.color.rgb_to_ycbcr(output_image)
            y_o, cr, cb = output_korn.chunk(dim=-3, chunks=3)
            
            y_channel_input = image_ycbcr[:,0:1,:,:]
            y_channel_output = output_ycbcr[:,0:1,:,:]
            
            
            
            psnr_batch = psnr_tensor(y_channel_output, y_channel_input)
            ssim_batch = ssim_tensor(y_channel_output, y_channel_input)
            
            target_features = loss_network(image)
            output_features = loss_network(output_image)
        
            squeezed_image = torch.squeeze(output_image, 0)
            output = squeezed_image.cpu().numpy()
            
            output = Image.fromarray(output, 'RGB')
            vutils.save_image(output_image.data, log_dir/f'inference_image_results/v_utils_output_{image_index}.png', scale_each=True, normalize=False, nrow=1)
            
            src = cv2.imread(str((log_dir/f'inference_image_results/v_utils_output_{image_index}.png').resolve()))
            ref = cv2.imread(str((log_dir/f'inference_image_results/v_utils_original_{image_index}.png').resolve()))
            
            # determine if we are performing multichannel histogram matching
            # and then perform histogram matching itself

            matched = exposure.match_histograms(src, ref, channel_axis=2)
            list_index = image_index - 1
            image_name = images[list_index].split('.')[0]
            cv2.imwrite(str((log_dir/f'final_blur/images/x{args.res}/SRC_{image_name}_{architecture}_layer_{layers}.png').resolve()), src)
            cv2.imwrite(str((log_dir/f'final_blur/images/x{args.res}/{image_name}_{architecture}_layer_{layers}.png').resolve()), matched)
            vutils.save_image(blur_image.data, log_dir/f'final_blur/images/x{args.res}/BLUR_{image_name}_{architecture}_layer_{layers}.png', scale_each=True, normalize=False, nrow=1)
            
            feature_loss = calc_Content_Loss(output_features, target_features)


            wandb_original_images = wandb.Image(image, caption="Super Resolution Images in batch")
            wandb_blur_images = wandb.Image(blur_image, caption="Blur Image")
            wandb_images = wandb.Image(output_image, caption="Output images in batch")
            
            wandb.log({f"Blur_Images_batch_{i}": wandb_blur_images})
            wandb.log({f"Super_Resolution_Images_batch_{i}": wandb_original_images})
            wandb.log({f"Ouput_Images_Training_batch_{i}": wandb_images})
            #------- Metrics ----------
            input_image_path = log_dir/f'inference_image_results/v_utils_original_{image_index}.png'
            output_image_path = log_dir/f'inference_image_results/v_utils_output_{image_index}.png'
            psnr_list.append(psnr_batch)
            
            sr_imgs_y = convert_image(str(input_image_path.resolve()), rgb_weights)
            
            hr_imgs_y = convert_image(str(output_image_path.resolve()), rgb_weights)
            
            ssim = structural_similarity(sr_imgs_y, hr_imgs_y, data_range=1)
            

            psnr_2 = peak_signal_noise_ratio(sr_imgs_y, hr_imgs_y)
            psnr_list_2.append(psnr_2)
            
            mssim_tensor_list.append(ssim_batch.item())
            mssim_list.append(ssim)
            total_loss += feature_loss.item()
            
            image_index += 1
            
        psnr_total = sum(psnr_list) #/len(training_data.dataset)
        mssim_total = sum(mssim_list)
        mssim_tensor_total = sum(mssim_tensor_list)
        total_psnr_2 =sum(psnr_list_2)
        psnrs_tensors = sum(psnr_tensor_list)
        loss = total_loss/len(test_data.dataset)
        
        return loss, psnr_total, mssim_total, total_psnr_2, psnr_list_2, mssim_tensor_total, mssim_tensor_list

 

def main():
    '''Main function'''
    parser = argparse.ArgumentParser(description='Super Resolution')
    parser.add_argument('--device', type=str, help='cuda device number', default='cuda:0')
    parser.add_argument('--feature_layers', type=int, nargs='+', help='layer indices to extract content features', default=None)  
    parser.add_argument('--batch_size', type=int, help='Batch size', default=1)
    parser.add_argument('--imsize', type=int, help='Size for resize image during training', default=288)
    parser.add_argument('--cropsize', type=int, help='Size for crop image durning training', default=None)
    parser.add_argument('--loss_net', type=str, help='Pretrained architecture for calculating losses', default='alexnet')
    parser.add_argument('--res', type=int, help='resolution factor, either 4 or 8', default=4)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--use_experiment_setup', action='store_true', help='Overrides loss network parameters to run predefined experiments')
    parser.add_argument('--no-download', action='store_true', help='Prevents automatic downloading of datasets')


    args = parser.parse_args()
   

    layer_list = []
    
    
    
    loss_nets = {}
    if args.use_experiment_setup:
        for loss_net in architecture_attributes.keys():
            if architecture_attributes[loss_net]['used_in_experiments']:
                loss_nets[loss_net] = architecture_attributes[loss_net]['layers_in_experiments']
    else:
        if args.feature_layers is None:
            if architecture_attributes[args.loss_net]['used_in_experiments']:
                loss_nets[loss_net] = architecture_attributes[args.loss_net]['layers_in_experiments']
            else:
                raise ValueError(f'Running without setting --feature_layers is not available for model {args.loss_net}')
        else:
            loss_nets[args.loss_net] = args.feature_layers
    
    #download SET5, SET14, and BSD100 if they aren't downloaded already
    if not args.no_download:
        print('Collecting Set 5...')
        dataset_collector(dataset='SET5', split='all')
        print('Collecting Set 14...')
        dataset_collector(dataset='SET14', split='all')
        print('Collecting BSD100 (The test set of BSD300)... ')
        dataset_collector(dataset='BSD300', split='test')
    
    test_sets = [data_dir/'SET5', data_dir/'SET14', data_dir/'BSD300/test']


    for loss_net in loss_nets.keys():
        for layer in loss_nets[loss_net]:
            for test_set in test_sets:
            
                image_dir = test_set
                print(f'Network: {loss_net}, Layer: {layer}')
                
                if len(layer_list)<4:
                    layer_list.append(layer)
                device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
                
                if args.res == int(8):
                    transform_network = Transform_Network_Super_x8()
                    print('Super Resolution x', args.res)
                else:
                    transform_network = Transform_Network_Super_x4()
                    
                
                model = loss_net
                model_path = checkpoint_dir/f'sup_x{str(args.res)}_{model}_layer_{layer}.pth'
                transform_network.load_state_dict(torch.load(model_path))
                transform_network = transform_network.to(device)

                print(f'MODEL FROM {model_path} LOADED')

                images = [x for x in os.listdir(image_dir)]
                
                test_data = Super_Resolution_Dataset(image_dir, img_transform(args.imsize, 1), img_transform(args.imsize, args.res))
                print('length of testing data', len(test_data))
                
                test_loader = DataLoader(test_data, args.batch_size, shuffle=False)
                
                loss_network = loss_net
                print(loss_network)
                fe = FeatureExtractor(loss_network, [layer], pretrained=True, frozen=True, normalize_in=False)#, device = device)
                fe = fe.to(device)
            
                test_loss, test_psnr, test_mssim, test_psnr_2, psnr_list, mssim_tensor, ssim_list = test(transform_network, test_loader, fe, device, args, layer, images)
                
                print(f'Inference for {test_set}')
                print('Test loss: {loss: 8.5f}, '\
                        'Test PSNR: {psnr:8.5f}, Test MSSIM: {ssim:8.5f}, Test PSNR_2: {psnr_2:8.5f}, Tensor MSSIM: {ssim_t:8.5f}'.format(
                            loss=test_loss, psnr=test_psnr/len(test_data), ssim=test_mssim/len(test_data), psnr_2=test_psnr_2/len(test_data), ssim_t=mssim_tensor/len(test_data)))
            
                    
        print(f'Finished test with {loss_net} super res x{str(args.res)}')

if __name__ == '__main__':
    main()