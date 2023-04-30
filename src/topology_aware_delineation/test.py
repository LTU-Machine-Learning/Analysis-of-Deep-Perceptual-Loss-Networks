import argparse
import csv

import numpy as np
import torch
from torch.utils.data import DataLoader

from train import *
from dataset_collector import dataset_collector, JointTransformDataset

# Path to workspace
from workspace_path import home_path
checkpoint_dir = home_path / 'checkpoints/delineation'
checkpoint_dir.mkdir(parents=True, exist_ok=True)

log_dir = home_path / 'logs/delineation'
log_dir.mkdir(parents=True, exist_ok=True)

data_dir = home_path / 'datasets/MRD'
data_dir.mkdir(parents=True, exist_ok=True)

def metrics(target, predict):

    target = target.cpu().numpy()
    #predict = predict.cpu().numpy()
    TP = float(np.logical_and(target > 0, predict > 0).sum())
    FN = float(np.logical_and(target > 0, predict < 0.1).sum())
    FP = float(np.logical_and(target < 0.1, predict > 0).sum())

    print(FN)

    complete = TP / (TP + FN)
    correct = TP / (TP + FP)
    quality = TP / (TP + FP + FN)

    return complete, correct, quality


if __name__ == "__main__":

    # Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--K', type = int, default = 3, help = 'number of iterative steps')
    # Model settings
    parser.add_argument('--prelud', type = float, default = 0.2, help = 'prelu coeff for down-sampling block')
    parser.add_argument('--preluu', type = float, default = 0., help = 'prelu coeff for up-sampling block')
    parser.add_argument('--dropout', type = float, default = 0., help = 'dropout coeff for all blocks')
    parser.add_argument('--bn', type = bool, default = True, help = 'batch-normalization coeff for all blocks')
    parser.add_argument('--color', type = bool, default = True, help = 'True if input is RGB, False otherwise')
    # Test setting
    parser.add_argument('--thresh', type = float, default = 0.3, help = 'the threshold for identifying boundary')
    parser.add_argument('--no_gpu', action='store_true', help = 'GPUs will not be used even if they are available')
    parser.add_argument('--no_download', action='store_true', help='Prevents automatic downloading of datasets')
    
    
    option = parser.parse_args()
    print (option)

    # Use GPU if it is available unless given --no-gpu flag
    gpu = (not option.no_gpu) and torch.cuda.is_available()

    transform_test = JointTransform2D(resize = 224, crop = False, p_flip = 0)

    # models
    unet_models_dict = {}
    results_dict = {}
    test_image_count = 0
    # Set up model 
    inp_ch = 4 if option.color else 2
    for unet_model in list(checkpoint_dir.iterdir()):
        unet = UNet_Model(inp_ch, batchNorm_down = option.bn, batchNorm_up = True, leakyReLU_factor_down = option.prelud, dropout_downward = option.dropout, leakyReLU_factor_up = option.preluu, dropout_upward = option.dropout)
        if gpu:
            unet.cuda()
        unet.load_state_dict(torch.load(checkpoint_dir/unet_model)["t_net"])
        encoder_name = unet_model.stem
        unet_models_dict[encoder_name] = unet
        print(f"Unet Model trained with {encoder_name} feature extractor is loaded")
        results_dict[encoder_name] = {
            "Completeness": 0.0,
            "Correctness":0.0,
            "Quality":0.0,
            "Samples": 0
            }
    
    testset = dataset_collector(dataset='MRD', split='val', to_rgb=False, download = not option.no_download)
    testset = JointTransformDataset(testset, transform_test)
    test_loader = DataLoader(testset, batch_size = 1, shuffle = False)
    
    transform_test = JointTransform2D(resize = 224, crop = False, p_flip = 0)
    for data in test_loader:
        input, target = data['input'], data['output']
        test_image_count += 1
        
        lpredicts = []
        
        with torch.no_grad():
            init_target = torch.zeros_like(target)

            if gpu:
                input = input.cuda()
                target = target.cuda()
                init_target = init_target.cuda()

            for k in range(option.K):
                curr_input = torch.cat((input, init_target), dim = 1)
                #test all models
                for unet_model_by_encoder in list(unet_models_dict.keys()):
                    
                    unet = unet_models_dict[unet_model_by_encoder]
                    init_target = unet(curr_input)
                    lpredicts.append(torch.squeeze(init_target).data.cpu().detach().numpy() > option.thresh)
            
                    for i, img in enumerate(lpredicts):
                        complete, correct, quality = metrics(target > 0.5, img)
                        print("Iteration %d: " % (i + 1))
                        print("Completeness: %f" % complete)
                        print("Correctness : %f" % correct)
                        print("Quality     : %f" % quality)
                        results_dict[unet_model_by_encoder]["Completeness"] += complete
                        results_dict[unet_model_by_encoder]["Correctness"] += correct
                        results_dict[unet_model_by_encoder]["Quality"] += quality

    
    with open(log_dir/'results.csv' , 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['encoder', 'Completeness', 'Correctness', 'Quality'])
        for encoder in list(results_dict.keys()):
            print(encoder, results_dict[encoder]['Completeness']/test_image_count, results_dict[encoder]['Correctness']/test_image_count, results_dict[encoder]['Quality']/test_image_count)
            csv_writer.writerow([encoder, results_dict[encoder]['Completeness']/test_image_count, results_dict[encoder]['Correctness']/test_image_count, results_dict[encoder]['Quality']/test_image_count])
        
    

    



    