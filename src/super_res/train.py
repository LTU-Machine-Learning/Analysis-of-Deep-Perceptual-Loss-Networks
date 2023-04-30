import time
import torch
from network import Transform_Network_Super_x4, Transform_Network_Super_x8
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import torchvision.utils as vutils
import numpy as np
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from loss_networks import *
import wandb
from main_utils import *
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import kornia 
from dataset_collector import dataset_collector


# Define log and checkpoint directory
from workspace_path import home_path
checkpoint_dir = home_path / 'checkpoints/super_resolution'
checkpoint_dir.mkdir(parents=True, exist_ok=True)
log_dir = home_path / 'logs/super_resolution'
log_dir.mkdir(parents=True, exist_ok=True)
data_dir = home_path / 'datasets/COCO2014'


def train_epoch(model, training_data, loss_network, layers, optimizer, device, args, epoch):
    '''Epoch operation in training phase'''
    
    model.train()
    total_loss = 0
    run_psnr = 0
    PSNRs = AverageMeter()
    SSIMs = AverageMeter()
    
    for i, data in enumerate(tqdm(training_data)):
        
        image = data[0].to(device)
        
        blur_image = data[1].to(device)
        
        optimizer.zero_grad()
        
        output_image = model(blur_image)
    
        
        target_features = loss_network(image)
        output_features = loss_network(output_image)
        
        feature_loss = calc_Content_Loss(output_features, target_features)
        
        image_ycbcr = rgb_to_ycbcr(image)
        
        image_korn = kornia.color.rgb_to_ycbcr(image)
        y_i, cr, cb = image_korn.chunk(dim=-3, chunks=3)
        
        output_ycbcr = rgb_to_ycbcr(output_image)
        output_korn = kornia.color.rgb_to_ycbcr(output_image)
        y_o, cr, cb = output_korn.chunk(dim=-3, chunks=3)
        
        y_channel_input = image_ycbcr[:,0:1,:,:]
        y_channel_output = output_ycbcr[:,0:1,:,:]
        
        #------- Metrics ----------
        rgb_weights = torch.FloatTensor([65.481, 128.553, 24.966]).to(device)
        image = torch.matmul(255. * image.permute(0, 2, 3, 1)[:, 4:-4, 4:-4, :], rgb_weights) / 255. + 16.
        output_image = torch.matmul(255. * output_image.permute(0, 2, 3, 1)[:, 4:-4, 4:-4, :], rgb_weights) / 255. + 16.
        
        psnr_tensor = PeakSignalNoiseRatio().to(device)
        psnr_batch = psnr_tensor(output_ycbcr, image_ycbcr)
        
        psnr_korn = psnr_tensor(y_o, y_i)
        PSNRs.update(psnr_batch, blur_image.size(0))
        
        run_psnr += psnr_batch
        feature_loss.backward()
        optimizer.step()
        total_loss += feature_loss.item()
    
    loss = total_loss/len(training_data.dataset)
    final_psnr = run_psnr/len(training_data.dataset)
    return loss, PSNRs


def eval_epoch(model, val_data, loss_network, layers, device, args):
    ''' Validation '''
    
    model.eval()
    
    total_loss = 0
    run_psnr = 0
    PSNRs = AverageMeter()
    SSIMs = AverageMeter()
    
    with torch.no_grad():
        for i, data in enumerate(tqdm(val_data)):
            image = data[0].to(device)
            blur_image = data[1].to(device)
            output_image = model(blur_image)

            target_features = loss_network(image)
            output_features = loss_network(output_image)
            
            image_ycbcr = rgb_to_ycbcr(image)
            output_ycbcr = rgb_to_ycbcr(output_image)
        
            feature_loss = calc_Content_Loss(output_features, target_features)
            
            if i == 10:
                wandb_original_images = wandb.Image(image, caption="Super Resolution Images in batch")
                wandb_blur_images = wandb.Image(blur_image, caption="Blur Image")
                wandb_images = wandb.Image(output_image, caption="Output images in batch")
                wandb.log({f"Super_Resolution_Images_batch_{i}": wandb_original_images})
                wandb.log({f"Ouput_Images_Training_batch_{i}": wandb_images})
            #------- Metrics ----------
            psnr_tensor = PeakSignalNoiseRatio().to(device)
            
            psnr_batch = psnr_tensor(output_ycbcr, image_ycbcr)
            
            run_psnr += psnr_batch
            PSNRs.update(psnr_batch, blur_image.size(0))
            total_loss += feature_loss.item() 
            
        loss = total_loss/len(val_data.dataset)
        return loss, PSNRs 



def train(model, training_data, validation_data, loss_network, layers, optimizer, lr, device, args): #scheduler # after optimizer
    ''' Start training '''

    loss_net_name = loss_network.architecture
    valid_psnrs = []
    num_of_no_improvement = 0
    best_psnr = 0
    
    for epoch_i in range(args.epochs):
        print(f'[Epoch {epoch_i} - Layer {layers}]')

        start = time.time()

        train_loss, train_psnr = train_epoch(model, training_data, loss_network, layers, optimizer, device, args, epoch_i)
        print('Training loss: {loss: 8.5f}, '\
              'Training PSNR: {psnrs.avg:.3f}, elapse: {elapse:3.3f} min'.format(
                  loss=train_loss, psnrs=train_psnr, elapse=(time.time()-start)/60))
        
        start = time.time()

        wandb.log({'epoch': epoch_i, 'train loss': train_loss, 'train psnr': train_psnr.avg})

        if validation_data != None:
            val_loss, val_psnr = eval_epoch(model, validation_data, loss_network, layers, device, args)
            print('Validation loss: {loss: 8.5f}, '\
                    'Validation PSNR: {psnrs.avg:.3f}'.format(loss=val_loss, psnrs=val_psnr))
            
            wandb.log({'epoch': epoch_i, 'val loss': train_loss, 'val psnr': val_psnr.avg})
            
            model_state_dict = model.state_dict()
            checkpoint = {'model': model_state_dict, 'settings': args, 'epoch': epoch_i}

            valid_psnrs += [val_psnr.avg]
            
            if val_psnr.avg > best_psnr:
                wandb.run.summary["best_psnr"] = best_psnr
                model_name = f'{checkpoint_dir}/x{str(args.res)}{loss_net_name}_best_model_{layers}.chkpt'
                print('- [Info] The checkpoint file has been updated.')
                best_psnr = val_psnr.avg
                torch.save(model.state_dict(), f"{checkpoint_dir}/sup_x{str(args.res)}_{loss_net_name}_layer_{layers}.pth")
                
        else:
            model_state_dict = model.state_dict()
            checkpoint = {'model': model_state_dict, 'settings': args, 'epoch': epoch_i}
            torch.save(model.state_dict(), f"{checkpoint_dir}/x{str(args.res)}/sup_{loss_net_name}_layer_{layers}.pth")
            if train_psnr > best_psnr:
                model_name = f'{checkpoint_dir}/x{str(args.res)}/{loss_net_name}_best_psnr_model_{layers}.chkpt'
                print('- [Info] The checkpoint file has been updated.')
                best_psnr = train_psnr
                torch.save(model.state_dict(), f"{checkpoint_dir}/x{str(args.res)}/{loss_net_name}_layer_{layers}_model_best_psnr_state_dict.pth")
                num_of_no_improvement = 0
            else:
                num_of_no_improvement +=1


def main():
    '''Main function'''
    parser = argparse.ArgumentParser(description='Super Resolution')
    parser.add_argument('--device', type=str, help='cuda device id (eg. "cuda:0")', default='cuda:0')
    parser.add_argument('--max-iter', type=int, help='Train iterations', default=15000)
    parser.add_argument('--feature_layers', type=list, nargs='+', help='layer indices to extract content features', default=None)
    parser.add_argument('--batch_size', type=int, help='Batch size', default=4)
    parser.add_argument('--epochs', type=int, default=80, help='epochs for training')
    parser.add_argument('--lr', type=float, help='Learning rate to optimize network', default=0.001)
    parser.add_argument('--check-iter', type=int, help='Number of iteration to check training logs', default=1)
    parser.add_argument('--imsize', type=int, help='Size for resize image during training', default=288)
    parser.add_argument('--cropsize', type=int, help='Size for crop image durning training', default=None)
    parser.add_argument('--loss_net', type=str, help='Pretrained architecture for calculating losses', default='alexnet')
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--res', type=int, help='Resolution factor, either 4 or 8', default=4)
    parser.add_argument('--seed', type=int, help='choose a number for seed', default=42)
    parser.add_argument('--stop', type=int, default=10, help='early stopping when validation does not improve for 10 epochs')
    parser.add_argument('--use_experiment_setup', action='store_true', help='Overrides loss network parameters to run predefined experiments')
    parser.add_argument('--no-download', action='store_true', help='Prevents automatic downloading of datasets')
    parser.add_argument('--num_workers', type=int, default=8, help='The number of workers to use for data management')

    args = parser.parse_args()
    
    #create dir for saved model
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
        print('Directory created for saved models')
    
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

    #download COCO2014 if it isn't downloaded already
    if not args.no_download:
        dataset_collector(dataset='COCO2014', split='test')

    num_workers = args.num_workers

    print('Torch version: ', torch.__version__)
    for loss_net in loss_nets.keys():
        for layer in loss_nets[loss_net]:
        
            runs = wandb.init(project=f'*{loss_net}_sup_x{str(args.res)}', reinit=True)
            wandb.config.update(args)
            print(f'Start Training for {loss_net} with extraction layer {layer}')
            
            device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

            if args.res == int(8):
                transform_network = Transform_Network_Super_x8()
                print('Super Resolution x', args.res)
            else:
                transform_network = Transform_Network_Super_x4()

            
        
            
            
            g = torch.Generator()
            g.manual_seed(args.seed)
            random_seed = args.seed # or any of your favorite number 
            torch.manual_seed(random_seed)
            torch.cuda.manual_seed(random_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            np.random.seed(random_seed)
                
            transform_network = transform_network.to(device)
            
            train_data = Super_Resolution_Dataset(data_dir/'train2014', img_transform(args.imsize, 1), img_transform(args.imsize, args.res)) # for x4 super res and 8 for x8 super res
            print('length of training data', len(train_data))
            
            # We take 10K images from the train set of MSCOCO - we use the same seed for all experiments
            indices = np.arange(len(train_data))
            train_indices, test_indices = train_test_split(indices, train_size=10000, random_state = args.seed) #stratify=train_data.targets

            # Warp into Subsets and DataLoaders
            train_dataset = Subset(train_data, train_indices)
            print('Length of new train set', len(train_dataset))
        
            train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, 
                                    num_workers=num_workers, worker_init_fn=seed_worker, generator=g)
            
            # Validation set
            val_data = Super_Resolution_Dataset(data_dir/'val2014', img_transform(args.imsize, 1), img_transform(args.imsize, args.res)) # for x4 super res and 8 for x8 super res
            val_indices = np.arange(len(val_data))
            val_indices, test_indices = train_test_split(val_indices, train_size=100, random_state = args.seed) #stratify=train_data.targets
            val_dataset = Subset(val_data, val_indices)
            print('Length of new val set', len(val_dataset))
        
            val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False, num_workers=num_workers, 
                                    worker_init_fn=seed_worker, generator=g)
            
            optimizer = torch.optim.Adam(params=transform_network.parameters(), lr=args.lr)
            
            loss_network = args.loss_net
            
            fe = FeatureExtractor(loss_network, [layer], pretrained=True, frozen=True, normalize_in=False)#, device = device)
            fe = fe.to(device)
            
            
            transform_network = train(transform_network, train_loader, val_loader, fe, layer, optimizer, args.lr, device, args)
            
            runs.finish()
        
if __name__ == '__main__':
    main()