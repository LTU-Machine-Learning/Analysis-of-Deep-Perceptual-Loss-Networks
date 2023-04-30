import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.transforms import functional

import numpy as np

# Path to workspace
from workspace_path import home_path

# Set the directory for torchvision models
torch.hub.set_dir(home_path/'checkpoints/hub')

mean_pretrained = torch.tensor([0.485, 0.456, 0.406], requires_grad = False).view((1, 3, 1, 1))
std_pretrained = torch.tensor([0.229, 0.224, 0.225], requires_grad = False).view((1, 3, 1, 1))

def select_FNN(name):
    fnn_model = models.__dict__[name](pretrained=True)
    for params in fnn_model.parameters():
        params.requires_grad = False

    return fnn_model


class FNN_CNN_Net(nn.Module):
    
    def __init__(self, name, layers):
        # Super Constrcutor
        super(FNN_CNN_Net, self).__init__()
        # Selection of pretrained VGG architecture
        self.fnn = select_FNN(name)
        self.layers = layers
        self.model_list = []

        for index in self.layers:
           sub_model = (self.fnn.features)[:index + 1]
           self.model_list.append(sub_model)

        self.models = nn.ModuleList(self.model_list).eval()

        # Use imageNet pretrained image mean and standard deviation if CUDA is not enabled, for smoother learning
        self.register_buffer('mean', mean_pretrained.clone())
        self.register_buffer('std', std_pretrained.clone())


    # Forward pass
    def forward(self, inputX):
        inputX = (inputX - self.mean) / self.std
        result_list = []
        for ii, model in enumerate(self.models):
            x = model(inputX)
            #if ii in self.layers:
            result_list.append(x.view(x.shape[0], -1))

        return result_list

class UNet_block_creator(nn.Module):
    
    def __init__(self, input_channel, output_channel, block_name, transposedConvolution = False, leakyReLU = 0, dropout = 0, batchNorm = True):
        # Super constrcutor call
        super(UNet_block_creator, self).__init__()

        # Building block unit for UNet
        blockOf = nn.Sequential()
    
        # Activation unit decision  
        if not leakyReLU == 0:
            blockOf.add_module('%s_ReLU' % block_name, nn.ReLU(inplace=True))
        else:
            blockOf.add_module('%s_leakyReLU' % block_name, nn.LeakyReLU(leakyReLU, inplace=True))
        
        # Convolution layer - transposed2D or not
        if not transposedConvolution:
            blockOf.add_module('%s_convolution' % block_name, nn.Conv2d(input_channel, output_channel, 4, 2, 1, bias=False))
        else:
            blockOf.add_module('%s_transposed_convolution' % block_name, nn.ConvTranspose2d(input_channel, output_channel, 4, 2, 1, bias=False))

        # Batchnorm decision     
        if batchNorm:
            blockOf.add_module('%s_batch_norm' % block_name, nn.BatchNorm2d(output_channel))

        # Dropout layer addtion decision    
        if dropout != 0:
            blockOf.add_module('%s_dropout' % block_name, nn.Dropout2d(dropout, inplace=True))

        #Composing it to layer
        self.layer = blockOf


    def forward(self, inputX):
        inputX = self.layer(inputX)
        return inputX

class UNet_Model(nn.Module):
    
    def __init__(self, inp_ch, batchNorm_down = True, batchNorm_up = True, leakyReLU_factor_down = 0.2, dropout_downward = 0, leakyReLU_factor_up = 0, dropout_upward = 0):
        super(UNet_Model, self).__init__()
        # downwards of Unet arcitecture
        # down1 - 1 x 2
        self.down1 = UNet_block_creator(inp_ch, inp_ch * 2, "UNet_down_1", transposedConvolution = False, leakyReLU = leakyReLU_factor_down, dropout = dropout_downward, batchNorm = batchNorm_down)
        # down2 - 2 x 4
        self.down2 = UNet_block_creator(inp_ch * 2, inp_ch * 4, "UNet_down_2", transposedConvolution = False, leakyReLU = leakyReLU_factor_down, dropout = dropout_downward, batchNorm = batchNorm_down)
        # down3 - 4 x 8
        self.down3 = UNet_block_creator(inp_ch * 4, inp_ch * 8, "UNet_down_3", transposedConvolution = False, leakyReLU = leakyReLU_factor_down, dropout = dropout_downward, batchNorm = batchNorm_down)
        # down4 - 8 x 8
        self.down4 = UNet_block_creator(inp_ch * 8, inp_ch * 8, "UNet_down_4", transposedConvolution = False, leakyReLU = leakyReLU_factor_down, dropout = dropout_downward, batchNorm = batchNorm_down)

        # upwaords of Unet arcitecture
        # up4 - 8 x 8
        self.up4 = UNet_block_creator(inp_ch * 8, inp_ch * 8, "UNet_up_4", transposedConvolution = True, leakyReLU = leakyReLU_factor_up, dropout = dropout_upward, batchNorm = batchNorm_up)
        # up3 - 16 x 4
        self.up3 = UNet_block_creator(inp_ch * 16, inp_ch * 4, "UNet_up_3", transposedConvolution = True, leakyReLU = leakyReLU_factor_up, dropout = dropout_upward, batchNorm = batchNorm_up)
        # up2 - 8 x 2
        self.up2 = UNet_block_creator(inp_ch * 8, inp_ch * 2, "UNet_up_2", transposedConvolution = True, leakyReLU = leakyReLU_factor_up, dropout = dropout_upward, batchNorm = batchNorm_up)
        # up1 - 4 x 1
        self.up1 = UNet_block_creator(inp_ch * 4, 1, "UNet_up_1", transposedConvolution = True, leakyReLU = leakyReLU_factor_up, dropout = dropout_upward, batchNorm = batchNorm_up)
        #sigmoid
        self.active = nn.Sigmoid()

    def forward(self, input):
        input1 = self.down1(input)
        input2 = self.down2(input1)
        input3 = self.down3(input2)
        input4 = self.down4(input3)

        input = self.up4(input4)
        input = self.up3(torch.cat([input, input3], 1))
        input = self.up2(torch.cat([input, input2], 1))
        input = self.up1(torch.cat([input, input1], 1))

        return self.active(input)
    
class Topology_Aware_Delineation_Net(nn.Module):
    
    def __init__(self, UNet, F_Net, itr = 1):
        super(Topology_Aware_Delineation_Net, self).__init__()
        self.UNet = UNet
        self.F_Net = F_Net
        self.itr = itr

    def forward(self, inputX, labelY):
        result_list = []
        for i in range(self.itr):
            input = torch.cat((inputX, labelY), dim = 1)
            labelY = self.UNet(input)
            label = torch.cat((labelY, labelY, labelY), dim = 1)
            labelY_topology = self.F_Net(label)
            result_list.append([labelY, labelY_topology])

        return result_list


class JointTransform2D:
    """
    Adapted from: https://github.com/cosmic-cortex/pytorch-UNet/blob/master/unet/dataset.py
    Performs augmentation on image and mask when called. Due to the randomness of augmentation transforms,
    it is not enough to simply apply the same Transform from torchvision on the image and mask separetely.
    Doing this will result in messing up the ground truth mask. To circumvent this problem, this class can
    be used, which will take care of the problems above.
    Args:
        resize: tuple describing the size of the resize. If bool(resize) evaluates to False, no resize will
            be taken.
        crop: tuple describing the size of the random crop. If bool(crop) evaluates to False, no crop will
            be taken.
        p_flip: float, the probability of performing a random horizontal flip.
        color_jitter_params: tuple describing the parameters of torchvision.transforms.ColorJitter.
            If bool(color_jitter_params) evaluates to false, no color jitter transformation will be used.
        p_random_affine: float, the probability of performing a random affine transform using
            torchvision.transforms.RandomAffine.
        long_mask: bool, if True, returns the mask as LongTensor in label-encoded format.
    """
    def __init__(self, resize = False, crop = (224, 224), p_flip = 0.5, p_random_affine = 0, long_mask = False):
        self.resize = resize
        self.crop = crop
        self.p_flip = p_flip
        self.p_random_affine = p_random_affine
        self.long_mask = long_mask

    def __call__(self, image, mask):

        if self.resize:
            image, mask = functional.resize(image, self.resize), functional.resize(mask, self.resize)

        # random crop
        if self.crop:
            i, j, h, w = transforms.RandomCrop.get_params(image, self.crop)
            image, mask = functional.crop(image, i, j, h, w), functional.crop(mask, i, j, h, w)

        if np.random.rand() < self.p_flip:
            image, mask = functional.hflip(image), functional.hflip(mask)

        # random affine transform
        if np.random.rand() < self.p_random_affine:
            affine_params = transforms.RandomAffine(180).get_params((-90, 90), (1, 1), (2, 2), (-45, 45), self.crop)
            image, mask = functional.affine(image, *affine_params), functional.affine(mask, *affine_params)

        return image, mask
    
class Training_Utility:
    def iteration_loss(self, preds, VGG_loss, pred_loss, VGG_true, y_true):
        K, example_count = len(preds), len(VGG_true)
        loss_pred, loss_vgg = 0., 0.

        for i in range(K):
            loss_pred += (i + 1) * pred_loss(preds[i][0], y_true)
            for j in range(example_count):
                loss_vgg += (i + 1) * VGG_loss(preds[i][1][j], VGG_true[j])

        coeff = 0.5 * K * (K + 1)

        return loss_pred / coeff, loss_vgg / (coeff * example_count)


    def update_model(self, topology_net_model, path):  
        state = {'t_net': topology_net_model.UNet.state_dict(),}
        torch.save(state, path)

    def epoch_in_train(self, topology_net_model, model_optimizer, dataloader, VGG_loss, prediction_loss, mu, epoch, gpu=True):
        cumulative_VGG_loss = 0.0
        cumulative_prediction_loss = 0.0
        cumulative_training_loss = 0.0
        
        topology_net_model.train()
        example_count = len(dataloader)
        if gpu:
            topology_net_model.cuda()

        for i, data in enumerate(dataloader, 0):
            model_optimizer.zero_grad()
            image_examples, labels = data['input'], data['output']
            init_labels = torch.zeros_like(labels)

            if gpu:
                image_examples = image_examples.cuda()
                labels = labels.cuda()
                init_labels = init_labels.cuda()

            predictions = topology_net_model(image_examples, init_labels)

            VGG_labels = topology_net_model.F_Net(torch.cat((labels, labels, labels), dim = 1))

            current_prediction_loss, current_VGG_loss = self.iteration_loss(predictions, VGG_loss, prediction_loss, VGG_labels, labels)
            training_loss = current_prediction_loss + mu * current_VGG_loss
            training_loss.backward()
            model_optimizer.step()

            cumulative_training_loss += training_loss
            cumulative_prediction_loss += current_prediction_loss
            cumulative_VGG_loss += current_VGG_loss
          
        print("Epoch %d (Train): cumulative loss - %f, cumulative prediction loss - %f, cumulative vgg loss - %f" % (epoch, cumulative_training_loss / example_count, cumulative_prediction_loss / example_count, cumulative_VGG_loss / example_count))

    def epoch_in_val(self, topology_net_model, model_optimizer, dataloader, VGG_loss, prediction_loss, mu, epoch, gpu=True):
        print("Epoch %d (Val)" % epoch)
        cumulative_training_loss = 0.0
        cumulative_prediction_loss = 0.0
        cumulative_VGG_loss = 0.0
        cumulative_bce_loss = 0.0

        topology_net_model.eval()
        if gpu:
            topology_net_model.cuda()

        example_count = len(dataloader)

        with torch.no_grad():
            for i, data in enumerate(dataloader, 0):
                model_optimizer.zero_grad()
                image_examples, labels = data['input'], data['output']
                init_labels = torch.zeros_like(labels)
        
                image_examples = image_examples
                labels = labels
                init_labels = init_labels
                if gpu:
                    image_examples = image_examples.cuda()
                    labels = labels.cuda()
                    init_labels = init_labels.cuda()
                
                predictions = topology_net_model(image_examples, init_labels)
                VGG_labels = topology_net_model.F_Net(torch.cat((labels, labels, labels), dim = 1))

                current_prediction_loss, current_VGG_loss = self.iteration_loss(predictions, VGG_loss, prediction_loss, VGG_labels, labels)
                current_bce_loss = prediction_loss(predictions[-1][0], labels)

                cumulative_training_loss += (current_prediction_loss + mu * current_VGG_loss)#.cpu().detach()
                cumulative_prediction_loss += current_prediction_loss#.data.cpu().detach()
                cumulative_VGG_loss += current_VGG_loss#.data.cpu().detach()
                cumulative_bce_loss += current_bce_loss#.data.cpu().detach()
              
        print("Epoch %d (Val): cumulative loss - %f, cumulative prediction loss - %f, cumulative VGG loss - %f, BCE loss - %f" % 
            (epoch, cumulative_training_loss / example_count, cumulative_prediction_loss / example_count, cumulative_VGG_loss / example_count, cumulative_bce_loss / example_count))

        return cumulative_bce_loss