import torch
import torch.nn as nn

class Transform_Network_Style(nn.Module):    
    def __init__(self):        
        super(Transform_Network_Style, self).__init__()        
        
        self.layers = nn.Sequential(            
            ConvLayer(3, 32, 9, 1),
            ConvLayer(32, 64, 3, 2),
            ConvLayer(64, 128, 3, 2),
            
            ResidualLayer(128, 128, 3, 1),
            ResidualLayer(128, 128, 3, 1),
            ResidualLayer(128, 128, 3, 1),
            ResidualLayer(128, 128, 3, 1),
            ResidualLayer(128, 128, 3, 1),
            
            DeconvLayer(128, 64, 3, 1),
            DeconvLayer(64, 32, 3, 1),
            ConvLayer(32, 3, 9, 1, activation='tanh'))
        
    def forward(self, x):
        x = self.layers(x)

        #Scaled from 0 to 1 instead of 0 - 255
        x = torch.add(x, 1.)
        x = torch.mul(x, 0.5)
        return x

class Transform_Network_Super_x4(nn.Module):    
    def __init__(self):        
        super(Transform_Network_Super_x4, self).__init__()        
        
        self.layers = nn.Sequential(            
            ConvLayer(3, 64, 9, 1),
            ResidualLayer(64, 64, 3, 1),
            ResidualLayer(64, 64, 3, 1),
            ResidualLayer(64, 64, 3, 1),
            ResidualLayer(64, 64, 3, 1),
            DeconvLayer(64, 64, 3, 1),
            DeconvLayer(64, 64, 3, 1),
            ConvLayer(64, 3, 9, 1, activation='tanh'))
        
    def forward(self, x):
        x = self.layers(x)

        #Scaled from 0 to 1 instead of 0 - 255
        x = torch.add(x, 1.)
        x = torch.mul(x, 0.5)
        return x



class Transform_Network_Super_x8(nn.Module):    
    def __init__(self):        
        super(Transform_Network_Super_x8, self).__init__()        
        
        self.layers = nn.Sequential(            
            ConvLayer(3, 64, 9, 1),
            ResidualLayer(64, 64, 3, 1),
            ResidualLayer(64, 64, 3, 1),
            ResidualLayer(64, 64, 3, 1),
            ResidualLayer(64, 64, 3, 1),
            DeconvLayer(64, 64, 3, 1),
            DeconvLayer(64, 64, 3, 1),
            DeconvLayer(64, 64, 3, 1),
            ConvLayer(64, 3, 9, 1, activation='tanh'))
        
    def forward(self, x):
        x = self.layers(x)

        #Scaled from 0 to 1 instead of 0 - 255
        x = torch.add(x, 1.)
        x = torch.mul(x, 0.5)
        return x


class ConvLayer(nn.Module):    
    def __init__(self, in_ch, out_ch, kernel_size, stride, pad='reflect', activation='relu', normalization='batch'):        
        super(ConvLayer, self).__init__()
        
        # padding
        if pad == 'reflect':            
            self.pad = nn.ReflectionPad2d(kernel_size//2)
        elif pad == 'zero':
            self.pad = nn.ZeroPad2d(kernel_size//2)
        else:
            raise NotImplementedError("Not expected pad flag !!!")
    
            
        # convolution
        self.conv_layer = nn.Conv2d(in_ch, out_ch, 
                                    kernel_size=kernel_size,
                                    stride=stride)
        
        # activation
        if activation == 'relu':
            self.activation = nn.ReLU()        
        elif activation == 'linear':
            self.activation = lambda x : x
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise NotImplementedError("Not expected activation flag !!!")

        # normalization 
        if normalization == 'instance':            
            self.normalization = nn.InstanceNorm2d(out_ch, affine=True)
        if normalization == 'batch':            
            self.normalization = nn.BatchNorm2d(out_ch, affine=True)

        else:
            raise NotImplementedError("Not expected normalization flag !!!")

    def forward(self, x):
        x = self.pad(x)
        x = self.conv_layer(x)      
        x = self.normalization(x)
        x = self.activation(x)        
        return x
    
class ResidualLayer(nn.Module):    
    def __init__(self, in_ch, out_ch, kernel_size, stride, pad='reflect', normalization='batch'):        
        super(ResidualLayer, self).__init__()
        
        self.conv1 = ConvLayer(in_ch, out_ch, kernel_size, stride, pad, 
                               activation='relu', 
                               normalization=normalization)
        
        self.conv2 = ConvLayer(out_ch, out_ch, kernel_size, stride, pad, 
                               activation='linear', 
                               normalization=normalization)
        
    def forward(self, x):
        y = self.conv1(x)
        return self.conv2(y) + x
        
class DeconvLayer(nn.Module):    
    def __init__(self, in_ch, out_ch, kernel_size, stride, pad='reflect', activation='relu', normalization='batch', upsample='nearest'):        
        super(DeconvLayer, self).__init__()
        
        # upsample
        self.upsample = upsample
        
        # pad
        if pad == 'reflect':            
            self.pad = nn.ReflectionPad2d(kernel_size//2)
        elif pad == 'zero':
            self.pad = nn.ZeroPad2d(kernel_size//2)
        else:
            raise NotImplementedError("Not expected pad flag !!!")        
        
        # conv
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride)
        
        # activation
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'linear':
            self.activation = lambda x : x
        else:
            raise NotImplementedError("Not expected activation flag !!!")
        
        # normalization
        if normalization == 'instance':
            self.normalization = nn.InstanceNorm2d(out_ch, affine=True)
        if normalization == 'batch':            
            self.normalization = nn.BatchNorm2d(out_ch, affine=True)
        else:
            raise NotImplementedError("Not expected normalization flag !!!")
        
    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2, mode=self.upsample)        
        x = self.pad(x)
        x = self.conv(x)
        x = self.normalization(x)        
        x = self.activation(x)        
        return x
