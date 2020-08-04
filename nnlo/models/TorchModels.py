import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F
import numpy
import logging


### Build a customized CNN with given hyperparameters
class _ConvBlock(nn.Sequential):
    def __init__(self, conv_layers, dropout, in_ch=5):
        super().__init__()
        for i in range(conv_layers):
            channel_in = in_ch*((i+1)%5)
            channel_out = in_ch*((i+2)%5)
            if channel_in == 0: channel_in += 1
            if channel_out == 0: channel_out += 1
            self.add_module('convlayer%d'%(i), nn.Conv2d(channel_in, out_channels=channel_out,kernel_size=(3,3),stride=1, padding=1))
            self.add_module('relu%d'%(i), nn.ReLU(inplace=True))
        self.add_module('convlayer%d'%(conv_layers), nn.Conv2d(channel_out, out_channels=1, kernel_size=(3,3), stride=2, padding=1))
        self.dropout = dropout

    def forward(self, x):
        x = super().forward(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class _DenseBlock(nn.Sequential):
    def __init__(self, dense_layers, dropout ,base):
        super().__init__()
        for i in range(dense_layers):
            il = int(base//(2**i))
            ol = int(base//(2**(i+1)))
            logging.info("{} =>> {}".format(il,ol))
            self.add_module('denselayer%d'%(i), nn.Linear(il, ol))
            self.add_module('relu%d'%(i), nn.ReLU(inplace=True))
        self.dropout = dropout

    def forward(self, x):
        x = super().forward(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class CNN(nn.Module):
    def __init__(self, conv_layers=2, dense_layers=2, dropout=0.5, classes=3, in_channels=5):
        super().__init__()
        self.build_net(conv_layers, dense_layers, dropout, classes, in_channels)
    
    def build_net(self,*args, **kwargs):
        base_2 = 10
        base = base_2**2
        self.conv_layers = _ConvBlock(args[0], args[2], args[4])
        self.dense_layers = _DenseBlock(args[1], args[2], base)
        self.adapt_pool = nn.AdaptiveMaxPool2d((base_2,base_2))
        il = int(base//(2**(args[1])))
        ol = int(args[3])
        logging.info("{} =>> {}".format(il,ol))
        self.output = nn.Linear(il, ol)

    def forward(self, x):
        x = x.permute(0,3,1,2).float()        
        x = self.conv_layers(x)
        x = self.adapt_pool(x)
        x = x.view(x.shape[0], -1) # flatten
        x = self.dense_layers(x)
        return self.output(x)
            

