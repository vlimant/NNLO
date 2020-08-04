#!/usr/bin/env python
# Rui Zhang 8.2020
# rui.zhang@cern.ch
import torch.nn as nn
import torch.nn.functional as F

def get_name():
    return 'mnist_torch'

class MNistNet(nn.Module):
    def __init__(self, **args):
        super(MNistNet, self).__init__()
        ks = int(args.get('kernel_size',5))
        do = float(args.get('dropout',0.5))
        dense = int(args.get('dense',50))
        self.conv1 = nn.Conv2d(1, 10, kernel_size=ks)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=ks)
        self.conv2_drop = nn.Dropout2d(do)
        self.fc1 = nn.Linear(320, dense)
        self.fc2 = nn.Linear(dense, 10)

    def forward(self, x):
        x = x.permute(0,3,1,2).float()
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        #return F.log_softmax(x, dim=1)
        #return F.softmax(x)
        #return F.cross_entropy(x)
        return x

def get_model(**args):
    if args:logging.debug("receiving arguments {}".format(args))
    model = MNistNet(**args)
    return model

from skopt.space import Real, Integer, Categorical
get_model.parameter_range = [
    Integer(2,10, name='kernel_size'),
    Integer(50,200, name='dense'),
    Real(0.0, 1.0, name='dropout')
]
