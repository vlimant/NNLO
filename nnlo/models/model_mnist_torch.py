#!/usr/bin/env python
# Rui Zhang 8.2020
# rui.zhang@cern.ch

def get_name():
    return 'mnist_torch'

def get_model(**args):
    if args:logging.debug("receiving arguments {}".format(args))
    try:
        from TorchModels import MNistNet
    except:
        from .TorchModels import MNistNet
    model = MNistNet(**args)
    return model

from skopt.space import Real, Integer, Categorical
get_model.parameter_range = [
    Integer(2,10, name='kernel_size'),
    Integer(50,200, name='dense'),
    Real(0.0, 1.0, name='dropout')
]
