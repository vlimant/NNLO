#!/usr/bin/env python
# Rui Zhang 8.2020
# rui.zhang@cern.ch

def get_name():
    return 'topclass_torch'

def get_model(**args):
    if args:logging.debug("receiving arguments {}".format(args))
    conv_layers=args.get('conv_layers',2)
    dense_layers=args.get('dense_layers',2)
    dropout=args.get('dropout',0.5)
    classes=3
    in_channels=5
    try:
        from TorchModels import CNN
    except:
        from .TorchModels import CNN
    model = CNN(conv_layers=conv_layers, dense_layers=dense_layers, dropout=dropout, classes=classes, in_channels=in_channels)
    return model

from skopt.space import Real, Integer, Categorical
get_model.parameter_range =    [
    Integer(1,6, name='conv_layers'),
    Integer(1,6, name='dense_layers'),
    Real(0.0,1.0, name='dropout')
]
