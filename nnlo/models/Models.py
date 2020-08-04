### Predefined Keras models

import sys
import logging
from nnlo.util.utils import import_keras
import_keras()

def model_function(model_name):
    """Constructs the Keras model indicated by model_name"""
    model_maker_dict = {
    #        'example':make_example_model,
    #        'mnist':make_mnist_model,
    #        'cifar10':make_cifar10_model,
    #        'mnist_torch':make_mnist_torch_model,
    #        'topclass': make_topclass_model,
    #        'topclass_torch':make_topclass_torch_model
        
            }
    return model_maker_dict[model_name]    
def make_model(model_name, **args):
    m_fn = model_function(model_name)
    if args and hasattr(m_fn,'parameter_range'):
        provided = set(args.keys())
        accepted = set([a.name for a in m_fn.parameter_range])
        if not provided.issubset( accepted ):
            logging.error("provided arguments {} do not match the accepted ones {}".format(sorted(provided),sorted(accepted)))
            sys.exit(-1)
    return model_function(model_name)(**args)

