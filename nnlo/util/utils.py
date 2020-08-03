### Utilities for mpi_learn module
import os
import sys
import numpy as np
import logging

class Error(Exception):
    pass

def weights_from_shapes(weights_shapes):
    """Returns a list of numpy arrays representing the NN architecture"""
    return [ np.zeros( shape, dtype=np.float32 ) for shape in weights_shapes ]

def shapes_from_weights(weights):
    """Returns a list of tuples indicating the array shape of each layer of the NN"""
    return [ w.shape for w in weights ]

def opt_tag_lookup(tag):
    """
    Gets the integer corresponding to the given tag string
    """
    tags = {
            'json':1,
            'result':2,
            'mbuilder':3,
            'params':4
            }
    return tags.get(tag, 0)

def import_keras(tries=10):
    """There is an issue when multiple processes import Keras simultaneously --
        the file .keras/keras.json is sometimes not read correctly.  
        as a workaround, just try several times to import keras."""
    for try_num in range(tries):
        try:
            stderr = sys.stderr
            sys.stderr = open(os.devnull, 'w')
            import tensorflow.keras as keras
            sys.stderr = stderr
            return
        except ValueError:
            logging.warning("Unable to import keras. Trying again: {0:d}".format(try_num))
            from time import sleep
            sleep(0.1)
    logging.error("Failed to import keras!")

def load_model(filename=None, model=None, weights_file=None, custom_objects={}):
    """Loads model architecture from JSON and instantiates the model.
        filename: path to JSON file specifying model architecture
        model:    (or) a Keras model to be cloned
        weights_file: path to HDF5 file containing model weights
	custom_objects: A Dictionary of custom classes used in the model keyed by name"""
    import_keras()
    from tensorflow.keras.models import model_from_json, clone_model
    if filename is not None:
        with open( filename ) as arch_f:
            json_str = arch_f.readline()
            new_model = model_from_json( json_str, custom_objects=custom_objects) 
        logging.info(f"Load model from filename")
    elif model is not None:
        new_model = clone_model(model)
        logging.info(f"Load model from model")
    elif weights_file is not None:
        new_model.load_weights( weights_file )
        logging.info(f"Load model from weights_file")
    else:
        logging.error(f"Cannot load model: filename, model and weights_file are None")
    return new_model

