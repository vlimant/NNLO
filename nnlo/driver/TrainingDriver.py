#!/usr/bin/env python3

### This script creates an MPIManager object and launches distributed training.

import sys,os
import numpy as np
import argparse
import json
import re
import logging
import glob

from mpi4py import MPI
from time import time,sleep
import importlib

from nnlo.mpi.manager import MPIManager, get_device
from nnlo.train.algo import Algo
from nnlo.train.data import H5Data
from nnlo.train.model import ModelFromJson, ModelTensorFlow, ModelPytorch
from nnlo.util.utils import import_keras
from nnlo.util.timeline import Timeline
from nnlo.util.logger import initialize_logger

def make_Block_Parser():
    pass
def add_log_option(parser):
    # logging configuration
    parser.add_argument('--log-file', default=None, dest='log_file', help='log file to write, in additon to output stream')
    parser.add_argument('--log-level', default='info', dest='log_level', help='log level (debug, info, warn, error)')
    parser.add_argument('--output', default='./', dest='output', help='output folder')

def add_master_option(parser):
    parser.add_argument('--master-gpu',help='master process should get a gpu',
            action='store_true', dest='master_gpu')
    parser.add_argument('--synchronous',help='run in synchronous mode',action='store_true')
    
def add_worker_options(parser):
    parser.add_argument('--worker-optimizer',help='optimizer for workers to use',
            dest='worker_optimizer', default='adam')
    parser.add_argument('--worker-optimizer-params',help='worker optimizer parameters (string representation of a dict)',
            dest='worker_optimizer_params', default='{}')
    
    
def add_gem_options(parser):
    parser.add_argument('--gem-lr',help='learning rate for GEM',type=float,default=0.01, dest='gem_lr')
    parser.add_argument('--gem-momentum',help='momentum for GEM',type=float, default=0.9, dest='gem_momentum')
    parser.add_argument('--gem-kappa',help='Proxy amplification parameter for GEM',type=float, default=2.0, dest='gem_kappa')    

def add_easgd_options(parser):
    parser.add_argument('--elastic-force',help='beta parameter for EASGD',type=float,default=0.9)
    parser.add_argument('--elastic-lr',help='worker SGD learning rate for EASGD',
            type=float, default=1.0, dest='elastic_lr')
    parser.add_argument('--elastic-momentum',help='worker SGD momentum for EASGD',
            type=float, default=0, dest='elastic_momentum')

def add_downpour_options(parser):
    parser.add_argument('--optimizer',help='optimizer for master to use in downpour',default='adam')


def add_loader_options(parser):
    parser.add_argument('--preload-data', help='Preload files as we read them', default=0, type=int, dest='data_preload')
    parser.add_argument('--cache-data', help='Cache the input files to a provided directory', default='', dest='caching_dir')
    parser.add_argument('--copy-command', help='Specific command line to copy the data into the cache. Expect a string with two {} first is the source (from input file list), second is the bare file name at destination. Like "cp {} {}"', default=None, dest='copy_command')


def add_target_options(parser):
    parser.add_argument('--early-stopping', default=None,
                        dest='early_stopping', help='patience for early stopping')
    parser.add_argument('--target-metric', default=None,
                        dest='target_metric', help='Passing configuration for a target metric')
    

def make_train_parser():
    parser = argparse.ArgumentParser()    
    parser.add_argument('--timeline',help='Record timeline of activity', action='store_true')
    add_train_options(parser)

    return parser

def add_checkpoint_options(parser):
    parser.add_argument('--restore', help='pass a file to retore the variables from', default=None)
    parser.add_argument('--checkpoint', help='Base name of the checkpointing file. If omitted no checkpointing will be done', default=None)
    parser.add_argument('--checkpoint-interval', help='Number of epochs between checkpoints', default=5, type=int, dest='checkpoint_interval')
    
def add_train_options(parser):
    parser.add_argument('--verbose',help='display metrics for each training batch',action='store_true')
    parser.add_argument('--monitor',help='Monitor cpu and gpu utilization', action='store_true')

    parser.add_argument('--backend', help='specify the backend to be used', choices= ['keras','torch'],default='keras')
    parser.add_argument('--thread_validation', help='run a single process', action='store_true')
    
    # model arguments
    parser.add_argument('--model', choices=['mnist', 'mnist_torch', 'cifar10', 'cifar10_torch'], help='File containing model architecture (serialized in JSON/pickle, or provided in a .py file')
    parser.add_argument('--trial-name', help='descriptive name for trial', 
            default='train', dest='trial_name')

    # training data arguments
    parser.add_argument('--train_data', help='text file listing data inputs for training', default=None)
    parser.add_argument('--val_data', help='text file lis`ting data inputs for validation', default=None)
    parser.add_argument('--features-name', help='name of HDF5 dataset with input features',
            default='features', dest='features_name')
    parser.add_argument('--labels-name', help='name of HDF5 dataset with output labels',
            default='labels', dest='labels_name')
    
    parser.add_argument('--batch', help='batch size', default=100, type=int)

    

    # configuration of network topology
    parser.add_argument('--n-masters', dest='n_masters', help='number of master processes', default=1, type=int)
    parser.add_argument('--n-processes', dest='n_processes', help='number of processes per worker', default=1, type=int)
    parser.add_argument('--max-gpus', dest='max_gpus', help='max GPUs to use', type=int, default=1)


    # configuration of training process
    parser.add_argument('--epochs', help='number of training epochs', default=1, type=int)

    parser.add_argument('--loss',help='loss function',default='binary_crossentropy')

    add_target_options(parser)

    add_worker_options(parser)

    parser.add_argument('--sync-every', help='how often to sync weights with master', 
            default=1, type=int, dest='sync_every')
    parser.add_argument('--mode',help='Mode of operation.'
                        'One of "downpour" (Downpour), "easgd" (Elastic Averaging SGD) or "gem" (Gradient Energy Matching)',default='gem',choices=['downpour','easgd','gem'])

    add_master_option(parser)
    add_gem_options(parser)
    add_easgd_options(parser)
    add_downpour_options(parser)
    
    add_loader_options(parser)
    
    add_log_option(parser)
    add_checkpoint_options(parser)

def make_loader( args, features_name, labels_name, train_list):
    data = H5Data( batch_size=args.batch,
                   cache = args.caching_dir,
                   copy_command = args.copy_command,                   
                   preloading = args.data_preload,
                   features_name=features_name,
                   labels_name=labels_name,
    )
    # We initialize the Data object with the training data list
    # so that we can use it to count the number of training examples
    data.set_full_file_names( train_list )
    
    return data

def make_model_weight(args, use_torch):
    model_weights = None
    if args.restore:
        args.restore = re.sub(r'\.algo$', '', args.restore)
        if os.path.isfile(args.restore + '.latest'):
            with open(args.restore + '.latest', 'r') as latest:
                args.restore = latest.read().splitlines()[-1]
        if any([os.path.isfile(ff) for ff in glob.glob('./*'+args.restore + '.model')]):
            if use_torch:
                args.model = args.restore + '.model'
                model_weights = args.restore +'.model_w'
            else:
                model_weights = args.restore + '.model'
                
    return model_weights
                        
def make_algo( args, use_tf, comm, validate_every ):
    args_opt = args.optimizer
    if use_tf:
        if not args_opt.endswith("tf"):
            args_opt = args_opt + 'tf'
    else:
        if not args_opt.endswith("torch"):
            args_opt = args_opt + 'torch'
            
    if args.mode == 'easgd':
        algo = Algo(None, loss=args.loss, validate_every=validate_every,
                mode='easgd', sync_every=args.sync_every,
                worker_optimizer=args.worker_optimizer,
                worker_optimizer_params=args.worker_optimizer_params,
                elastic_force=args.elastic_force/(max(1,comm.Get_size()-1)),
                elastic_lr=args.elastic_lr, 
                elastic_momentum=args.elastic_momentum) 
    elif args.mode == 'gem':
        algo = Algo('gem', loss=args.loss, validate_every=validate_every,
                mode='gem', sync_every=args.sync_every,
                worker_optimizer=args.worker_optimizer,
                worker_optimizer_params=args.worker_optimizer_params,
                learning_rate=args.gem_lr, momentum=args.gem_momentum, kappa=args.gem_kappa)
    elif args.mode == 'downpour':
        algo = Algo(args_opt, loss=args.loss, validate_every=validate_every,
                sync_every=args.sync_every, worker_optimizer=args.worker_optimizer,
                worker_optimizer_params=args.worker_optimizer_params)
    else:
        logging.info("%s not supported mode", args.mode)
    return algo

def make_train_val_lists(m_module, args):
    train_list = val_list = []
    if args.train_data:
        with open(args.train_data) as train_list_file:
            train_list = [ s.strip() for s in train_list_file.readlines() ]
    elif m_module is not None:
        train_list = m_module.get_train()
    else:
        logging.info("no training data provided")
        
    if args.val_data:
        with open(args.val_data) as val_list_file:
            val_list = [ s.strip() for s in val_list_file.readlines() ]
    elif m_module is not None:
        val_list = m_module.get_val()
    else:
        logging.info("no validation data provided")

    if not train_list:
        logging.error("No training data provided")
    if not val_list:
        logging.error("No validation data provided")
    return (train_list, val_list) 

def make_features_labels(m_module, args):
    features_name = m_module.get_features() if m_module is not None and hasattr(m_module,"get_features") else args.features_name
    labels_name = m_module.get_labels() if m_module is not None and hasattr(m_module,"get_labels") else args.labels_name
    return (features_name, labels_name)

def main():
    parser = make_train_parser()
    args = parser.parse_args()    
    initialize_logger(filename=args.log_file, file_level=args.log_level, stream_level=args.log_level)

    a_backend = args.backend
    if 'torch' in args.model:
        a_backend = 'torch'
        
    m_module, model_source = None, None
    try:
        if args.model == 'mnist':
            m_module = importlib.import_module(f'nnlo.models.model_mnist_tf')
            model_source = 'models/model_mnist_tf.py'
        elif args.model == 'mnist_torch':
            m_module = importlib.import_module(f'nnlo.models.model_mnist_torch')
            model_source = 'models/model_mnist_torch.py'
        elif args.model == 'cifar10':
            m_module = importlib.import_module(f'nnlo.models.model_cifar10_tf')
            model_source = 'models/model_cifar10_tf.py'
    except Exception as e:
        logging.fatal(e)

    (features_name, labels_name) = make_features_labels(m_module, args)
    (train_list, val_list) = make_train_val_lists(m_module, args)
    comm = MPI.COMM_WORLD.Dup()

    if args.timeline: Timeline.enable()

    use_tf = a_backend == 'keras'
    use_torch = not use_tf

    model_weights = make_model_weight(args, use_torch)

    # Theano is the default backend; use tensorflow if --tf is specified.
    # In the theano case it is necessary to specify the device before importing.
    device = get_device( comm, args.n_masters, gpu_limit=args.max_gpus,
                gpu_for_master=args.master_gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = device[-1] if 'gpu' in device else ''
    logging.debug('set to device %s',os.environ['CUDA_VISIBLE_DEVICES'])

    if use_torch:
        logging.debug("Using pytorch")
        model_builder = ModelPytorch(comm, source=model_source, weights=model_weights, gpus=1 if 'gpu' in device else 0)
    else:
        logging.debug("Using TensorFlow")
        os.environ['KERAS_BACKEND'] = 'tensorflow'

        import tensorflow as tf
        import_keras()
        #tf.config.gpu.set_per_process_memory_fraction(0.1)
        #gpu_options=K.tf.GPUOptions(
        #    per_process_gpu_memory_fraction=0.1, #was 0.0
        #    allow_growth = True,
        #    visible_device_list = device[-1] if 'gpu' in device else '')
        #gpu_options=K.tf.GPUOptions(
        #    per_process_gpu_memory_fraction=0.0,
        #    allow_growth = True,)     
        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        for device in gpu_devices:
            tf.config.experimental.set_memory_growth(device, True)

        #NTHREADS=(2,1)
        #NTHREADS=None
        #if NTHREADS is None:
        #    K.set_session( K.tf.Session( config=K.tf.ConfigProto(
        #        allow_soft_placement=True, log_device_placement=False,
        #        gpu_options=gpu_options
        #    ) ) )
        #else:
        #    K.set_session( K.tf.Session( config=K.tf.ConfigProto(
        #        allow_soft_placement=True, log_device_placement=False,
        #        gpu_options=gpu_options,
        #        intra_op_parallelism_threads=NTHREADS[0], 
        #        inter_op_parallelism_threads=NTHREADS[1],
        #    ) ) )
        

        model_builder = ModelTensorFlow( comm, source=model_source, weights=model_weights)


    data = make_loader(args, features_name, labels_name, train_list)

    # Some input arguments may be ignored depending on chosen algorithm
    algo = make_algo( args, use_tf, comm, validate_every=int(data.count_data()/args.batch ))
    
    if args.restore:
        algo.load(args.restore)

    # Creating the MPIManager object causes all needed worker and master nodes to be created
    manager = MPIManager( comm=comm, data=data, algo=algo, model_builder=model_builder,
                          num_epochs=args.epochs, train_list=train_list, val_list=val_list, 
                          num_masters=args.n_masters, num_processes=args.n_processes,
                          synchronous=args.synchronous, 
                          verbose=args.verbose, monitor=args.monitor,
                          early_stopping=args.early_stopping,
                          target_metric=args.target_metric,
                          thread_validation = args.thread_validation,
                          checkpoint=args.checkpoint, checkpoint_interval=args.checkpoint_interval)


    if m_module:
        model_name =m_module.get_name()
    else:
        model_name = os.path.basename(args.model).replace('.json','')

    json_name = args.output + '/' + '_'.join([model_name,args.trial_name,"history.json"])
    tl_json_name = args.output + '/' + '_'.join([model_name,args.trial_name,"timeline.json"])

    # Process 0 launches the training procedure
    if comm.Get_rank() == 0:
        logging.debug('Training configuration: %s', algo.get_config())

        t_0 = time()
        histories = manager.process.train() 
        delta_t = time() - t_0
        logging.info("Training finished in {0:.3f} seconds".format(delta_t))

        manager.process.record_details(json_name,
                                       meta={"args":vars(args)})            
        logging.info("Wrote trial information to {0}".format(json_name))
        manager.close()

    comm.barrier()
    logging.info("Terminating")
    if args.timeline: Timeline.collect(clean=True, file_name=tl_json_name)

if __name__ == '__main__':
    main()
