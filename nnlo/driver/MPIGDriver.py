#!/usr/bin/env python

### This script creates an MPIManager object and launches distributed training.

import sys,os
import numpy as np
import argparse
import json
import re
import logging

from mpi4py import MPI
from time import time,sleep

from nnlo.mpi.manager import MPIManager, get_device
from nnlo.train.algo import Algo
from nnlo.train.data import H5Data
from nnlo.train.model import ModelFromJson, ModelTensorFlow
from nnlo.util.utils import import_keras
from nnlo.util.logger import initialize_logger
import socket


def main():
    from TrainingDriver import add_loader_options
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose',help='display metrics for each training batch',action='store_true')
    parser.add_argument('--profile',help='profile theano code',action='store_true')
    parser.add_argument('--monitor',help='Monitor cpu and gpu utilization', action='store_true')
    parser.add_argument('--tf', help='use tensorflow backend', action='store_true')

    # model arguments
    parser.add_argument('model_json', help='JSON file containing model architecture')
    parser.add_argument('--trial-name', help='descriptive name for trial', 
            default='train', dest='trial_name')

    # training data arguments
    parser.add_argument('train_data', help='text file listing data inputs for training')
    parser.add_argument('val_data', help='text file listing data inputs for validation')
    parser.add_argument('--features-name', help='name of HDF5 dataset with input features',
            default='features', dest='features_name')
    parser.add_argument('--labels-name', help='name of HDF5 dataset with output labels',
            default='labels', dest='labels_name')
    parser.add_argument('--batch', help='batch size', default=100, type=int)
    add_loader_options(parser)

    # configuration of network topology
    parser.add_argument('--masters', help='number of master processes', default=1, type=int)
    parser.add_argument('--n-processes', dest='processes', help='number of processes per worker', default=1, type=int)
    parser.add_argument('--max-gpus', dest='max_gpus', help='max GPUs to use', 
            type=int, default=-1)
    parser.add_argument('--master-gpu',help='master process should get a gpu',
            action='store_true', dest='master_gpu')
    parser.add_argument('--synchronous',help='run in synchronous mode',action='store_true')

    # configuration of training process
    parser.add_argument('--epochs', help='number of training epochs', default=1, type=int)
    parser.add_argument('--optimizer',help='optimizer for master to use',default='adam')
    parser.add_argument('--loss',help='loss function',default='binary_crossentropy')
    parser.add_argument('--early-stopping', default=None,
                        dest='early_stopping', help='Configuration for early stopping')
    parser.add_argument('--target-metric', default=None,
                        dest='target_metric', help='Passing configuration for a target metric')
    parser.add_argument('--worker-optimizer',help='optimizer for workers to use',
            dest='worker_optimizer', default='sgd')
    parser.add_argument('--worker-optimizer-params',help='worker optimizer parameters (string representation of a dict)',
            dest='worker_optimizer_params', default='{}')
    parser.add_argument('--sync-every', help='how often to sync weights with master', 
            default=1, type=int, dest='sync_every')
    parser.add_argument('--mode',help='Mode of operation.'
                        'One of "downpour" (Downpour), "easgd" (Elastic Averaging SGD) or "gem" (Gradient Energy Matching)',default='downpour',choices=['downpour','easgd','gem'])

    parser.add_argument('--elastic-force',help='beta parameter for EASGD',type=float,default=0.9)
    parser.add_argument('--elastic-lr',help='worker SGD learning rate for EASGD',
            type=float, default=1.0, dest='elastic_lr')
    parser.add_argument('--elastic-momentum',help='worker SGD momentum for EASGD',
            type=float, default=0, dest='elastic_momentum')
    parser.add_argument('--restore', help='pass a file to retore the variables from', default=None)
    parser.add_argument('--log-file', default=None, dest='log_file', help='log file to write, in additon to output stream')
    parser.add_argument('--log-level', default='info', dest='log_level', help='log level (debug, info, warn, error)')

    parser.add_argument('--checkpoint', help='Base name of the checkpointing file. If omitted no checkpointing will be done', default=None)
    parser.add_argument('--checkpoint-interval', help='Number of epochs between checkpoints', default=5, type=int, dest='checkpoint_interval')
    

    args = parser.parse_args()
    model_name = os.path.basename(args.model_json).replace('.json','')
    initialize_logger(filename=args.log_file, file_level=args.log_level, stream_level=args.log_level)    

    with open(args.train_data) as train_list_file:
        train_list = [ s.strip() for s in train_list_file.readlines() ]
    with open(args.val_data) as val_list_file:
        val_list = [ s.strip() for s in val_list_file.readlines() ]

    comm = MPI.COMM_WORLD.Dup()

    use_tf = args.tf
    use_torch = not use_tf
    
    from TrainingDriver import make_model_weight, make_algo, make_loader
    model_weights = make_model_weight(args, use_torch)

    device = get_device( comm, args.masters, gpu_limit=args.max_gpus,
                gpu_for_master=args.master_gpu)
    if use_tf:
        backend = 'tensorflow'
        if not args.optimizer.endswith("tf"):
            args.optimizer = args.optimizer + 'tf'
        os.environ['CUDA_VISIBLE_DEVICES'] = device[-1] if 'gpu' in device else ''
        logging.info('set to device %s %s'%(os.environ['CUDA_VISIBLE_DEVICES'], socket.gethostname()))
    os.environ['KERAS_BACKEND'] = backend

    logging.info(backend)
    if use_tf:
        import_keras()
        import keras.backend as K
        gpu_options=K.tf.GPUOptions(
            per_process_gpu_memory_fraction=0.0,
            allow_growth = True,)
        K.set_session( K.tf.Session( config=K.tf.ConfigProto(
            allow_soft_placement=True,
            #allow_soft_placement=False,
            #log_device_placement=True , # was false
            log_device_placement=False , # was false
            gpu_options=gpu_options
            ) ) )

    if use_tf:
        from nnlo.train.GanModel import GANModelBuilder
        model_builder  = GANModelBuilder( comm , tf= True, weights=model_weights)


    data = make_loader(args, args.features_name, args.labels_name, train_list)
    algo = make_algo( args, use_tf, comm, validate_every=int(data.count_data()/args.batch ))

    if args.restore:
        algo.load(args.restore)

    # Creating the MPIManager object causes all needed worker and master nodes to be created
    manager = MPIManager( comm=comm, data=data, algo=algo, model_builder=model_builder,
                          num_epochs=args.epochs, train_list=train_list, val_list=val_list, 
                          num_masters=args.masters, num_processes=args.processes,
                          synchronous=args.synchronous, 
                          verbose=args.verbose , monitor=args.monitor,
                          early_stopping=args.early_stopping,target_metric=args.target_metric ,
                          checkpoint=args.checkpoint, checkpoint_interval=args.checkpoint_interval)

    # Process 0 launches the training procedure
    if comm.Get_rank() == 0:
        logging.info(algo)

        t_0 = time()
        histories = manager.process.train() 
        delta_t = time() - t_0
        manager.free_comms()
        logging.info("Training finished in {0:.3f} seconds".format(delta_t))

        json_name = '_'.join([model_name,args.trial_name,"history.json"]) 
        manager.process.record_details(json_name,
                                       meta={"args":vars(args)})
        logging.info("Wrote trial information to {0}".format(json_name))

    comm.Barrier()
    logging.info("Terminating")

if __name__ == '__main__':
    main()
