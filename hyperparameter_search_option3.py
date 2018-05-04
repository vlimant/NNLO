#!/usr/bin/env python3

import sys,os
import numpy as np
import argparse
import json
import time
import glob
from mpi4py import MPI

sys.path.append(os.path.dirname(os.path.realpath(__file__))+'/mpi_learn_src')
from mpi_learn.train.algo import Algo
from mpi_learn.train.data import H5Data
from mpi_learn.train.model import ModelFromJsonTF
from mpi_learn.utils import import_keras
import mpi_learn.mpi.manager as mm
from mpi_learn.train.model import ModelFromJsonTF
from skopt.space import Real, Integer

class BuilderFromFunction(object):
    def __init__(self, model_fn, parameters):
        self.model_fn = model_fn
        self.parameters = parameters

    def builder(self,*params):
        args = dict(zip([p.name for p in self.parameters],params))
        model_json = self.model_fn( **args )
        return ModelFromJsonTF(None,
                               json_str=model_json)

import coordinator
import process_block
import mpiLAPI as mpi

def get_block_num(comm, block_size):
    """
    Gets the correct block number for this process.
    The coordinator (process 0) is in block 999.
    The other processes are divided according to the block size.
    """
    rank = comm.Get_rank()
    if rank == 0:
        return 0
    block_num = int((rank-1) / block_size) + 1
    return block_num

def check_sanity(args):
    assert args.block_size > 1, "Block size must be at least 2 (master + worker)"

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true')

    parser.add_argument('--batch', help='batch size', default=100, type=int)
    parser.add_argument('--epochs', help='number of training epochs', default=1, type=int)
    parser.add_argument('--optimizer',help='optimizer for master to use',default='adam')
    parser.add_argument('--loss',help='loss function',default='binary_crossentropy')
    parser.add_argument('--early-stopping', type=int, 
            dest='early_stopping', help='patience for early stopping')
    parser.add_argument('--sync-every', help='how often to sync weights with master', 
            default=1, type=int, dest='sync_every')

    parser.add_argument('--block-size', type=int, default=2,
            help='number of MPI processes per block')
    return parser


if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
    check_sanity(args)


    test = 'mnist'
    if test == 'topclass':
        ### topclass example
        model_provider = BuilderFromFunction( model_fn = mpi.test_cnn,
                                              parameters = [ Real(0.0, 1.0, name='dropout'),
                                                             Integer(1,6, name='kernel_size'),
                                                             Real(1.,10., name = 'llr')
                                                         ]
                                          )
        #train_list = glob.glob('/bigdata/shared/LCDJets_Remake/train/04*.h5')
        #val_list = glob.glob('/bigdata/shared/LCDJets_Remake/val/020*.h5')
        train_list = glob.glob('/scratch/snx3000/vlimant/data/LCDJets_Remake/train/04*.h5')
        val_list = glob.glob('/scratch/snx3000/vlimant/data/LCDJets_Remake/val/020*.h5')

    elif test == 'mnist':
        ### mnist example
        model_provider = BuilderFromFunction( model_fn = mpi.test_mnist,
                                              parameters = [ Integer(10,50, name='nb_filters'),
                                                             Integer(2,10, name='pool_size'),
                                                             Integer(2,10, name='kernel_size'),
                                                             Integer(50,200, name='dense'),
                                                             Real(0.0, 1.0, name='dropout')
                                                         ]
                                          )
        all_list = glob.glob('/scratch/snx3000/vlimant/data/mnist/*.h5')
        train_list = all_list[:-10]
        val_list = all_list[-10:]

    print("Initializing...")
    comm_world = MPI.COMM_WORLD.Dup()
    num_blocks = int(comm_world.Get_size()/args.block_size)
    block_num = get_block_num(comm_world, args.block_size)
    device = mm.get_device(comm_world, num_blocks)
    backend = 'tensorflow'
    print("Process {} using device {}".format(comm_world.Get_rank(), device))
    comm_block = comm_world.Split(block_num)

    # MPI process 0 coordinates the Bayesian optimization procedure
    if block_num == 0:
        opt_coordinator = coordinator.Coordinator(comm_world, num_blocks,
                                                  model_provider.parameters)
        opt_coordinator.run(num_iterations=30)
    else:
        data = H5Data(batch_size=args.batch, 
                features_name='Images', labels_name='Labels')
        data.set_file_names( train_list )
        validate_every = data.count_data()/args.batch 
        algo = Algo(args.optimizer, loss=args.loss, validate_every=validate_every,
                sync_every=args.sync_every) 
        os.environ['KERAS_BACKEND'] = backend
        import_keras()
        import keras.callbacks as cbks
        callbacks = []
        if args.early_stopping is not None:
            callbacks.append( cbks.EarlyStopping( patience=args.early_stopping,
                verbose=1 ) )
        block = process_block.ProcessBlock(comm_world, comm_block, algo, data, device,
                                           model_provider,
                                           args.epochs, train_list, val_list, callbacks, verbose=args.verbose)
        block.run()
