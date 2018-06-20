#!/usr/bin/env python3

import sys,os
import numpy as np
import argparse
import json
import time
import glob
import socket
from mpi4py import MPI

sys.path.append(os.path.dirname(os.path.realpath(__file__))+'/mpi_learn_src')
from mpi_learn.train.algo import Algo
from mpi_learn.train.data import H5Data
from mpi_learn.train.model import ModelFromJsonTF
from mpi_learn.utils import import_keras
import mpi_learn.mpi.manager as mm
from mpi_learn.train.model import ModelFromJsonTF
from mpi_learn.train.GanModel import GANBuilder
from skopt.space import Real, Integer, Categorical

class BuilderFromFunction(object):
    def __init__(self, model_fn, parameters):
        self.model_fn = model_fn
        self.parameters = parameters

    def builder(self,*params):
        args = dict(zip([p.name for p in self.parameters],params))
        model_json = self.model_fn( **args )
        return ModelFromJsonTF(None,
                               json_str=model_json)

from ga_coordinator import GACoordinator
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
    block_num, rank_in_block = divmod( rank-1, block_size)
    #block_num = int((rank-1) / block_size) + 1
    block_num+=1 ## as blocknum 0 is the skopt-master
    return block_num

def check_sanity(args):
    assert args.block_size > 1, "Block size must be at least 2 (master + worker)"

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true')

    parser.add_argument('--batch', help='batch size', default=128, type=int)
    parser.add_argument('--epochs', help='number of training epochs', default=10, type=int)
    parser.add_argument('--optimizer',help='optimizer for master to use',default='adam')
    parser.add_argument('--loss',help='loss function',default='binary_crossentropy')
    parser.add_argument('--early-stopping', type=int, 
            dest='early_stopping', help='patience for early stopping')
    parser.add_argument('--sync-every', help='how often to sync weights with master', 
            default=1, type=int, dest='sync_every')
    ############################
    ## EASGD block of option
    parser.add_argument('--easgd',help='use Elastic Averaging SGD',action='store_true')
    parser.add_argument('--worker-optimizer',help='optimizer for workers to use',
            dest='worker_optimizer', default='sgd')
    parser.add_argument('--elastic-force',help='beta parameter for EASGD',type=float,default=0.9)
    parser.add_argument('--elastic-lr',help='worker SGD learning rate for EASGD',
            type=float, default=1.0, dest='elastic_lr')
    parser.add_argument('--elastic-momentum',help='worker SGD momentum for EASGD',
            type=float, default=0, dest='elastic_momentum')
    ############################
    parser.add_argument('--block-size', type=int, default=2,
            help='number of MPI processes per block')
    parser.add_argument('--n-fold', type=int, default=1, dest='n_fold',
                        help='Number of folds used to estimate the figure of merit')
    parser.add_argument('--n-master', type=int, default=1, dest='n_master',
                        help='Number of master per group')
    parser.add_argument('--n-process', type=int, default=1, dest='n_process',
                        help='Number of process per worker instance')
    parser.add_argument('--num-iterations', type=int, default=10,
                        help='The number of steps in the skopt process')
    parser.add_argument('--example', default='mnist', choices=['topclass','mnist','gan', 'cifar10'])
    return parser


if __name__ == '__main__':

    print ("I am on",socket.gethostname())
    parser = make_parser()
    args = parser.parse_args()
    check_sanity(args)


    test = args.example
    if test == 'topclass':
        ### topclass example
        model_provider = BuilderFromFunction( model_fn = mpi.test_cnn,
                                              parameters = [ Real(0.0, 1.0, name='dropout'),
                                                             Integer(1,6, name='kernel_size'),
                                                             Real(1.,10., name = 'llr')
                                                         ]
                                          )
        if 'daint' in os.environ.get('HOST','') or 'daint' in os.environ.get('HOSTNAME',''):
            train_list = glob.glob('/scratch/snx3000/vlimant/data/LCDJets_Remake/train/*.h5')
            val_list = glob.glob('/scratch/snx3000/vlimant/data/LCDJets_Remake/val/*.h5')
        else:
            train_list = glob.glob('/bigdata/shared/LCDJets_Remake/train/04*.h5')
            val_list = glob.glob('/bigdata/shared/LCDJets_Remake/val/020*.h5')
        features_name='Images'
        labels_name='Labels'
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
        if 'daint' in os.environ.get('HOST','') or 'daint' in os.environ.get('HOSTNAME',''):
            all_list = glob.glob('/scratch/snx3000/vlimant/data/mnist/*.h5')
        else:
            all_list = glob.glob('/bigdata/shared/mnist/*.h5')
        l = int( len(all_list)*0.70)
        train_list = all_list[:l]
        val_list = all_list[l:]
        features_name='features'
        labels_name='labels'
    elif test == 'cifar10':
        ### cifar10 example
        model_provider = BuilderFromFunction( model_fn = mpi.test_cifar10,
                                              parameters = [ Integer(10,300, name='nb_filters1'),
                                                             Integer(10,300, name='nb_filters2'),
                                                             Integer(10,300, name='nb_filters3'),
                                                             Integer(50,1000, name='dense1'),
                                                             Integer(50,1000, name='dense2'),
                                                             Real(0.0, 1.0, name='dropout1'),
                                                             Real(0.0, 1.0, name='dropout2'),
                                                             Real(0.0, 1.0, name='dropout3'),
                                                             Real(0.0, 1.0, name='dropout4'),
                                                             Real(0.0, 1.0, name='dropout5')
                                                         ]
        )
        all_list = glob.glob('/nfshome/quantummind/mpi_opt/cifar10/*.h5')
        l = int( len(all_list)*0.70)
        train_list = all_list[:l]
        val_list = all_list[l:]
        features_name='features'
        labels_name='labels'
    elif test == 'gan':
        ### the gan example
        model_provider = GANBuilder( parameters = [ Integer(50,400, name='latent_size' ),
                                                    Real(0.0, 1.0, name='discr_drop_out'),
                                                    Integer(1, 8, name='gen_weight'),
                                                    Real(0.1, 10, name='aux_weight'),
                                                    Real(0.1, 10, name='ecal_weight'),
                                                ]
        )
        ## only this mode functions
        args.easgd = True
        args.worker_optimizer = 'rmsprop'
        if 'daint' in os.environ.get('HOST','') or 'daint' in os.environ.get('HOSTNAME',''):
            all_list = glob.glob('/scratch/snx3000/vlimant/data/3DGAN/*.h5')
        else:
            all_list = glob.glob('/data/shared/3DGAN/*.h5')

        l = int( len(all_list)*0.70)
        train_list = all_list[:l]
        val_list = all_list[l:]
        features_name='X'
        labels_name='y'
        
    print (len(train_list),"train files",len(val_list),"validation files")
    print("Initializing...")
    comm_world = MPI.COMM_WORLD.Dup()
    ## consistency check to make sure everything is appropriate
    num_blocks, left_over = divmod( (comm_world.Get_size()-1), args.block_size)
    if left_over:
        print ("The last block is going to be made of {} nodes, make inconsistent block size {}".format( left_over,
                                                                                                         args.block_size))
        num_blocks += 1 ## to accoun for the last block
        if left_over<2:
            print ("The last block is going to be too small for mpi_learn, with no workers")
        sys.exit(1)


    block_num = get_block_num(comm_world, args.block_size)
    device = mm.get_device(comm_world, num_blocks)
    backend = 'tensorflow'
    import keras.backend as K
    hide_device = True
    if hide_device:
        os.environ['CUDA_VISIBLE_DEVICES'] = device[-1] if 'gpu' in device else ''
        print ('set to device',os.environ['CUDA_VISIBLE_DEVICES'])
    gpu_options=K.tf.GPUOptions(
        per_process_gpu_memory_fraction=0.1,
        allow_growth = True,
        visible_device_list = device[-1] if 'gpu' in device else '')
    if hide_device:
        gpu_options=K.tf.GPUOptions(
                            per_process_gpu_memory_fraction=0.0,
            allow_growth = True,)        
    K.set_session( K.tf.Session( config=K.tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False,
        gpu_options=gpu_options
    ) ) )    
    print("Process {} using device {}".format(comm_world.Get_rank(), device))
    comm_block = comm_world.Split(block_num)
    print ("Process {} sees {} blocks, has block number {}, and rank {} in that block".format(comm_world.Get_rank(),
                                                                                              num_blocks,
                                                                                              block_num,
                                                                                              comm_block.Get_rank()
                                                                                            ))
    ## you need to sync every one up here
    #all_block_nums = comm_world.allgather( block_num )
    #print ("we gathered all these blocks {}".format( all_block_nums ))

    if args.n_process>1:
        t_b_processes= []
        if block_num !=0:
            _,_, b_processes = mm.get_groups(comm_block, args.n_master, args.n_process)
            ## collect all block=>world rank translation
            r2r = (comm_block.Get_rank() , comm_world.Get_rank())
            all_r2r = comm_block.allgather( r2r )
            translate = dict( all_r2r ) #key is the rank in block, value is rank in world
            t_b_processes = []
            for pr in b_processes:
                t_pr = []
                for p in pr:
                    t_pr.append( translate[p])
                t_b_processes.append( t_pr )
            #print ("translate process ranks from ",b_processes,"to",t_b_processes)
        
        #need to collect all the processes lists
        all_t_b_processes = comm_world.allgather( t_b_processes )
        w_processes = set()
        for gb in all_t_b_processes:
            if gb:
                hgb = map(tuple, gb)
                w_processes.update( hgb )
        if block_num == 0:
            print ("all collect processes",w_processes)
            ## now you have the ranks that needs to be initialized in rings.
        
    # MPI process 0 coordinates the Bayesian optimization procedure
    if block_num == 0:
        opt_coordinator = GACoordinator(comm_world, num_blocks,
                                                  model_provider.parameters)
        opt_coordinator.run(num_iterations=args.num_iterations)
    else:
        print ("Process {} on block {}, rank {}, create a process block".format( comm_world.Get_rank(),
                                                                                 block_num,
                                                                                 comm_block.Get_rank()))
        data = H5Data(batch_size=args.batch, 
                      features_name=features_name,
                      labels_name=labels_name
        )
        data.set_file_names( train_list )
        validate_every = data.count_data()/args.batch 
        print (data.count_data(),"samples to train on")
        if args.easgd:
            algo = Algo(None, loss=args.loss, validate_every=validate_every,
                        mode='easgd', sync_every=args.sync_every,
                        worker_optimizer=args.worker_optimizer,
                        elastic_force=args.elastic_force/(comm_block.Get_size()-1),
                        elastic_lr=args.elastic_lr, 
                        elastic_momentum=args.elastic_momentum) 
        else:
            algo = Algo(args.optimizer, 
                        loss=args.loss, 
                        validate_every=validate_every,
                        sync_every=args.sync_every,
                        worker_optimizer=args.worker_optimizer
                    )
 
        os.environ['KERAS_BACKEND'] = backend
        import_keras()
        import keras.callbacks as cbks
        callbacks = []
        if args.early_stopping is not None:
            callbacks.append( cbks.EarlyStopping( patience=args.early_stopping,
                verbose=1 ) )
        block = process_block.ProcessBlock(comm_world, comm_block, algo, data, device,
                                           model_provider,
                                           args.epochs, train_list, val_list, 
                                           folds = args.n_fold,
                                           num_masters = args.n_master,
                                           num_process = args.n_process,
                                           callbacks=callbacks, verbose=args.verbose)
        block.run()
