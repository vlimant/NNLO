#!/usr/bin/env python3

import sys,os
import numpy as np
import argparse
import time
import glob
import socket
import logging
from mpi4py import MPI

from nnlo.train.algo import Algo
from nnlo.train.data import H5Data
from nnlo.train.model import ModelTensorFlow, ModelPytorch
from nnlo.util.utils import import_keras
from nnlo.mpi.manager import get_device, get_groups
from nnlo.optimize.coordinator import Coordinator
from nnlo.optimize.process_block import ProcessBlock
from skopt.space import Real, Integer, Categorical
from nnlo.util.logger import initialize_logger

class BuilderFromFunction(object):
    def __init__(self, model_fn, parameters=None):
        self.model_fn = model_fn
        if parameters is None:
            self.parameters = model_fn.parameter_range
        else:
            self.parameters = parameters

    def _args(self,*params):
        args = dict(zip([p.name for p in self.parameters],params))
        return args
    
    def builder(self,*params):
        try:
            model = self.model_fn( **self._args(*params))
            return ModelTensorFlow(None, source=model)
        except:
            str_param = ','.join('{0}={1!r}'.format(k,v) for k,v in self._args(*params).items())
            logging.warning("Failed to build model with params: {}".format(str_param))
            return None
        
class TorchBuilderFromFunction(BuilderFromFunction):
    def __init__(self, model_fn, parameters=None, gpus=0):
        super().__init__(model_fn, parameters)
        self.gpus = gpus

    def builder(self, *params):
        args = dict(zip([p.name for p in self.parameters], params))
        try:
            model_pytorch = self.model_fn(**args)
            return ModelPytorch(None, source=model_pytorch, gpus=self.gpus)
        except:
            str_param = ','.join('{0}={1!r}'.format(k,v) for k,v in args.items())
            logging.warning("Failed to build model with params: {}".format(str_param))
            return None

import models.Models as models

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

from TrainingDriver import add_train_options
from TrainingDriver import make_loader, make_train_val_lists, make_features_labels

def make_opt_parser():
    parser = argparse.ArgumentParser()

    ############################
    parser.add_argument('--block-size', type=int, default=2,
            help='number of MPI processes per block')
    parser.add_argument('--n-fold', type=int, default=1, dest='n_fold',
                        help='Number of folds used to estimate the figure of merit')

    parser.add_argument('--num-iterations', type=int, default=10,
                        help='The number of steps in the skopt process')
    parser.add_argument('--hyper-opt', dest='hyper_opt', default='bayesian', choices=['bayesian','genetic'],
                        help='The algorithm to use for the hyper paramater optimization')
    parser.add_argument('--ga-populations', help='population size for genetic algorithm',
                        default=10, type=int, dest='population')

    parser.add_argument('--opt-restore', help='Try to resume optimisation from saved state', dest='opt_restore', action='store_true')

    parser.add_argument('--target-objective', type=float, default=None,dest='target_objective',
                        help='A value to reach and stop in the parameter optimisation')

    ## opt specific arguments
    parser.add_argument('--example', default='mnist', choices=['topclass','mnist','gan','cifar10'])

    add_train_options(parser)
    
    return parser

def main():
    logging.info("Process is on {}".format(socket.gethostname()))
    parser = make_opt_parser()
    args = parser.parse_args()
    check_sanity(args)
    initialize_logger(filename=args.log_file, file_level=args.log_level, stream_level=args.log_level)

    import socket
    host = os.environ.get('HOST',os.environ.get('HOSTNAME',socket.gethostname()))

    test = args.example
    model_source = args.model
    a_backend = args.backend
    if args.model and 'torch' in args.model:
        a_backend = 'torch'
    use_tf = a_backend == 'keras'
    use_torch = not use_tf

        ##starting the configuration of the processes
    logging.info("Initializing...")
    comm_world = MPI.COMM_WORLD.Dup()
    ## consistency check to make sure everything is appropriate
    num_blocks, left_over = divmod( (comm_world.Get_size()-1), args.block_size)
    if left_over:
        logging.warning("The last block is going to be made of {} nodes, make inconsistent block size {}".format( left_over,
                                                                                                         args.block_size))
        num_blocks += 1 ## to accoun for the last block
        if left_over<2:
            logging.warning("The last block is going to be too small for mpi_learn, with no workers")
        MPI.COMM_WORLD.Abort()

    block_num = get_block_num(comm_world, args.block_size)
    device = get_device(comm_world, num_blocks,
                        gpu_limit=args.max_gpus)
    logging.info("Process {} using device {}".format(comm_world.Get_rank(), device))

    os.environ['CUDA_VISIBLE_DEVICES'] = device[-1] if 'gpu' in device else ''
    logging.info('set to device %s',os.environ['CUDA_VISIBLE_DEVICES'])

    if use_tf:
        import keras.backend as K
        gpu_options=K.tf.GPUOptions(
            per_process_gpu_memory_fraction=0.0,
            allow_growth = True,)        
        K.set_session( K.tf.Session( config=K.tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=True,
            gpu_options=gpu_options
        ) ) )

        
    if model_source is not None:
        ## provide the model details here
        module = __import__(args.model.replace('.py','').replace('/', '.'), fromlist=[None])
        if use_tf:
            model_provider = BuilderFromFunction( model_fn = module.get_model )
        else:
            model_provider = TorchBuilderFromFunction( model_fn = module.get_model)

        (train_list, val_list) = make_train_val_lists(module, args)
        (features_name, labels_name) = make_features_labels(module, args)
    elif test == 'topclass':
        ### topclass example
        if not args.torch:
            model_provider = BuilderFromFunction( model_fn = models.make_topclass_model )
        else:
            model_provider = TorchBuilderFromFunction( model_fn = models.make_topclass_torch_model)

        if 'daint' in host:
            train_list = glob.glob('/scratch/snx3000/vlimant/data/LCDJets_Remake/train/*.h5')
            val_list = glob.glob('/scratch/snx3000/vlimant/data/LCDJets_Remake/val/*.h5')
        elif 'titan' in host:
            train_list = glob.glob('/ccs/proj/csc291/DATA/LCDJets_Abstract_IsoLep_lt_20/train/*.h5')
            val_list = glob.glob('/ccs/proj/csc291/DATA/LCDJets_Abstract_IsoLep_lt_20/val/*.h5')
        else:
            train_list = glob.glob('/bigdata/shared/LCDJets_Abstract_IsoLep_lt_20/train/0*.h5')
            val_list = glob.glob('/bigdata/shared/LCDJets_Abstract_IsoLep_lt_20/val/0*.h5')
        features_name='Images'
        labels_name='Labels'
    elif test == 'mnist':
        ### mnist example
        if args.torch:
            model_provider = TorchBuilderFromFunction( model_fn = models.make_mnist_torch_model)
        else:
            model_provider = BuilderFromFunction( model_fn = models.make_mnist_model)

        if 'daint' in host:
            all_list = glob.glob('/scratch/snx3000/vlimant/data/mnist/*.h5')
        elif 'titan' in host:
            all_list = glob.glob('/ccs/proj/csc291/DATA/mnist/*.h5')
        else:
            all_list = glob.glob('/bigdata/shared/mnist/*.h5')
        l = int( len(all_list)*0.70)
        train_list = all_list[:l]
        val_list = all_list[l:]
        features_name='features'
        labels_name='labels'
    elif test == 'cifar10':
        ### cifar10 example
        model_provider = BuilderFromFunction( model_fn = models.make_cifar10_model )

        if 'daint' in host:
            all_list = []
        elif 'titan' in host:
            all_list = glob.glob('/ccs/proj/csc291/DATA/cifar10/*.h5')
        else:
            all_list = glob.glob('/bigdata/shared/cifar10/*.h5')
        l = int( len(all_list)*0.70)
        train_list = all_list[:l]
        val_list = all_list[l:]
        features_name='features'
        labels_name='labels'
    elif test == 'gan':
        from nnlo.train.GanModel import GANBuilder
        ### the gan example
        model_provider = GANBuilder( parameters = [ Integer(50,400, name='latent_size' ),
                                                    Real(0.0, 1.0, name='discr_drop_out'),
                                                    Categorical([1, 2, 5, 6, 8], name='gen_weight'),
                                                    Categorical([0.1, 0.2, 1, 2, 10], name='aux_weight'),
                                                    Categorical([0.1, 0.2, 1, 2, 10], name='ecal_weight'),
                                                ]
        )
        ## only this mode functions
        setattr(args,"mode",'easgd')
        args.worker_optimizer = 'rmsprop'
        if 'daint' in host:
            all_list = glob.glob('/scratch/snx3000/vlimant/data/3DGAN/*.h5')
        elif 'titan' in host:
            all_list = glob.glob('/ccs/proj/csc291/DATA/3DGAN/*.h5')
        else:
            all_list = glob.glob('/data/shared/3DGAN/*.h5')

        #l = int( len(all_list)*0.70)
        #train_list = all_list[:l]
        #val_list = all_list[l:]
        N= MPI.COMM_WORLD.Get_size()        
        train_list = all_list[:N]
        val_list = all_list[-1:]
        features_name='X'
        labels_name='y'




    if use_torch:
        if 'gpu' in device:
            model_provider.gpus=1
            

    comm_block = comm_world.Split(block_num)
    logging.debug("Process {} sees {} blocks, has block number {}, and rank {} in that block".format(comm_world.Get_rank(),
                                                                                              num_blocks,
                                                                                              block_num,
                                                                                              comm_block.Get_rank()
                                                                                            ))
    if args.n_processes>1:
        t_b_processes= []
        if block_num !=0:
            _,_, b_processes = get_groups(comm_block, args.n_masters, args.n_processes)
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
        
        #need to collect all the processes lists
        all_t_b_processes = comm_world.allgather( t_b_processes )
        w_processes = set()
        for gb in all_t_b_processes:
            if gb:
                hgb = map(tuple, gb)
                w_processes.update( hgb )
        if block_num == 0:
            logging.info("all collect processes {}".format(w_processes))
            ## now you have the ranks that needs to be initialized in rings.

    # MPI process 0 coordinates the Bayesian optimization procedure
    if block_num == 0:
        opt_coordinator = Coordinator(comm_world, num_blocks,
                                      model_provider.parameters,
                                      (args.hyper_opt=='genetic'),args.population,
                                      checkpointing =  args.checkpoint,
                                      label = args.trial_name
        )
        if args.opt_restore: opt_coordinator.load()
        if args.target_objective: opt_coordinator.target_fom = args.target_objective
        opt_coordinator.run(num_iterations=args.num_iterations)
        opt_coordinator.record_details()
    else:
        logging.debug("Process {} on block {}, rank {}, create a process block".format( comm_world.Get_rank(),
                                                                                 block_num,
                                                                                 comm_block.Get_rank()))
        data = make_loader(args, features_name, labels_name, train_list)

        from TrainingDriver import make_algo
        algo = make_algo( args, use_tf, comm_block , validate_every=int(data.count_data()/args.batch ))
 
        block = ProcessBlock(comm_world, comm_block, algo, data, device,
                             model_provider,
                             args.epochs, train_list, val_list, 
                             folds = args.n_fold,
                             num_masters = args.n_masters,
                             num_process = args.n_processes,
                             verbose=args.verbose,
                             early_stopping=args.early_stopping,
                             target_metric=args.target_metric,
                             monitor=args.monitor,
                             label = args.trial_name,
                             restore = args.opt_restore,
                             checkpoint=args.checkpoint,
                             checkpoint_interval=args.checkpoint_interval)
        block.run()

if __name__ == '__main__':
    main()
