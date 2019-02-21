#!/usr/bin/env python3

import sys,os
import numpy as np
import argparse
import json
import time
import glob
import socket
from mpi4py import MPI
import hashlib

sys.path.append(os.path.dirname(os.path.realpath(__file__))+'/mpi_learn_src')
from mpi_learn.train.algo import Algo
from mpi_learn.train.data import H5Data
from mpi_learn.train.model import ModelFromJsonTF, ModelPytorch
from mpi_learn.utils import import_keras
import mpi_learn.mpi.manager as mm
from mpi_learn.train.GanModel import GANBuilder
from skopt.space import Real, Integer, Categorical

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

    def _json(self,*params):
        m = self.model_fn( **self._args(*params))
        return m.to_json()
    
    def builder(self,*params):
        try:
            return ModelFromJsonTF(None,
                                   json_str=self._json(*params))
        except:
            str_param = ','.join('{0}={1!r}'.format(k,v) for k,v in args.items())
            print("Failed to build model with params: {}".format(str_param))
            return None
        
    
class BuilderFromFunctionJ(BuilderFromFunction):
    def __init__(self, model_fn, parameters=None):
        BuilderFromFunction.__init__(self, model_fn, parameters)
        
    def _json(self,*params):
        return self.model_fn( **self._args(*params))

class TorchBuilderFromFunction(BuilderFromFunction):
    def __init__(self, model_fn, parameters=None, gpus=0):
        super().__init__(model_fn, parameters)
        self.gpus = gpus

    def builder(self, *params):
        args = dict(zip([p.name for p in self.parameters], params))
        try:
            model_pytorch = self.model_fn(**args)
            ## save it to a temp file indeed
            username = os.environ.get('USER')
            os.system('mkdir -p /tmp/{}'.format( username ))
            args_s = str(args).encode('utf-8')
            hashs = hashlib.sha224(args_s).hexdigest()
            
            model_path = "/tmp/{}/_{}_{}_pytorch.torch".format(username,os.getpid(),hashs)
            torch.save(model_pytorch, model_path)
            return ModelPytorch(None, filename=model_path, gpus=self.gpus)
        except:
            str_param = ','.join('{0}={1!r}'.format(k,v) for k,v in args.items())
            print("Failed to build model with params: {}".format(str_param))
            return None

import coordinator
import process_block
try:
    ## first try to get from mpi_learn
    import models.Models as models
except:
    print ("failed to load mpi_learn")

## where the models were defined before
#import mpiLAPI as mpi 


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
    parser.add_argument('--monitor',help='Monitor cpu and gpu utilization', action='store_true')
    parser.add_argument('--label',default='hOpt')
    parser.add_argument('--batch', help='batch size', default=128, type=int)
    parser.add_argument('--epochs', help='number of training epochs', default=10, type=int)
    parser.add_argument('--optimizer',help='optimizer for master to use',default='adam')
    parser.add_argument('--loss',help='loss function',default='binary_crossentropy')
    parser.add_argument('--sync-every', help='how often to sync weights with master', 
            default=1, type=int, dest='sync_every')
    parser.add_argument('--preload-data', help='Preload files as we read them', default=0, type=int, dest='data_preload')
    parser.add_argument('--cache-data', help='Cache the input files to a provided directory', default='', dest='caching_dir')
    parser.add_argument('--early-stopping', default=None,
                        dest='early_stopping', help='patience for early stopping')
    parser.add_argument('--target-metric', default=None,
                        dest='target_metric', help='Passing configuration for a target metric')

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
    parser.add_argument('--hyper-opt', dest='hyper_opt', default='bayesian', choices=['bayesian','genetic'],
                        help='The algorithm to use for the hyper paramater optimization')
    parser.add_argument('--ga-populations', help='population size for genetic algorithm',
                        default=10, type=int, dest='population')
    parser.add_argument('--previous-result', help='Load the optimizer state from a previous run', default=None,dest='previous_state')
    parser.add_argument('--target-objective', type=float, default=None,dest='target_objective',
                        help='A value to reach and stop in the parameter optimisation')
    parser.add_argument('--example', default='mnist', choices=['topclass','mnist','gan','cifar10'])
    parser.add_argument('--torch', action='store_true',
                        help='Use PyTorch instead of (default) Keras')
    return parser


if __name__ == '__main__':

    print ("Process is on",socket.gethostname())
    parser = make_parser()
    args = parser.parse_args()
    check_sanity(args)

    import socket
    host = os.environ.get('HOST',os.environ.get('HOSTNAME',socket.gethostname()))

    test = args.example
    if test == 'topclass':
        ### topclass example
        if not args.torch:
            model_provider = BuilderFromFunction( model_fn = models.make_topclass_model )
        else:
            model_provider = TorchBuilderFromFunction( model_fn = models.make_topclass_torch_model )

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
        ### the gan example
        model_provider = GANBuilder( parameters = [ Integer(50,400, name='latent_size' ),
                                                    Real(0.0, 1.0, name='discr_drop_out'),
                                                    Categorical([1, 2, 5, 6, 8], name='gen_weight'),
                                                    Categorical([0.1, 0.2, 1, 2, 10], name='aux_weight'),
                                                    Categorical([0.1, 0.2, 1, 2, 10], name='ecal_weight'),
                                                ]
        )
        ## only this mode functions
        args.easgd = True
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
    hide_device = True
    if hide_device:
        os.environ['CUDA_VISIBLE_DEVICES'] = device[-1] if 'gpu' in device else ''
        print ('set to device',os.environ['CUDA_VISIBLE_DEVICES'])

    if not args.torch:
        import keras.backend as K
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
    else:
        import torch
        if 'gpu' in device and not hide_device:
            torch.cuda.set_device(int(device[-1]))
        if 'gpu' in device:
            model_provider.gpus=1
            
    print("Process {} using device {}".format(comm_world.Get_rank(), device))
    comm_block = comm_world.Split(block_num)
    print ("Process {} sees {} blocks, has block number {}, and rank {} in that block".format(comm_world.Get_rank(),
                                                                                              num_blocks,
                                                                                              block_num,
                                                                                              comm_block.Get_rank()
                                                                                            ))
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
        opt_coordinator = coordinator.Coordinator(comm_world, num_blocks,
                                                  model_provider.parameters,
                                                  (args.hyper_opt=='genetic'),args.population)
        if args.previous_state: opt_coordinator.load(args.previous_state)
        if args.target_objective: opt_coordinator.target_fom = args.target_objective
        opt_coordinator.label = args.label
        opt_coordinator.run(num_iterations=args.num_iterations)
        opt_coordinator.record_details()
    else:
        print ("Process {} on block {}, rank {}, create a process block".format( comm_world.Get_rank(),
                                                                                 block_num,
                                                                                 comm_block.Get_rank()))
        data = H5Data(batch_size=args.batch,
                      cache = args.caching_dir,
                      preloading = args.data_preload,
                      features_name=features_name,
                      labels_name=labels_name
        )
        print('found data')
        data.set_file_names( train_list )
        print('set file names')
        validate_every = int(data.count_data()/args.batch )
        print('validate every')
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
        #import_keras()
        block = process_block.ProcessBlock(comm_world, comm_block, algo, data, device,
                                           model_provider,
                                           args.epochs, train_list, val_list, 
                                           folds = args.n_fold,
                                           num_masters = args.n_master,
                                           num_process = args.n_process,
                                           verbose=args.verbose,
                                           early_stopping=args.early_stopping,
                                           target_metric=args.target_metric,
                                           monitor=args.monitor)
        block.label = args.label
        block.run()
