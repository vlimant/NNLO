import time 
import numpy as np
import os
import hashlib
import logging
import glob
from ..mpi.manager import MPIKFoldManager
from ..util.logger import set_logging_prefix
from ..util.utils import opt_tag_lookup

class ProcessBlock(object):
    """
    This class represents a block of processes that run model training together.

    Attributes:
    comm_world: MPI communicator with all processes.
        Used to communicate with process 0, the coordinator
    comm_block: MPI communicator with the processes in this block.
        Rank 0 is the master, other ranks are workers.
    algo: MPI Algo object
    data: MPI Data object
    device: string indicating which device (cpu or gpu) should be used
    epochs: number of training epochs
    train_list: list of training data files
    val_list: list of validation data files
    verbose: print detailed output from underlying training machinery
    """

    def __init__(self, comm_world, comm_block, algo, data, device, model_provider,
                 epochs, train_list, val_list, folds=1,
                 num_masters=1,
                 num_process=1,
                 verbose=False,
                 early_stopping=None,
                 target_metric=None,
                 monitor=False,
                 label = None,
                 restore = False,
                 checkpoint=None,
                 checkpoint_interval=5):
        set_logging_prefix(
            comm_world.Get_rank(),
            comm_block.Get_rank() if comm_block is not None else '-',
            '-',
            'B'
        )
        logging.debug("Initializing ProcessBlock")
        self.comm_world = comm_world
        self.comm_block = comm_block
        self.folds = folds
        self.num_masters = num_masters
        self.num_process = num_process
        self.algo = algo
        self.data = data
        self.device = device
        self.model_provider = model_provider
        self.epochs = epochs
        self.train_list = train_list
        self.val_list = val_list
        self.verbose = verbose
        self.last_params = None
        self.early_stopping=early_stopping
        self.target_metric=target_metric
        self.monitor = monitor
        self.current_builder = None
        self.restore = restore
        self.checkpoint = checkpoint
        self.checkpoint_interval = checkpoint_interval
        self.label = checkpoint if checkpoint else label
        
    def wait_for_model(self):
        """
        Blocks until the parent sends a parameter set
        indicating the model that should be trained.
        """
        logging.debug("Waiting for model params")
        self.last_params = self.comm_world.recv(source=0, tag=opt_tag_lookup('params'))
        params = self.last_params
        if params is not None:
            logging.debug("Received parameters {}".format(params))
            model_builder = self.model_provider.builder(*params)
            if model_builder:
                model_builder.comm = self.comm_block
                self.current_builder = model_builder
            else:
                self.current_builder = None
            return True
        return False

    def train_model(self):
        if self.current_builder is None:
            # Invalid model, return nonsense FoM
            return np.nan
        fake_train = False
        if fake_train:
            if self.comm_block.Get_rank() == 0:
                    time.sleep(abs(np.random.randn()*30))
                    result = np.random.randn()
                    logging.debug("Finished training with result {}".format(result))
                    return result
        else:
            logging.debug("Creating a manager")
            history_name = '{}-block-{}'.format(self.label,
                                                hashlib.md5(str(self.last_params).encode('utf-8')).hexdigest())
            ## need to reset this part to avoid cached values
            self.algo.reset()
            if self.restore:
                if os.path.isfile(history_name + '.latest'):
                    with open(history_name + '.latest', 'r') as latest:
                        restore_name = latest.read().splitlines()[-1]
                else:
                    restore_name = history_name
                if any([os.path.isfile(ff) for ff in glob.glob('./*' + restore_name + '.model')]):
                    self.current_builder.weights = restore_name + '.model'
                self.algo.load(restore_name)
                self.restore= False
            manager = MPIKFoldManager( self.folds,
                                       self.comm_block, self.data, self.algo, self.current_builder,
                                       self.epochs, self.train_list, self.val_list,
                                       num_masters=self.num_masters,
                                       num_process=self.num_process,
                                       verbose=self.verbose,
                                       early_stopping=self.early_stopping,
                                       target_metric=self.target_metric,
                                       monitor=self.monitor,
                                       checkpoint=history_name if self.checkpoint else None,
                                       checkpoint_interval=self.checkpoint_interval)
            manager.train()
            fom = manager.figure_of_merit()
            manager.manager.process.record_details(
                json_name=history_name + '.json',
                meta={'parameters': list(map(float,self.last_params)),
                                                         'fold' : manager.fold_num})
            manager.close()
            return fom

    def send_result(self, result):
        if self.comm_block.Get_rank() == 0:
            ## only the rank=0 in the block is sending back his fom
            logging.debug("Sending result {} to coordinator".format(result))
            self.comm_world.isend(result, dest=0, tag=opt_tag_lookup('result')) 

    def run(self):
        """
        Awaits instructions from the parent to train a model.
        Then trains it and returns the loss to the parent.
        """
        while True:
            self.comm_block.Barrier()
            logging.debug("Waiting for model")
            have_builder = self.wait_for_model()
            if not have_builder:
                logging.debug("Received exit signal from coordinator")
                break
            
            logging.debug("Will train model")
            fom = self.train_model()
            logging.debug("Done training, will send result if needed")
            self.send_result(fom)
        self.comm_world.Barrier()


