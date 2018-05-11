import time 
import numpy as np

import mpi_learn.mpi.manager as mm
import mpi_learn.train.model as model

from tag_lookup import tag_lookup

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
    callbacks: list of callback objects
    verbose: print detailed output from underlying mpi_learn machinery
    """

    def __init__(self, comm_world, comm_block, algo, data, device, model_provider,
                 epochs, train_list, val_list, folds=1, callbacks=None, verbose=False):
        print("Initializing ProcessBlock")
        self.comm_world = comm_world
        self.comm_block = comm_block
        self.folds = folds
        self.algo = algo
        self.data = data
        self.device = device
        self.model_provider = model_provider
        self.epochs = epochs
        self.train_list = train_list
        self.val_list = val_list
        self.callbacks = callbacks
        self.verbose = verbose

    def ranks(self):
        return "Process {}, sub-process {}".format( self.comm_world.Get_rank(), self.comm_block.Get_rank() )

    def wait_for_model(self):
        """
        Blocks until the parent sends a parameter set
        indicating the model that should be trained.
        """
        print("{} waiting for model params".format(self.ranks()))
        params = self.comm_world.recv(source=0, tag=tag_lookup('params'))
        if params is not None:
            print ("{} received parameters {}".format( self.ranks(), params))
            model_builder = self.model_provider.builder(*params)
            if model_builder:
                model_builder.comm = self.comm_block
                model_builder.device = model_builder.get_device_name(self.device)
            return model_builder
        return None

    def train_model(self, model_builder):
        fake_train = False
        if fake_train:
            if self.comm_block.Get_rank() == 0:
                    time.sleep(abs(np.random.randn()*30))
                    result = np.random.randn()
                    print("{} finished training with result {}".format(self.ranks(), result))
                    return result
        else:
            print("{} creating MPIManager".format(self.ranks()))
            ## need to reset this part to avoid cached values
            self.algo.reset()
            manager = mm.MPIKFoldManager( self.folds,
                                          self.comm_block, self.data, self.algo, model_builder,
                                          self.epochs, self.train_list, self.val_list, callbacks=self.callbacks,
                                          verbose=self.verbose)
            #if self.comm_block.Get_rank() == 0:
            #    print("{} launching training".format(self.ranks()))
            manager.train()
            fom = manager.figure_of_merit()
            ## delete the object
            del manager
            return fom

    def send_result(self, result):
        if self.comm_block.Get_rank() == 0:
            ## only the rank=0 in the block is sending back his fom
            print("{} sending result {} to coordinator".format(self.ranks(), result))
            self.comm_world.isend(result, dest=0, tag=tag_lookup('result')) 

    def run(self):
        """
        Awaits instructions from the parent to train a model.
        Then trains it and returns the loss to the parent.
        """
        while True:
            print("{} waiting for model".format(self.ranks()))
            cur_builder = self.wait_for_model()
            if cur_builder == None:
                print("{} received exit signal from coordinator".format(self.ranks()))
                break
            
            print("{} will train model".format(self.ranks()))
            fom = self.train_model(cur_builder)
            print("{} will send result if needed".format(self.ranks()))
            self.send_result(fom)


