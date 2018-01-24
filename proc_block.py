import time 

import mpi_learn.mpi.manager as mm
import mpi_learn.train.model as model

class ProcBlock(object):
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
    """

    def __init__(self, comm_world, comm_block, algo, data,
            epochs, train_list, val_list, callbacks=None):
        self.comm_world = comm_world
        self.comm_block = comm_block
        self.algo = algo
        self.data = data
        self.device = device
        self.epochs = epochs
        self.train_list = train_list
        self.val_list = val_list
        self.callbacks = callbacks

    def wait_for_model(self):
        """
        Blocks until the parent sends a JSON string
        indicating the model that should be trained.
        """
        model_str = self.comm_world.Recv(source=0, tag='json') # note: will not work
        return model_str

    def train_model(self, model_json):
        model_builder = model.ModelFromJsonTF(self.comm_block, # note: will not work
            json_str=model_json, device_name=self.device)
        manager = mm.MPIManager(self.comm_block, self.data, self.algo, model_builder,
                self.epochs, self.train_list, self.val_list, callbacks=self.callbacks)
        if self.comm_block.Get_rank() == 0:
            histories = manager.process.train()
        print(histories)
        return histories['0']['val_loss'][-1]

    def send_result(self, result):
        self.comm_world.isend(result, dest=0, tag='result') # note: will not work

    def run(self):
        """
        Awaits instructions from the parent to train a model.
        Then trains it and returns the loss to the parent.
        """
        while True:
            cur_model = self.wait_for_model()
            fom = self.train_model(cur_model)
            self.send_result(fom)
