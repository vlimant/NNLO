import skopt 

class Coordinator(object):
    """
    This class coordinates the Bayesian optimization procedure.
    
    Attributes:
    comm: MPI communicator object containing all processes.
    num_blocks: int, number of blocks of workers
    opt_params: list of parameter ranges, suitable to pass
        to skopt.Optimizer's constructor
    optimizer: skopt.Optimizer instance
    model_fn: python function.  should take a list of parameters
        and return a JSON string representing the keras model to train
    param_list: list of parameter sets tried
    fom_list: list of figure of merit (FOM) values obtained
    block_dict: dict keeping track of which block is running which model
    """

    def __init__(comm, num_blocks, opt_params,
            model_fn):
        self.comm = comm
        self.num_blocks = num_blocks
        self.opt_params = opt_params
        self.model_fn = model_fn
       
        self.optimizer = skopt.Optimizer(opt_params)
        self.param_list = []
        self.fom_list = []
        self.block_dict = {}

    def run(self, num_iterations=1):
        for step in range(num_iterations):
            next_block = self.wait_for_idle_block()
            next_params = self.optimizer.ask()
            self.run_block(next_block, next_params) 
        
    def wait_for_idle_block(self):
        """
        In a loop, checks each block of processes to see if it's
        idle.  This function blocks until there is an available process.
        """
        cur_block = 0
        while True:
            idle = self.check_block(cur_block)
            if idle:
                return cur_block
            cur_block = (cur_block + 1) % self.num_blocks

    def check_block(self, block_num):
        block_size = int((self.comm.Get_size()-1)/self.num_blocks)
        proc = block_num * block_size
        result = self.comm.irecv(proc, tag='result') # note: will not work
        if result:
            params = self.block_dict.pop(block_num)
            self.param_list.append(params)
            self.fom_list.append(result)
            return True
        return False

    def run_block(self, block_num, params):
        self.block_dict[block_num] = params
        model_json = self.model_fn(next_params)
        # In the current setup, we need to signal each GPU in the 
        # block to start training
        block_size = int((self.comm.Get_size()-1)/self.num_blocks)
        start = block_num * block_size
        end = (block_num+1) * block_size
        for proc in range(start, end):
            self.comm.Send(model_json, dest=proc, tag='json') # note: will not work
