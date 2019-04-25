### MPIWorker and MPIMaster classes 

import os,sys,json
import numpy as np
import socket
import time
import threading, queue
import logging

from mpi4py import MPI

from ..train.monitor import Monitor
from ..train.trace import Trace, trace
from ..utils import Error, weights_from_shapes, shapes_from_weights
from ..logger import get_logger, set_logging_prefix

### Classes ###

class MPIProcess(object):
    """Base class for processes that communicate with one another via MPI.  

       Attributes:
           verbose: display verbose output
           parent_comm: MPI intracommunicator used to communicate with this process's parent
           parent_rank (integer): rank of this node's parent in parent_comm
           rank (integer): rank of this node in parent_comm
           model: Keras model to train
           model_builder: ModelBuilder object specifying model
           algo: Algo object defining how to optimize model weights
           weights_shapes: list of tuples indicating the shape of each layer of the model
           weights: list of numpy arrays storing the last weights received from the parent
           update: latest update obtained from training
           data: Data object used to generate training or validation data
           time_step: for keeping track of time
           num_epochs: integer giving the number of epochs to train for
           stop_training: becomes true when it is time to stop training
    """

    def __init__(self, parent_comm, process_comm, parent_rank=None, num_epochs=1, data=None, algo=None,
            model_builder=None, verbose=False, monitor=False, custom_objects={},
            checkpoint=None, checkpoint_interval=5):
        """If the rank of the parent is given, initialize this process and immediately start 
            training. If no parent is indicated, training should be launched with train().
            
            Parameters:
              parent_comm: MPI intracommunicator used to communicate with parent
              parent_rank (integer): rank of this node's parent in parent_comm
              num_epochs: number of training epochs
              data: Data object used to generate training or validation data
              algo: Algo object used to configure the training process
              model_builder: ModelBuilder object specifying model
              verbose: whether to print verbose output
              monitor: whether to monitor CPU/GPU usage
        """
        self.parent_comm = parent_comm
        self.process_comm = process_comm
        self.parent_rank = parent_rank
        self.num_epochs = num_epochs
        self.data = data
        self.algo = algo
        self.model_builder = model_builder
        self.verbose = verbose
        self.histories = {}
        self.custom_objects = custom_objects

        self.update = None
        self.stop_training = False
        self.time_step = 0
        self._short_batches = 0
        self._is_shadow = (self.process_comm is not None and self.process_comm.Get_rank()!=0)

        self.monitor = Monitor() if monitor else None

        process_type = self.__class__.__name__.replace('MPI', '')[0]
        set_logging_prefix(
            MPI.COMM_WORLD.Get_rank(),
            self.parent_comm.Get_rank() if self.parent_comm is not None else '-',
            self.process_comm.Get_rank() if self.process_comm is not None else '-',
            process_type
        )
        self.logger = get_logger()

        if self.process_comm is not None and self.process_comm.Get_size() > 1:
            if self.model_builder.get_backend_name() == 'pytorch':
                import horovod.torch as hvd
            else:
                import horovod.keras as hvd
            self.logger.debug("initializing horovod")
            self.process_comm.Barrier()
            hvd.init(comm=self.process_comm)
            self.process_comm.Barrier()
            self.algo.worker_optimizer_builder.horovod_wrapper = True
        
        self.rank = parent_comm.Get_rank() if parent_comm else 0
        self.ranks = "{0}:{1}:{2}".format(
            MPI.COMM_WORLD.Get_rank(),
            self.parent_comm.Get_rank() if self.parent_comm is not None else '-',
            self.process_comm.Get_rank() if self.process_comm is not None else '-')

        Trace.set_process_name(type(self).__name__ + " " + self.ranks)

        self.epoch = 0
        self.checkpoint = checkpoint
        self.checkpoint_interval = checkpoint_interval
        if self.algo.restore and self.checkpoint is not None:
            try:
                with open(self.checkpoint + '.latest', 'r') as latest:
                    latest_file = latest.read().splitlines()[-1]
                epoch = int(latest_file.split('-')[-1])
            except:
                epoch = 0
                self.logger.error("Failed to restore epoch from checkpoint {}".format(self.checkpoint))
            if epoch < num_epochs:
                self.epoch = epoch
                self.num_epochs = num_epochs - epoch
                self.logger.info("Continuing training from epoch {} for {} epochs".format(self.epoch, self.num_epochs))

        self.build_model()
        if (self.parent_rank is not None and self.parent_comm is not None):
            self.bcast_weights( self.parent_comm )
        if (self.parent_rank is not None and self.parent_comm is not None) or (self.process_comm):
            self.train()

    def record_details(self, json_name=None, meta=None):
        """to be implemented by the specific process
        """
        pass

    def save_checkpoint(self):
        if self.checkpoint is not None and self.epoch % self.checkpoint_interval == 0:
            file_name = '{}-{}'.format(self.checkpoint, self.epoch)
            if self.model:
                self.model.save(file_name + '.model')
            if self.algo:
                self.algo.save(file_name + '.algo')

            with open(self.checkpoint + '.latest', 'w') as latest:
                latest.write(file_name)

    def history_key(self):
        #return str(self.rank)
        return self.ranks

    def update_monitor(self, perf):
        r= self.history_key()
        self.histories.setdefault(r,{})['monitor'] = perf
        
    def update_history(self, items):
        r= self.history_key()
        self.model.update_history( items, self.histories.setdefault(r,{}))

    def is_shadow(self, sync = False):
        """signals that the process is a sub-process and should not act normally"""
        if self.process_comm and sync:
            import inspect
            self.logger.debug("syncing on the process communicator from",inspect.stack()[1][3])
            self.process_comm.Barrier()
        return self._is_shadow

    def build_model(self, local_session=True):
        """Builds the Keras model and updates model-related attributes"""
        self.model = self.model_builder.build_model(local_session=local_session)
        self.weights = self.model.get_weights()
        self.compile_model()
        self.weights = self.model.get_weights()
        self.update = self.model.format_update()

    def check_sanity(self):
        """Throws an exception if any model attribute has not been set yet."""
        for par in ['model',
                    #'weights_shapes',
                    'weights','update']:
            if not hasattr(self, par) or getattr(self, par) is None:
                raise Error("%s not found!  Process %s does not seem to be set up correctly." % (par,self.ranks))

    def train(self):
        """To be implemented in derived classes"""
        raise NotImplementedError

    def compile_model(self):
        """Compile the model. Note that the compilation settings
            are relevant only for Workers because the Master updates
            its weights using an mpi_learn optimizer."""
        self.logger.debug("Compiling model")
        self.algo.compile_model( self.model )
        self.logger.debug("Compiled")
        
    def print_metrics(self, metrics):
        """Display metrics computed during training or validation"""
        self.model.print_metrics( metrics )

    def get_logs(self, metrics, val=False):
        """Get dictionary of logs computed during training.
            If val is True, appends 'val' to the beginning of each metric name"""
        MPI.COMM_WORLD.Abort()
        if val:
            return { 'val_'+name:np.asscalar(metric) for name, metric in 
                    zip( self.model.metrics_names(), metrics ) }
        else:
            return { name:np.asscalar(metric) for name, metric in 
                    zip( self.model.metrics_names(), metrics ) }

    @trace(category="MPI")
    def do_send_sequence(self):
        """Actions to take when sending an update to parent:
            -Send the update (if the parent accepts it)
            -Sync time and model weights with parent"""
        if self.is_shadow():
            return
        #time.sleep(1)
        logging.debug("do_send_sequence:: sending update")
        self.send_update(check_permission=True)
        logging.debug("do_send_sequence:: receiving time step")
        self.time_step = self.recv_time_step()
        logging.debug("do_send_sequence:: receiving weights")
        self.recv_weights()
        logging.debug("do_send_sequence:: setting weight")        
        self.algo.set_worker_model_weights( self.model, self.weights )
        logging.debug("do_send_sequence:: done")
        
    def apply_update(self):
        """Updates weights according to update received from worker process"""
        with np.errstate( divide='raise',
                          invalid='raise',
                          over='raise',
                          #under ='raise'## might not be too bad
                      ):
            self.weights = self.algo.apply_update( self.weights, self.update )
        
    ### MPI-related functions below ###

    # This dict associates message strings with integers to be passed as MPI tags.
    tag_lookup = {
            'any':          MPI.ANY_TAG,
            'train':          0,
            'exit':           1,
            'begin_weights':  2,
            'begin_update'  : 3,
            'time':           4,
            'bool':           5,
            'history':        6,
            'weights':        12,
            'update':         13,
            'begin_gem':      14,
            'update_gem':     16,
            }
    # This dict is for reverse tag lookups.
    inv_tag_lookup = { value:key for key,value in tag_lookup.items() }

    def lookup_mpi_tag( self, name, inv=False ):
        """Searches for the indicated name in the tag lookup table and returns it if found.
            Params:
              name: item to be looked up
              inv: boolean that is True if an inverse lookup should be performed (int --> string)"""
        if inv:
            lookup = self.inv_tag_lookup
        else:
            lookup = self.tag_lookup
        try:
            return lookup[name]
        except KeyError:
            self.logger.error("Not found in tag dictionary: {0} -- returning None".format(name))
            return None

    def recv(self, obj=None, tag=MPI.ANY_TAG, source=None, buffer=False, status=None, comm=None):
        """Wrapper around MPI.recv/Recv. Returns the received object.
            Params:
              obj: variable into which the received object should be placed
              tag: string indicating which MPI tag should be received
              source: integer rank of the message source.  Defaults to self.parent_rank
              buffer: True if the received object should be sent as a single-segment buffer
                (e.g. for numpy arrays) using MPI.Recv rather than MPI.recv
              status: MPI status object that is filled with received status information
              comm: MPI communicator to use.  Defaults to self.parent_comm"""
        if comm is None:
            comm = self.parent_comm
        if source is None:
            if self.parent_rank is None:
                raise Error("Attempting to receive %s from parent, but parent rank is None" % tag)
            source = self.parent_rank 
        tag_num = self.lookup_mpi_tag(tag)
        if tag == 'history':
            obj = comm.recv( source=source, tag=tag_num, status=status )
            return obj
        #if tag in ['bool','time']:
        #    comm.Recv(obj, source=source, tag=tag_num, status=status )
        #    return obj
        logging.debug("recv:: comm {} size {} rank {} from {}".format(comm.Get_group(), comm.Get_size(), comm.Get_rank(), source))        
        if buffer:
            if type(obj) == list:
                for o in obj:
                    if type(o) == list:
                        for sub_o in o:
                            logging.debug("recv::Recv sub_o {} {} {}".format(source, tag_num, status))                            
                            comm.Recv( sub_o, source=source, tag=tag_num, status=status )
                            logging.debug("recv::Recv +sub_o {} {} {}".format(source, tag_num, status))                                                        
                    else:
                        logging.debug("recv::Recv o {} {} {}".format(source, tag_num, status))
                        comm.Recv( o, source=source, tag=tag_num, status=status )
                        logging.debug("recv::Recv +o {} {} {}".format(source, tag_num, status))                                                
            else:
                logging.debug("recv::Recv obj {} {} {}".format(source, tag_num, status))
                comm.Recv( obj, source=source, tag=tag_num, status=status )
                logging.debug("recv::Recv +obj {} {} {}".format(source, tag_num, status))
            return obj
        else:
            logging.debug("recv::recv {} {} {}".format(source, tag_num, status))            
            obj = comm.recv( source=source, tag=tag_num, status=status )
            logging.debug("recv::recv + {} {} {}".format(source, tag_num, status))                        
            return obj

    def send(self, obj, tag, dest=None, buffer=False, comm=None):
        """Wrapper around MPI.send/Send.  Params:
             obj: object to send
             tag: string indicating which MPI tag to send
             dest: integer rank of the message destination.  Defaults to self.parent_rank
             buffer: True if the object should be sent as a single-segment buffer
                (e.g. for numpy arrays) using MPI.Send rather than MPI.send
             comm: MPI communicator to use.  Defaults to self.parent_comm"""
        if comm is None:
            comm = self.parent_comm
        if dest is None:
            if self.parent_rank is None:
                raise Error("Attempting to send %s to parent, but parent rank is None" % tag)
            dest = self.parent_rank
        tag_num = self.lookup_mpi_tag(tag)
        if tag in ['history']:
            comm.send( obj, dest=dest, tag=tag_num )
            return
        #if tag in ['time']:
        #    comm.Send( obj, dest=dest, tag=tag_num )
        #    return
        logging.debug("send:: comm {} size {} rank {} to {}".format(comm.Get_group(), comm.Get_size(), comm.Get_rank(), dest))
        if buffer:
            if type(obj) == list:
                for o in obj:
                    if type(o) == list:
                        for sub_o in o:
                            logging.debug("send::Send sub_o {} {}".format(dest, tag_num))
                            comm.Send( sub_o, dest=dest, tag=tag_num )
                            logging.debug("send::Send +sub_o {} {}".format(dest, tag_num))
                    else:
                        logging.debug("send::Send o {} {}".format(dest, tag_num))
                        comm.Send( o, dest=dest, tag=tag_num )
                        logging.debug("send::Send +o {} {}".format(dest, tag_num))
            else:
                logging.debug("send::Send obj {} {}".format(dest, tag_num))                 
                comm.Send( obj, dest=dest, tag=tag_num )
                logging.debug("send::Send +obj {} {}".format(dest, tag_num))                
        else:
            if type(obj) == list:
                for o in obj:
                    if type(o) == list:
                        for sub_o in o:
                            logging.debug("send::Send sub_o {} {}".format(dest, tag_num))                            
                            comm.Send( sub_o, dest=dest, tag=tag_num )
                            logging.debug("send::Send +sub_o {} {}".format(dest, tag_num))                                                        
                    else:
                        logging.debug("send::Send o {} {}".format(dest, tag_num))
                        comm.Send( o, dest=dest, tag=tag_num )
                        logging.debug("send::Send +o {} {}".format(dest, tag_num))                        
            else:
                logging.debug("send::send obj {} {}".format(dest, tag_num))                
                comm.send( obj, dest=dest, tag=tag_num )
                logging.debug("send::send +obj {} {}".format(dest, tag_num))                                

    def bcast(self, obj, root=0, buffer=False, comm=None):
        """Wrapper around MPI.bcast/Bcast.  Returns the broadcasted object.
            Params: 
              obj: object to broadcast
              root: rank of the node to broadcast from
              buffer: True if the object should be sent as a single-segment buffer
                (e.g. for numpy arrays) using MPI.Bcast rather than MPI.bcast
              comm: MPI communicator to use.  Defaults to self.parent_comm"""
        if comm is None:
            comm = self.parent_comm
        if buffer:
            if type(obj) == list:
                for o in obj:
                    if type(o) == list:
                        for sub_o in o:
                            comm.Bcast( sub_o, root=root )
                    else:
                        comm.Bcast(o, root=root )
            else:
                comm.Bcast( obj, root=root )
        else:
            obj = comm.bcast( obj, root=root )
            return obj

    def send_exit_to_parent(self):
        if self.is_shadow( sync = True): return
        """Send exit tag to parent process, if parent process exists"""
        if self.parent_rank is not None:
            self.send( None, 'exit' )

    def send_history_to_parent(self):
        if self.is_shadow():return        
        """Send keras history or dict of keras histories"""
        if self.parent_rank is not None:
            self.send( obj=self.histories, tag='history' )

    def send_arrays(self, obj, expect_tag, tag, comm=None, dest=None, check_permission=False):
        """Send a list of numpy arrays to the process specified by comm (MPI communicator) 
            and dest (rank).  We first send expect_tag to tell the dest process that we 
            are sending several buffer objects, then send the objects layer by layer.
            Optionally check first to see if the update will be accepted by the master"""
        logging.debug("send_array:: advertising {} to {}".format( expect_tag, dest ))
        self.send( None, expect_tag, comm=comm, dest=dest )
        if check_permission:
            logging.debug("send_array:: checking permission to send {} to {}".format( tag, dest))
            # To check permission we send the update's time stamp to the master.
            # Then we wait to receive the decision yes/no.
            self.send_time_step( comm=comm, dest=dest )
            decision = self.recv_bool( comm=comm, source=dest )
            if not decision: return
        logging.debug("send_array:: expecting to send {} object".format( len( obj)))
        for i_o,o in enumerate(obj):
            #if tag == 'weights': time.sleep(0.1)
            if type(o) == list:
                for w in o:
                    logging.debug("send_arrays:: send w {} to {} {}/{}".format(tag, dest, i_o, len(obj)))
                    self.send( w, tag, comm=comm, dest=dest, buffer=True )
            else:
                logging.debug("send_arrays:: send o {} to {} {}/{}".format(tag, dest, i_o, len(obj)))
                self.send( o, tag, comm=comm, dest=dest, buffer=True )

    def send_weights(self, comm=None, dest=None, check_permission=False):
        if self.is_shadow():return        
        """Send NN weights to the process specified by comm (MPI communicator) and dest (rank).
            Before sending the weights we first send the tag 'begin_weights'."""
        self.send_arrays( self.weights, expect_tag='begin_weights', tag='weights', 
                comm=comm, dest=dest, check_permission=check_permission )

    def send_update(self, comm=None, dest=None, check_permission=False):
        if self.is_shadow():return        
        """Send update to the process specified by comm (MPI communicator) and dest (rank).
            Before sending the update we first send the tag 'begin_update'"""
        self.send_arrays( self.update, expect_tag='begin_update', tag='update', 
                comm=comm, dest=dest, check_permission=check_permission )

    def send_time_step(self, comm=None, dest=None):
        if self.is_shadow():return        
        """Send the current time step"""
        new_t = int(self.time_step)
        self.send( obj=new_t, tag='time', dest=dest, comm=comm)

    def send_bool(self, obj, comm=None, dest=None):
        if self.is_shadow():return
        new_b = bool(obj)
        self.send( obj=new_b, tag='bool', dest=dest, comm=comm)

    def recv_arrays(self, obj, tag, comm=None, source=None, add_to_existing=False):
        """Receive a list of numpy arrays from the process specified by comm (MPI communicator) 
            and dest (rank).
              obj: list of destination arrays 
              tag: MPI tag accompanying the message
              add_to_existing: if true, add to existing object instead of replacing"""
        if add_to_existing:
            #TODO this needs some work
            tmp = weights_from_shapes( [ w.shape for w in obj ] )
            self.recv_arrays( tmp, tag=tag, comm=comm, source=source )
            for i in range(len(obj)):
                obj[i] += tmp[i]
            return
        logging.debug("recv_array:: expecting to receive {} object".format( len( obj)))
        for i_o,o in enumerate(obj):
            #if tag == 'weights': time.sleep(0.1)            
            if type(o) == list:
                for w in o:
                    logging.debug("recv_array:: recv w {} from {} {}/{}".format(tag, source, i_o, len(obj)))
                    self.recv( w, tag, comm=comm, source=source, buffer=True )
            else:
                logging.debug("recv_array:: recv o {} from {} {}/{}".format(tag, source, i_o, len(obj)))                
                self.recv( o, tag, comm=comm, source=source, buffer=True )

    def recv_weights(self, comm=None, source=None, add_to_existing=False):
        """Receive NN weights layer by layer from the process specified by comm and source"""
        if self.is_shadow():return        
        self.recv_arrays( self.weights, tag='weights', comm=comm, source=source,
                add_to_existing=add_to_existing )

    def recv_update(self, comm=None, source=None, add_to_existing=False):
        """Receive an update layer by layer from the process specified by comm and source.
            Add it to the current update if add_to_existing is True, 
            otherwise overwrite the current update"""
        if self.is_shadow():return        
        self.recv_arrays( self.update, tag='update', comm=comm, source=source,
                add_to_existing=add_to_existing )

    def recv_time_step(self, comm=None, source=None):
        """Receive the current time step"""
        if self.is_shadow():return
        return self.recv( tag='time', comm=comm, source=source)

    def recv_bool(self, comm=None, source=None):
        if self.is_shadow():return        
        return self.recv( tag='bool', comm=comm, source=source)

    def recv_exit_from_parent(self):
        ir = None
        if not self.is_shadow():
            ir = self.parent_comm.irecv( source=0, tag=self.lookup_mpi_tag('exit') )
        elif self.process_comm:
            ir = self.process_comm.irecv( source=0, tag=self.lookup_mpi_tag('exit') )
        return ir


    def bcast_weights(self, comm, root=0):
        """Broadcast weights shape and weights (layer by layer) 
            on communicator comm from the indicated root rank"""
        self.bcast( self.weights, comm=comm, root=root, buffer=True )


class MPIWorker(MPIProcess):
    """This class trains its NN model and exchanges weight updates with its parent."""

    def __init__(self, data, algo, model_builder, process_comm, parent_comm, parent_rank=None, 
            num_epochs=1, verbose=False, monitor=False, custom_objects={},
            checkpoint=None, checkpoint_interval=5):
        """Raises an exception if no parent rank is provided. Sets the number of epochs 
            using the argument provided, then calls the parent constructor"""
        info = "Creating MPIWorker with rank {0} and parent rank {1} on a communicator of size {2}"
        tell_comm = parent_comm if parent_comm is not None else process_comm
        if tell_comm: logging.info(info.format(tell_comm.Get_rank(),
                           parent_rank,
                           tell_comm.Get_size()))

        super(MPIWorker, self).__init__( parent_comm, process_comm, parent_rank,
                num_epochs=num_epochs, data=data, algo=algo, model_builder=model_builder,
                verbose=verbose, monitor=monitor, custom_objects=custom_objects,
                checkpoint=checkpoint, checkpoint_interval=checkpoint_interval )

    def build_model(self):
        """Builds the Keras model and updates model-related attributes"""

        self.validation_model = self.model_builder.build_model()
        self.algo.compile_model( self.validation_model )
        super(MPIWorker, self).build_model(local_session=False)

    def sync_with_parent(self):
        self.compute_update()
        if self.algo.mode == 'gem':
            self.do_gem_sequence()
        else:
            self.do_send_sequence()

    def notify_parent(self):
        """Notify parent that worker is ready to begin GEM sequence."""
        self.send( None, 'begin_gem' )

    def do_gem_sequence(self):
        """GEM sequence on worker:
         -Notify parent we are ready
         -Receive central variable (weights) from master
         -Compute the update with GEM
         -Send the update to master and apply it to own weights
        """
        self.notify_parent()
        self.recv_weights()
        self.update = self.algo.compute_update_worker(self.weights, self.update)
        self.send_update()
        self.apply_update()
        self.algo.set_worker_model_weights(self.model, self.weights)

    def train(self):
        """Wait for the signal to train. Then train for num_epochs epochs.
            In each step, train on one batch of input data, then send the update to the master
            and wait to receive a new set of weights.  When done, send 'exit' signal to parent.
        """
        Trace.begin("train")
        self.check_sanity()

        self.await_signal_from_parent()

        # periodically check this request to see if the parent has told us to stop training
        exit_request = self.recv_exit_from_parent()
        for epoch in range(1, self.num_epochs + 1):
            self.logger.info("Beginning epoch {:d}".format(self.epoch + epoch))
            Trace.begin("epoch")
            if self.monitor:
                self.monitor.start_monitor()
            epoch_metrics = np.zeros((1,))
            i_batch = 0

            for i_batch, batch in enumerate(self.data.generate_data()):
                if self.process_comm:
                    ## broadcast the weights to all processes
                    ## alternative is to load the broadcast callback from horovod and call it here
                    self.bcast_weights( comm=self.process_comm )
                    if self.process_comm.Get_rank()!=0:
                        self.model.set_weights(self.weights)

                Trace.begin("train_on_batch")
                self.logger.debug("Beginning batch {:d}".format(i_batch))
                train_metrics = self.model.train_on_batch( x=batch[0], y=batch[1] )
                #time.sleep(0.5)
                self.logger.debug("Done with batch {:d}".format(i_batch))
                Trace.end("train_on_batch")
                if epoch_metrics.shape != train_metrics.shape:
                    epoch_metrics = np.zeros( train_metrics.shape)
                epoch_metrics += train_metrics
                if self.algo.should_sync():
                    self.logger.debug("train:: synching with parent")                    
                    self.sync_with_parent()
                    self.logger.debug("train:: synched with parent")
                if exit_request and exit_request.Test():
                    self.stop_training = True
                    if self.process_comm:
                        for r in range(1, self.process_comm.Get_size()):
                            ## propagate the exit signal to processes of this worker
                            self.send_exit_to_child(r, comm=self.process_comm)
                    self.logger.info("Received exit request from master")
                    break
                if self._short_batches and i_batch>self._short_batches: break

            if self.monitor:
                self.monitor.stop_monitor()
            epoch_metrics = epoch_metrics / float(i_batch+1)
            l = self.model.get_logs( epoch_metrics )
            self.update_history( l )
            Trace.end("epoch")

            if self.stop_training:
                break

        self.logger.debug("Signing off")
        stats = None
        if self.monitor:
            stats = self.monitor.get_stats()
            self.update_monitor( stats )
        Trace.end("train", monitoring_stats=stats)
        self.send_exit_to_parent()
        self.send_history_to_parent()
        self.data.finalize()

    @trace
    def compute_update(self):
        """Compute the update from the new and old sets of model weights"""
        self.update = self.algo.compute_update( self.weights, self.model.get_weights() )

    def await_signal_from_parent(self):
        """Wait for 'train' signal from parent process"""
        if not self.is_shadow():
            tag = self.recv( tag='train' )
        if self.process_comm:
            if self.process_comm.Get_rank() == 0:
                for r in range(1,self.process_comm.Get_size()):
                    self.send( None, tag='train', comm=self.process_comm, dest=r)
            else:
                self.recv( tag='train', comm=self.process_comm )



class MPIMaster(MPIProcess):
    """This class sends model information to its worker processes and updates its model weights
        according to updates or weights received from the workers.
        
        Attributes:
          child_comm: MPI intracommunicator used to communicate with child processes
          has_parent: boolean indicating if this process has a parent process
          num_workers: integer giving the number of workers that work for this master
          best_val_loss: best validation loss computed so far during training
          running_workers: list of workers not yet done training
          waiting_workers_list: list of workers that sent updates and are now waiting
          num_sync_workers: number of worker updates to receive before performing an update
          update_tag: MPI tag to expect when workers send updates
          histories: model histories collected from workers
          epoch: current epoch number
    """

    def __init__(self, parent_comm, parent_rank=None, child_comm=None, 
            num_epochs=1, data=None, algo=None, model_builder=None, 
                 num_sync_workers=1, verbose=False, monitor=False, custom_objects={}, early_stopping=None, target_metric=None,
                 threaded_validation=False, checkpoint=None, checkpoint_interval=5):
        """Parameters:
              child_comm: MPI communicator used to contact children"""
        if child_comm is None:
            raise Error("MPIMaster initialized without child communicator")
        self.child_comm = child_comm
        self.has_parent = False
        if parent_rank is not None:
            self.has_parent = True
        self.best_val_loss = None
        self.target_metric = (target_metric if type(target_metric)==tuple else tuple(map(lambda s : float(s) if s.replace('.','').isdigit() else s, target_metric.split(',')))) if target_metric else None
        self.patience = (early_stopping if type(early_stopping)==tuple else tuple(map(lambda s : float(s) if s.replace('.','').isdigit() else s, early_stopping.split(',')))) if early_stopping else None
        ## a couple of example
        ##self.target_metric = ('val_acc','>', 0.97)
        #self.patience = ('val_loss','~<',4)
        #self.target_metric = ('combined_model:val_loss','<',1)#('val_acc','>', 0.97)
        #self.patience = ('combined_model:val_loss','~<',2)        
        self.num_workers = child_comm.Get_size() - 1 #all processes but one are workers
        self.num_sync_workers = num_sync_workers
        info = ("Creating MPIMaster with rank {0} and parent rank {1}. "
                "(Communicator size {2}, Child communicator size {3})")
        logging.info(info.format(parent_comm.Get_rank(),parent_rank,parent_comm.Get_size(), 
                child_comm.Get_size()))
        if self.num_sync_workers > 1:
            logging.info("Will wait for updates from {0:d} workers before synchronizing".format(self.num_sync_workers))
        self.threaded_validation = threaded_validation

        super(MPIMaster, self).__init__( parent_comm, process_comm=None,parent_rank=parent_rank, data=data, 
                algo=algo, model_builder=model_builder, num_epochs=num_epochs, 
                verbose=verbose, monitor=monitor, custom_objects=custom_objects,
                checkpoint=checkpoint, checkpoint_interval=checkpoint_interval )

    def decide_whether_to_sync(self):
        """Check whether enough workers have sent updates"""
        return ( len(self.waiting_workers_list) >= self.num_sync_workers )

    def is_synchronous(self):
        return self.num_sync_workers > 1

    def accept_update(self):
        """Returns true if the master should accept the latest worker's update, false otherwise"""
        return (not self.is_synchronous()) or self.algo.staleness == 0
        
    def sync_children(self):
        """Update model weights and signal all waiting workers to work again.
            Send our update to our parent, if we have one"""
        while self.waiting_workers_list:
            child = self.waiting_workers_list.pop()
            self.sync_child(child)

    def sync_child(self, child):
        logging.debug("sync_child:: sending time step to {}".format(child))
        self.send_time_step( dest=child, comm=self.child_comm )
        logging.debug("sync_child:: sending weights to {}".format(child))        
        self.send_weights( dest=child, comm=self.child_comm )

    def sync_parent(self):
        if self.has_parent:
            self.do_send_sequence()
        else:
            self.time_step += 1 

    def do_update_sequence(self, source):
        """Update procedure:
         -Compute the staleness of the update and decide whether to accept it.
         -If we accept, we signal the worker and wait to receive the update.
         -After receiving the update, we determine whether to sync with the workers.
         -Finally we run validation if we have completed one epoch's worth of updates."""
        logging.debug("do_update_sequence:: receiving time step")
        child_time = self.recv_time_step( source=source, comm=self.child_comm )
        self.algo.staleness = self.time_step - child_time
        logging.debug("do_update_sequence:: checking on update")
        accepted = self.accept_update()
        logging.debug("do_update_sequence:: messaging accepted {} update".format(accepted))
        self.send_bool( accepted, dest=source, comm=self.child_comm )
        
        if accepted:
            logging.debug("do_update_sequence:: receiving the update")
            self.recv_update( source=source, comm=self.child_comm, 
                    add_to_existing=self.is_synchronous() )
            logging.debug("do_update_sequence:: received")
            self.waiting_workers_list.append(source)
            if self.decide_whether_to_sync():
                logging.debug("do_update_sequence:: synching with enough workers update")
                if self.algo.send_before_apply:
                    logging.debug("do_update_sequence:: sending before applying")
                    self.sync_parent()
                    logging.debug("do_update_sequence:: parent synched")
                    self.sync_children()
                    logging.debug("do_update_sequence:: child synched")
                    self.apply_update()
                    logging.debug("do_update_sequence:: update applied")
                else:
                    logging.debug("do_update_sequence:: applying before sending")
                    self.apply_update()
                    logging.debug("do_update_sequence:: update applied")
                    self.sync_parent()
                    logging.debug("do_update_sequence:: parent synched")
                    self.sync_children()
                    logging.debug("do_update_sequence:: child synched")
                self.update = self.model.format_update()
            logging.debug("do_update_sequence:: synched")
            if (self.algo.validate_every > 0 and self.time_step > 0):
                if (self.time_step % self.algo.validate_every == 0) or (self._short_batches and self.time_step%self._short_batches == 0):
                    self.validate(self.weights)
                    self.epoch += 1
                    self.save_checkpoint()
            logging.debug("do_update_sequence:: validated")
        else:
            self.sync_child(source)

    def do_gem_sequence(self, source):
        """Gradient enery matching procedure:
         -Send the current central variable (weights) to worker.
         -Wait to receive the update and apply it.
         -Finally we run validation if we have completed one epoch's worth of updates."""
        self.send_weights( dest=source, comm=self.child_comm )

    def do_gem_update_sequence(self, source):
        self.recv_update( source=source, comm=self.child_comm )
        self.apply_update()
        #Update model weights and signal all waiting workers to work again.
        self.time_step += 1
        if (self.algo.validate_every > 0 and self.time_step > 0):
            if (self.time_step % self.algo.validate_every == 0) or (self._short_batches and self.time_step%self._short_batches == 0):
                self.validate(self.weights)
                self.epoch += 1
                self.save_checkpoint()

    def do_worker_finish_sequence(self, worker_id):
        """Actions to take when a worker finishes training and returns its history"""
        self.histories.update( self.recv_history_from_child(worker_id) )
        #key = "%d_%d" % (self.rank, worker_id)
        #self.histories[key] = self.recv_history_from_child(worker_id)
        self.running_workers.remove(worker_id)
        self.num_sync_workers -= 1

    def process_message(self, status):
        """Extracts message source and tag from the MPI status object and processes the message. 
            Returns the tag of the message received.
            Possible messages are:
            -begin_update: worker is ready to send a new update
            -begin_gem: Worker needs central variable to start GEM
            -exit: worker is done training and will shut down
        """
        logging.debug("process_message:: getting source")
        source = status.Get_source()
        logging.debug("process_message:: getting tag from {}".format(source))
        tag = self.lookup_mpi_tag( status.Get_tag(), inv=True )
        logging.debug("process_message:: processing tag {}".format(tag))
        if tag == 'begin_update':
            if self.algo.mode == 'gem':
                self.do_gem_update_sequence(source)
            else:
                self.do_update_sequence(source)
        elif tag == 'begin_gem':
            self.do_gem_sequence(source)
        elif tag == 'exit':
            self.do_worker_finish_sequence(source)
        else:
            raise ValueError("Tag %s not recognized" % tag)
        return tag

    def shut_down_workers(self):
        """Signal all running workers to shut down"""
        for worker_id in self.running_workers:
            self.logger.info("Signaling worker {0:d} to shut down".format(worker_id))
            self.send_exit_to_child( worker_id )

    def train(self):
        """Broadcasts model information to children and signals them to start training.
            Receive messages from workers and processes each message until training is done.
            When finished, signal the parent process that training is complete.
        """
        Trace.begin("train")
        self.start_time = time.time()
        self.check_sanity()
        self.bcast_weights( comm=self.child_comm )
        self.signal_children()

        if self.threaded_validation:
            self.validation_queue = queue.Queue()
            self.validation_thread = threading.Thread(target=MPIMaster.validation_worker, args = (self, ))
            self.validation_thread.start()

        status = MPI.Status()
        self.running_workers = list(range(1, self.num_workers+1))
        self.waiting_workers_list = []
        
        while self.running_workers:
            logging.debug("MPIMaster::train:: receiving something from a child")
            self.recv_any_from_child(status)
            logging.debug("MPIMaster::train:: processing message")
            self.process_message( status )
            logging.debug("MPIMaster::train:: moving on")
            if (self.stop_training):
                self.shut_down_workers()
        self.logger.info("MPIMaster::train:: Done training")
        # If we did not finish the last epoch, validate one more time.
        # (this happens if the batch size does not divide the dataset size)
        if self.epoch < self.num_epochs or not self.histories.get(self.history_key(),None):
            epoch_logs = self.validate(self.weights)
        if self.threaded_validation:
            self.validation_queue.put(None)
            self.validation_queue.join()
            self.validation_thread.join()
        self.save_checkpoint()
        self.send_exit_to_parent()
        self.send_history_to_parent()
        self.data.finalize()
        self.stop_time = time.time()
        Trace.end("train")

    def record_details(self, json_name=None, meta=None):
        ## for the uber master, save yourself
        out_dict = { "train_time" : self.stop_time - self.start_time,
                     "history":self.histories}
        if meta is not None:
            out_dict.update( meta )

        if not json_name:
            json_name = 'master-history-%s-%s.json'%(os.getpid(), self.start_time)

        if self.model:
            model_file = json_name.replace('.json','.model')
            self.model.save( model_file )

        if self.algo:
            algo_file = json_name.replace('.json','.algo')
            self.algo.save( algo_file )

        with open( json_name, 'w') as out_file:
            out_file.write( json.dumps(out_dict, indent=4, separators=(',',': ')) )

        return self.histories

    def validation_worker(self):
        """Main function of the validation thread"""
        self.logger.debug("Validation thread started")
        while True:
            item = self.validation_queue.get()
            if item is None:
                self.logger.debug("Validation thread signing off")
                self.validation_queue.task_done()
                break
            weights, model = item
            try:
                self.validate_aux(weights, model)
            except Exception as e:
                self.logger.error(e)
            self.validation_queue.task_done()

    def validate(self, weights):
        if self.threaded_validation:
            model = self.validation_model
            self.validation_queue.put((weights, model))
        else:
            return self.validate_aux(weights, self.model)
        
    
    def validate_aux(self, weights, model):
        """Compute the loss on the validation data.
            Return a dictionary of validation metrics."""
        Trace.begin("validation", "VALIDATION")
        if self.has_parent:
            return {}
        model.set_weights(weights)

        self.logger.debug("Starting validation")
        val_metrics = np.zeros((1,))
        i_batch = 0
        for i_batch, batch in enumerate(self.data.generate_data()):
            new_val_metrics =  model.test_on_batch(x=batch[0], y =batch[1] )
            if val_metrics.shape != new_val_metrics.shape:
                val_metrics =  np.zeros(new_val_metrics.shape)
            val_metrics += new_val_metrics
            if self._short_batches and i_batch>self._short_batches: break
        val_metrics = val_metrics / float(i_batch+1)
        l = model.get_logs(val_metrics, val=True)
        self.update_history( l )

        if self.target_metric:
            m,opp,v = self.target_metric
            model,m= m.split(':') if ':' in m else (None,m)
            r = self.history_key()
            use = self.histories[r].get(model,None) if model else self.histories[r]
            if use:
                if m in use and ((opp=='>' and use[m][-1]>v) or (opp=='<' and use[m][-1]<v)):
                    self.logger.debug("metric %s is %s %s, stop training",str(m),str(opp), str(v))
                    self.stop_training = True
            else:
                self.logger.error("fatal target stopping cannot get %s", str(m))
                MPI.COMM_WORLD.Abort()
                
        if self.patience:
            m,opp,p = self.patience
            p = int(p)
            model,m=m.split(':') if ':' in m else (None,m)
            r = self.history_key()            
            use = self.histories[r].get(model,None) if model else self.histories[r]
            ref = None
            smooth = False
            if use:
                if m in use:
                    if '~' in opp:
                        smooth = True
                        opp = opp.replace('~','')
                        ## do the averaging
                        if len(use[m])>=(2*p):
                            ref = np.mean(use[m][-(2*p):-p])
                            current = np.mean(use[m][-p:])
                        else:
                            if len(use[m])>=p:
                                ref = use[m][-p]
                                current = use[m][-1]
                                current = None
                    if ref is not None and current is not None and ((ref<current and opp=='<') or (ref>current and opp=='>')):
                        self.logger.info("metric %s is over %s patience boundary: %s (ref) %s %s (current) %s", str(m), str(p), str(ref), str(opp), str(current), "with smoothing" if smooth else "")
                        self.stop_training = True
                else:
                    self.logger.error("fatal early stopping cannot get %s", str(m))
                    MPI.COMM_WORLD.Abort()
            else:
                self.logger.error("fatal early stopping cannot get %s", str(m))
                MPI.COMM_WORLD.Abort()
        self.logger.info("Validation metrics:")
        self.print_metrics(val_metrics)
        self.logger.debug("Ending validation")
        Trace.end("validation", "VALIDATION")
        return None

    ### MPI-related functions below

    def signal_children(self):
        """Sends each child a message telling them to start training"""
        for child in range(1, self.child_comm.Get_size()):
            self.send( obj=None, tag='train', dest=child, comm=self.child_comm )

    def recv_any_from_child(self,status):
        """Receives any message from any child.  Returns the provided status object,
            populated with information about received message"""
        logging.debug("recv_any_from_child:: recv")
        self.recv( tag='any', source=MPI.ANY_SOURCE, status=status, comm=self.child_comm )
        logging.debug("recv_any_from_child:: + recv")
        return status

    def recv_history_from_child(self, child):
        return self.recv( tag='history', source=child, comm=self.child_comm )

    def send_exit_to_child(self, child, comm=None):
        if comm is None:
            comm = self.child_comm
        return comm.isend( None, dest=child, tag=self.lookup_mpi_tag('exit') )
    
    def build_model(self):
        """Builds the Keras model and updates model-related attributes"""

        if self.threaded_validation:
            self.validation_model = self.model_builder.build_model()
            self.algo.compile_model( self.validation_model )
        super(MPIMaster, self).build_model(local_session=self.threaded_validation)
