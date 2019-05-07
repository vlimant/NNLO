import os
import psutil
import time
import json
from functools import wraps
from mpi4py import MPI

def timeline(original_function=None, category=None, **decorator_kwargs):
    """Decorates a function. Can be called with or without additional arguments. Name of event is the name of decorated function.
        Params:
          category: Optional category of event (useful to show/hide category of events in timeline viewer)
          decorator_kwargs: Additional arguments
    """
    def _decorate(function):
        @wraps(function)
        def wrapped_function(*args, **kwargs):
            Timeline.begin(function.__name__, category, **decorator_kwargs)
            if len(args) == 1 and not kwargs and callable(args[0]):
                ret_val = function()(args[0])
            else:
                ret_val = function(*args, **kwargs)
            Timeline.end(function.__name__, category, **decorator_kwargs)
            return ret_val

        return wrapped_function

    if original_function:
        return _decorate(original_function)

    return _decorate

class Timeline(object):
    """ Class that traces events to display on timeline """
    _enabled = False
    _events = []
    _process_name = os.getpid()
    _process_file = None
    _flush_every = 100

    @classmethod
    def _record(cls, event_name, type, category=None, **kwargs):
        if cls._enabled:
            ts = int(round(time.time() * 1000000))
            if category is None: category = "TRACE"
            tid = kwargs.get("tid", "Main")
            
            event = {"name": event_name, "cat": category, "ph": type, "pid": cls._process_name, "tid": tid, "ts": ts, "args": kwargs}
            
            cls._events.append(event)

            # Write events to process-specific file (in case collect() is never called)
            if cls._flush_every > 0 and len(cls._events) % cls._flush_every == 0:
                with open(cls._process_file, 'a+') as timeline_file:
                    timeline_file.write(",\n".join(map(json.dumps, cls._events[-cls._flush_every:])))

    @classmethod
    def begin(cls, name, category=None, **kwargs):
        """Marks the beginning of an event.
            Params:
              name: Name of event
              category: Optional category of event (useful to show/hide category of events in trace viewer)
              kwargs: Additional arguments
        """
        cls._record(name, "B", category, **kwargs)
    
    @classmethod
    def end(cls, name, category=None, **kwargs):
        """Marks the end of an event.
            Params:
              name: Name of event
              category: Optional category of event (useful to show/hide category of events in trace viewer)
              kwargs: Additional arguments
        """
        cls._record(name, "E", category, **kwargs)

    @classmethod
    def enable(cls, flush_file=None, flush_every=100):
        """Enables collection of timeline events.
            Params:
              flush_file: name of per-process timeline file to temporarily write events to, if None, 'str(os.getpid()) + _timeline.json' is used
              flush_every: Number of events to collect before flushing them to temporary file. 0 disables writing.
        """
        cls._enabled = True
        cls._process_file = flush_file or str(os.getpid()) + "_timeline.json"
        cls._flush_every = flush_every

    @classmethod
    def set_process_name(cls, process_name):
        """Sets a user-defined name for the process in timeline, as opposed to using just pid of process
            Params:
              process_name: Name to use
        """
        cls._process_name = process_name

    @classmethod
    def collect(cls, file_name=None, clean=False, comm=None):
        """Collects events from processes and writes them to a timeline file.
            Params:
              file_name: name of timeline file to write events to, if None, 'nnlo_timeline.json' is used
              clean: If True, removes process-specific temporary timeline files
              comm: MPI communicator to use, if None, MPI.COMM_WORLD is used 
        """
        if not cls._enabled:
            return

        comm = comm or MPI.COMM_WORLD
        all_events = comm.gather(cls._events, root=0)

        if (comm.Get_rank() == 0):
            with open(file_name or "nnlo_timeline.json", 'w+') as master_file:
                master_file.write("[\n")
                flat_events = []
                for events in all_events:
                    flat_events.append(",\n".join(map(json.dumps, events)))
                master_file.write(",\n".join(flat_events))
                master_file.write("\n]\n")

        if clean and os.path.isfile(cls._process_file):
            os.remove(cls._process_file)
