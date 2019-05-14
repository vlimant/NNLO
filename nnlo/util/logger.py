from mpi4py import MPI
import time
import logging
from os.path import abspath
from functools import wraps
import inspect
import importlib
import numpy as np
import re

TRACE = logging.DEBUG - 5

level_map = {
    'trace': TRACE,
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warn': logging.WARNING,
    'error': logging.ERROR,
}

base_prefix = "{ptype} {world}:{parent}:{process}"
file_handler = None
stream_handler = None

start_time = time.time()

def addLoggingLevel(levelName, levelNum, methodName=None):
    """
    Comprehensively adds a new logging level to the `logging` module and the
    currently configured logging class.

    `levelName` becomes an attribute of the `logging` module with the value
    `levelNum`. `methodName` becomes a convenience method for both `logging`
    itself and the class returned by `logging.getLoggerClass()` (usually just
    `logging.Logger`). If `methodName` is not specified, `levelName.lower()` is
    used.
    """
    if not methodName:
        methodName = levelName.lower()

    if hasattr(logging, levelName):
       raise AttributeError('{} already defined in logging module'.format(levelName))
    if hasattr(logging, methodName):
       raise AttributeError('{} already defined in logging module'.format(methodName))
    if hasattr(logging.getLoggerClass(), methodName):
       raise AttributeError('{} already defined in logger class'.format(methodName))

    def logForLevel(self, message, *args, **kwargs):
        if self.isEnabledFor(levelNum):
            self._log(levelNum, message, args, **kwargs)
    def logToRoot(message, *args, **kwargs):
        logging.log(levelNum, message, *args, **kwargs)

    logging.addLevelName(levelNum, levelName)
    setattr(logging, levelName, levelNum)
    setattr(logging.getLoggerClass(), methodName, logForLevel)
    setattr(logging, methodName, logToRoot)

class ElapsedTimeFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        global start_time
        total_millis = int(record.created - start_time) * 1000 + int(record.msecs)

        millis = total_millis%1000
        millis = int(millis)
        seconds=(total_millis/1000)%60
        seconds = int(seconds)
        minutes=(total_millis/(1000*60))%60
        minutes = int(minutes)
        hours=(total_millis/(1000*60*60))%24
        hours = int(hours)

        elapsed = "{:04d}:{:02d}:{:02d}.{:03d}".format(hours, minutes, seconds, millis)
        return elapsed

class MPIFileHandler(logging.FileHandler):
    def __init__(self,
                 filename,
                 mode=MPI.MODE_WRONLY|MPI.MODE_CREATE|MPI.MODE_APPEND ,
                 encoding='utf-8',
                 delay=False,
                 comm=MPI.COMM_WORLD):
        self.baseFilename = abspath(filename)
        self.mode = mode
        self.encoding = encoding
        self.comm = comm.Dup()
        if delay:
            #We don't open the stream, but we still need to call the
            #Handler constructor to set level, formatter, lock etc.
            logging.Handler.__init__(self)
            self.stream = None
        else:
           logging.StreamHandler.__init__(self, self._open())

    def _open(self):
        stream = MPI.File.Open( self.comm, self.baseFilename, self.mode )
        stream.Set_atomicity(True)
        return stream

    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            stream.Write_shared((msg+self.terminator).encode(self.encoding))
            #self.flush()
        except Exception:
            self.handleError(record)

    def close(self):
        if self.stream:
            self.stream.Sync()
            self.stream.Close()
            self.stream = None

def initialize_logger(filename=None, file_level='info', stream=True, stream_level='info'):
    global file_handler
    global stream_handler
    addLoggingLevel('TRACE', TRACE)
    logger = logging.getLogger()
    logger.setLevel(min(get_log_level(file_level), get_log_level(stream_level))) # Lowest level that will be propagated to handlers
    logger.handlers = []
    if filename is not None:
        file_handler = MPIFileHandler(filename)
        file_handler.setLevel(get_log_level(file_level))
        logger.addHandler(file_handler)
    if stream:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(get_log_level(stream_level))
        logger.addHandler(stream_handler)
    set_logging_prefix(MPI.COMM_WORLD.rank)
    if file_level == 'trace' or stream_level == 'trace':
        # Disable tracing of optimize for now
        add_trace_decorators(train=True, mpi=True, optimize=False)

def get_log_level(levelstr='info'):
    levelstr = levelstr.lower()
    return level_map[levelstr]

def set_logging_prefix(world_rank, parent_rank='-', process_rank='-', process_type='P'):
    global file_handler
    global stream_handler
    prefix = base_prefix.format(ptype=process_type, world=world_rank, parent=parent_rank, process=process_rank)
    formatter = ElapsedTimeFormatter('%(asctime)s ' + prefix + ' [%(levelname)s] %(message)s')
    if file_handler is not None:
        file_handler.setFormatter(formatter)
    if stream_handler is not None:
        stream_handler.setFormatter(formatter)

def add_trace_decorators(train=True, mpi=True, optimize=False):
    train_module_names = ['algo', 'data', 'model', 'optimizer']
    mpi_module_names = ['manager', 'process', 'single_process']
    optimize_module_names = ['coordinator', 'process_block']

    if train:
        for mod_name in train_module_names:
            mod = importlib.import_module('..train.' + mod_name, package=__name__)
            _decorate_module(mod)

    if mpi:
        for mod_name in mpi_module_names:
            mod = importlib.import_module('..mpi.' + mod_name, package=__name__)
            _decorate_module(mod)

    #if optimize:
    #    for mod_name in optimize_module_names:
    #        mod = importlib.import_module('..optimize.' + mod_name, package=__name__)
    #        _decorate_module(mod)

def _decorate_module(mod):
    # Module-level functions
    tl_func_list = inspect.getmembers(mod, inspect.isfunction)
    for name, func in tl_func_list:
        skip = False
        for regex in skip_trace:
            if re.search(regex, func.__qualname__) is not None:
                skip = True
                break
        if skip:
            logging.log(TRACE, "Skipping tracing of {}".format(func.__qualname__))
        else:
            setattr(mod, name, _trace(func))
    # Class functions
    cls_list = inspect.getmembers(mod, inspect.isclass)
    for _, clazz in cls_list:
        cls_func_list = inspect.getmembers(clazz, lambda x: inspect.isfunction(x) or inspect.ismethod(x))
        logging.log(TRACE, "Class {} function list {}".format(clazz.__name__, cls_func_list))
        for fname, func in cls_func_list:
            skip = False
            for regex in skip_trace:
                if re.search(regex, func.__qualname__) is not None:
                    skip = True
                    break
            if skip:
                logging.log(TRACE, "Skipping tracing of {}".format(func.__qualname__))
            else:
                setattr(clazz, fname, _trace(func))


skip_trace = [
    'Monitor.*',
    'Timeline.*',
    'Data.is_numpy_array',
    'MPIProcess.lookup_mpi_tag',
    'Thread.*'
]

def _trace(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        def str_type(obj):
            if type(obj) == list:
                return 'list[{}]'.format(len(obj))
            elif isinstance(obj, np.ndarray):
                return 'ndarray[{}]'.format(obj.shape)
            else:
                return str(obj)
        real_args = args[1:] if '.' in function.__qualname__ else args
        print_args = [str_type(arg) for arg in real_args]
        print_kwargs = ['{}={}'.format(k, str_type(v)) for k, v in kwargs.items()]
        params = ', '.join(print_args + print_kwargs)
        logging.log(TRACE, '>>> {}.{}({})'.format(function.__module__, function.__qualname__, params))

        ret_val = function(*args, **kwargs)

        logging.log(TRACE, '<<< {}.{}'.format(function.__module__, function.__qualname__))
        return ret_val

    return wrapper
