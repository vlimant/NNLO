### Data class and associated helper methods

import numpy as np
import pandas as pd
import h5py
import os
import time
from threading import Thread
import itertools
import logging

class FilePreloader(Thread):
    def __init__(self, files_list, file_open,n_ahead=2):
        Thread.__init__(self)
        self.deamon = True
        self.n_concurrent = n_ahead
        self.files_list = files_list
        self.file_open = file_open
        self.loaded = {} ## a dict of the loaded objects
        self.should_stop = False
        
    def getFile(self, name):
        ## locks until the file is loaded, then return the handle
        return self.loaded.setdefault(name, self.file_open( name))

    def closeFile(self,name):
        ## close the file and
        if name in self.loaded:
            self.loaded.pop(name).close()
    
    def run(self):
        while not self.files_list:
            time.sleep(1)
        for name in itertools.cycle(self.files_list):
            if self.should_stop:
                break
            n_there = len(self.loaded.keys())
            if n_there< self.n_concurrent:
                logging.debug("preloading {} with {}".format(name,n_there))
                self.getFile( name )
            else:
                time.sleep(5)

    def stop(self):
        logging.debug("Stopping FilePreloader")
        self.should_stop = True

def data_class_getter(name):
    """Returns the specified Data class"""
    data_dict = {
            "H5Data":H5Data,
            }
    try:
        return data_dict[name]
    except KeyError:
        logging.warning("{0:s} is not a known Data class. Returning None...".format(name))
        return None


class Data(object):
    """Class providing an interface to the input training data.
        Derived classes should implement the load_data function.

        Attributes:
          file_names: list of data files to use for training
          batch_size: size of training batches
    """
    def finalize(self):
        if self.caching_directory:            
            for fn in self.relocated:
                logging.debug("removing cached file {}".format(fn))
                os.system('rm -f {}'.format(fn))

    def __init__(self, batch_size, cache=None, copy_command=None):
        """Stores the batch size and the names of the data files to be read.
            Params:
              batch_size: batch size for training
        """
        self.batch_size = batch_size
        self.caching_directory = cache if cache else os.environ.get('DATA_CACHE','')
        self.copy_command = copy_command if copy_command else os.environ.get('DATA_COPY_COMMAND','cp {} {}')
        ## for regular copy it is "cp {} {}"
        ## for s3 it should be "s3cmd get s3://gan-bucket/{} {}"
        ## for xrootd it should be "xrdcp root://cms-xrd-global.cern.ch/{} {}", assuming a valid proxy
        self.fpl = None

    def set_caching_directory(self, cache):
        self.caching_directory = cache
        
    def set_full_file_names(self, file_names):
        self.file_names = list(filter(None, file_names))

    def set_file_names(self, file_names):
        new_file_names = []
        self.relocated = []
        if self.caching_directory:
            goes_to = self.caching_directory
            goes_to += "/"+str(os.getpid())
            os.system('mkdir -p %s '%goes_to)
            for fn in file_names:
                relocate = goes_to+'/'+fn.split('/')[-1]
                if not os.path.isfile( relocate ):
                    logging.debug("Copying {} to {}".format( fn , relocate))
                    cmd = self.copy_command.format( fn, relocate )
                    if os.system(cmd)==0:
                        new_file_names.append( relocate )
                        self.relocated.append( relocate )
                    else:
                        logging.error("was unable to copy the file with {}".format( cmd ))
                        new_file_names.append( fn ) ## use the initial one
                else:
                    new_file_names.append( relocate )
                    self.relocated.append( relocate )
                        
            self.file_names = new_file_names
        else:
            self.file_names = file_names

        if self.fpl:
            self.fpl.files_list = self.file_names

    def inf_generate_data(self):
        while True:
            try:
                for B in self.generate_data():
                    yield B
            except StopIteration:
                logging.warning("start over generator loop")
                
    def generate_data(self):
       """Yields batches of training data until none are left."""
       leftovers = None
       for cur_file_name in self.file_names:
           cur_file_features, cur_file_labels = self.load_data(cur_file_name)
           # concatenate any leftover data from the previous file
           if leftovers is not None:
               cur_file_features = self.concat_data( leftovers[0], cur_file_features )
               cur_file_labels = self.concat_data( leftovers[1], cur_file_labels )
               leftovers = None
           num_in_file = self.get_num_samples( cur_file_features )

           for cur_pos in range(0, num_in_file, self.batch_size):
               next_pos = cur_pos + self.batch_size 
               if next_pos <= num_in_file:
                   yield ( self.get_batch( cur_file_features, cur_pos, next_pos ),
                           self.get_batch( cur_file_labels, cur_pos, next_pos ) )
               else:
                   leftovers = ( self.get_batch( cur_file_features, cur_pos, num_in_file ),
                                 self.get_batch( cur_file_labels, cur_pos, num_in_file ) )

    def count_data(self):
        """Counts the number of data points across all files"""
        num_data = 0
        for cur_file_name in self.file_names:
            cur_file_features, cur_file_labels = self.load_data(cur_file_name)
            num_data += self.get_num_samples( cur_file_features )
        return num_data

    def is_numpy_array(self, data):
        return isinstance( data, np.ndarray )

    def get_batch(self, data, start_pos, end_pos):
        """Input: a numpy array or list of numpy arrays.
            Gets elements between start_pos and end_pos in each array"""
        if self.is_numpy_array(data):
            return data[start_pos:end_pos] 
        else:
            return [ arr[start_pos:end_pos] for arr in data ]

    def concat_data(self, data1, data2):
        """Input: data1 as numpy array or list of numpy arrays.  data2 in the same format.
           Returns: numpy array or list of arrays, in which each array in data1 has been
             concatenated with the corresponding array in data2"""
        if self.is_numpy_array(data1):
            return np.concatenate( (data1, data2) )
        else:
            return [ self.concat_data( d1, d2 ) for d1,d2 in zip(data1,data2) ]

    def get_num_samples(self, data):
        """Input: dataset consisting of a numpy array or list of numpy arrays.
            Output: number of samples in the dataset"""
        if self.is_numpy_array(data):
            return len(data)
        else:
            return len(data[0])

    def load_data(self, in_file):
        """Input: name of file from which the data should be loaded
            Returns: tuple (X,Y) where X and Y are numpy arrays containing features 
                and labels, respectively, for all data in the file

            Not implemented in base class; derived classes should implement this function"""
        raise NotImplementedError

        
        
class FrameData(Data):
    """ Load pandas frame stored in hdf5 files """
    def __init__(self, batch_size,
                 feature_adaptor,
                 cache=None,
                 copy_command=None,
                 preloading=0,
                 frame_name='frame', 
                 labels_name='label'):
        """Initializes and stores names of feature and label datasets"""
        super(FrameData, self).__init__(batch_size,cache,copy_command)
        self.feature_adaptor = feature_adaptor
        self.frame_name = frame_name
        self.labels_name = labels_name
        ## initialize the data-preloader
        self.fpl = None   
        
    def load_data(self, in_file_name):
        frame = pd.read_hdf(in_file_name, self.frame_name)
        return frame        
           
    def count_data(self):
        num_data = 0
        for in_file_name in self.file_names:
            frame = pd.read_hdf(in_file_name, self.frame_name)
            num_data += len(frame)
        return num_data        

    def concat_data(self, data1, data2):
        return pd.concat([data1, data2])

    def generate_data(self):
        """ 
        Overwrite the the parent generate_data and adapt to pandas frames
        """
        leftovers = None
        for cur_file_name in self.file_names:
            cur_frame = self.load_data(cur_file_name)
            # concatenate any leftover data from the previous file
            if leftovers is not None:
                cur_frame = self.concat_data( leftovers, cur_frame )
                leftovers = None
            num_in_file = len(cur_frame)
            for cur_pos in range(0, num_in_file, self.batch_size):
                next_pos = cur_pos + self.batch_size 
                if next_pos <= num_in_file:
                    yield ( self.get_batch( cur_frame, cur_pos, next_pos ), 
                            cur_frame[self.labels_name].iloc[cur_pos : next_pos].values)
                else:
                    leftovers = cur_frame.iloc[cur_pos : num_in_file]   
              
    def get_batch(self, cur_frame, start_pos, end_pos ):
        """ 
        Convert the batch of the dataframe to a numpy array
        with the provided function
        """
        #print( 'Gen batch' )
        batch = cur_frame.iloc[start_pos : end_pos]
        return self.feature_adaptor( batch )

    def finalize(self):
        if self.fpl:
            self.fpl.stop()
        Data.finalize(self)
    
        
        
class H5Data(Data):
    """Loads data stored in hdf5 files
        Attributes:
          features_name, labels_name: names of the datasets containing the features
          and labels, respectively
    """
    def __init__(self, batch_size,
                 cache=None,
                 copy_command=None,
                 preloading=0,
                 features_name='features', labels_name='labels'):
        """Initializes and stores names of feature and label datasets"""
        super(H5Data, self).__init__(batch_size,cache,copy_command)
        self.features_name = features_name
        self.labels_name = labels_name
        ## initialize the data-preloader
        self.fpl = None
        if preloading:
            self.fpl = FilePreloader( [] , file_open = lambda n : h5py.File(n,'r'), n_ahead=preloading)
            self.fpl.start()          
       

    def load_data(self, in_file_name):
        """Loads numpy arrays from H5 file.
            If the features/labels groups contain more than one dataset,
            we load them all, alphabetically by key."""
        if self.fpl:
            h5_file = self.fpl.getFile( in_file_name )
        else:
            h5_file = h5py.File( in_file_name, 'r' )
        if type(self.features_name) == tuple:
            ## there is a data adaptor
            feature_name, feature_adaptor = self.features_name
        else:
            feature_adaptor = None
            feature_name = self.features_name
        if type(self.labels_name) == tuple:
            ## there is a data adaptor
            label_name, label_adaptor = self.labels_name
        else:
            label_adaptor = None
            label_name = self.labels_name
        
        X = self.load_hdf5_data( h5_file[feature_name] )
        Y = self.load_hdf5_data( h5_file[label_name] )
        if feature_adaptor is not None:
            X = feature_adaptor(X)
        if label_adaptor is not None:
            Y = label_adaptor(Y)
        if self.fpl:
            self.fpl.closeFile( in_file_name )
        else:
            h5_file.close()
        return X,Y 

    def load_hdf5_data(self, data):
        """Returns a numpy array or (possibly nested) list of numpy arrays 
            corresponding to the group structure of the input HDF5 data.
            If a group has more than one key, we give its datasets alphabetically by key"""
        if hasattr(data, 'keys'):
            out = [ self.load_hdf5_data( data[key] ) for key in sorted(data.keys()) ]
        else:
            out = data[:]
        return out

    def count_data(self):
        """This is faster than using the parent count_data
            because the datasets do not have to be loaded
            as numpy arrays"""
        num_data = 0
        for in_file_name in self.file_names:
            h5_file = h5py.File( in_file_name, 'r' )
            if type(self.features_name) == tuple:
                feature_name = self.features_name[0]
            else:
                feature_name = self.features_name
            X = h5_file[feature_name]
            if hasattr(X, 'keys'):
                num_data += len(X[ X.keys()[0] ])
            else:
                num_data += len(X)
            h5_file.close()
        return num_data

    def finalize(self):
        if self.fpl:
            self.fpl.stop()
        Data.finalize(self)
