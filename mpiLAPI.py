#!/usr/bin/env python3

import os
import keras
import glob
import h5py
import hashlib
import time
from densenet import DenseNet
from keras.optimizers import Adam
from argparse import ArgumentParser
from subprocess import check_output,call,getoutput,Popen
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf
import numpy as np
import keras.backend as K

class mpi_learn_api:
    def __init__(self, **args):
        if not os.path.isdir('./tmp'): 
            print("creating directory")
            os.makedirs('./tmp')
        if not 'nohash' in args:
            args['check'] = time.mktime(time.gmtime())
            hash = hashlib.md5(str(args).encode('utf-8')).hexdigest()
            self.json_file = './tmp/%s.json'% hash
            #print("self.jsonfile = {}".format(self.json_file))
            if os.path.isfile( self.json_file ) :
                print("hash",hash,"cannot work")
                sys.exit(1)
            self.train_files = 'tmp/%s_train.list'%hash
            self.val_files = 'tmp/%s_val.list'%hash
        else:
            self.train_files = 'tmp/train.list'
            self.val_files = 'tmp/val.list'
            if not 'model_name' in args: 
                self.json_file = 'tmp/tmp.json'
            else:
                self.json_file = 'tmp/{}.json'.format(args['model_name'])
        open(self.json_file,'w').write(args['model'])
        if 'train_files' in args:
            open(self.train_files,'w').write( '\n'.join(args['train_files']))
        elif 'train_pattern' in args:
            a_list = sorted(glob.glob( args['train_pattern']))
            if args.get('check_file',False): a_list = self._check_files(a_list)
            open(self.train_files,'w').write( '\n'.join( a_list ))
        else:
            self.train_files = args['train_list']

        if 'val_files' in args:
            open(self.val_files,'w').write( '\n'.join(args['val_files']))
        elif 'val_pattern' in args:
            a_list = sorted(glob.glob(args['val_pattern']))
            if args.get('check_file',False): a_list = self._check_files(a_list)
            open(self.val_files,'w').write( '\n'.join( a_list ))
        else:
            self.val_files = args['val_list']

    def _check_files(self, a_list):
        for fn in sorted(a_list):
            try:
                f = h5py.File(fn)
                l = sorted(f.keys())
                assert len(l)>1
                f.close()
            except:
                print(fn,"not usable")
                a_list.remove(fn)
        return a_list
    
    def train(self, **args):
        com = 'mpirun -n %d mpi_learn/MPIDriver.py %s %s %s'%(
            args.get('N', 2),
            self.json_file,
            self.train_files,
            self.val_files
        )
        for option,default in { 'trial_name' : 'mpi_run',
                                 'master_gpu' : True,
                                 'features_name' : 'X',
                                 'labels_name' : 'Y',
                                 'epoch' : 100,
                                 'batch' : 100,
                                 'loss' : 'categorical_crossentropy',
                                 'verbose': False,
                                 'early_stopping' : False,
                                 'easgd' : False,
                                 'tf': True,
                                 'elastic_force': 0.9,
                                 'elastic_momentum': 0.99,
                                 'elastic_lr':0.001,
                                 }.items():
            v = args.get(option,default)
            if type(v)==bool:
                com +=' --%s'%option.replace('_','-') if v else ''
            else:
                com+=' --%s %s'%(option.replace('_','-'), v)
        print(com)
        return getoutput(com)

    def train_async(self, get_output=True, **args):
        com = 'mpirun -n %d mpi_learn/MPIDriver.py %s %s %s'%(
            args.get('N', 2),
            self.json_file,
            self.train_files,
            self.val_files
        )
        for option,default in { 'trial_name' : 'mpi_run',
                                 'master_gpu' : True,
                                 'features_name' : 'X',
                                 'labels_name' : 'Y',
                                 'epoch' : 100,
                                 'batch' : 100,
                                 'loss' : 'categorical_crossentropy',
                                 'verbose': False,
                                 'early_stopping' : False,
                                 'easgd' : False,
                                 'tf': True,
                                 'elastic_force': 0.9,
                                 'elastic_momentum': 0.99,
                                 'elastic_lr':0.001,
                                 }.items():
            v = args.get(option,default)
            if type(v)==bool:
                com +=' --%s'%option.replace('_','-') if v else ''
            else:
                com+=' --%s %s'%(option.replace('_','-'), v)
        print(com)
        if not get_output:
            import tempfile
            tfil = tempfile.TemporaryFile()
            return Popen(com, shell=True, stdout=tfil, stderr=tfil)
        else:
            return Popen(com, shell=True)

def test_mnist(**args):
    """MNIST ConvNet from keras/examples/mnist_cnn.py"""
    nb_classes = 10
    # input image dimensions
    img_rows, img_cols = 28, 28
    # number of convolutional filters to use
    nb_filters = args.get('nb_filters',32)
    # size of pooling area for max pooling
    ps = args.get('pool_size',2)
    pool_size = (ps,ps)
    # convolution kernel size
    ks = args.get('kernel_size',3)
    kernel_size = (ks, ks)
    do = args.get('drop_out', 0.25)
    dense = args.get('dense', 128)
    if K.image_dim_ordering() == 'th':
        input_shape = (1, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, 1)
    model = Sequential()
    #model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
    ## suggested Conv2D(23, padding="valid", kernel_size=(2, 2), input_shape=(28, 28, 1...)`
    model.add(Conv2D(nb_filters, kernel_size = kernel_size,
                     #order_mode='valid',
                     padding="valid",
                     input_shape=input_shape))
    model.add(Activation('relu'))
    #model.add(Conv2D(nb_filters, kernel_size[0], kernel_size[1]))
    model.add(Conv2D(nb_filters, kernel_size = kernel_size))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(do))
    model.add(Flatten())
    model.add(Dense(dense))
    model.add(Activation('relu'))
    model.add(Dropout(do))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    return model.to_json()

def test_cnn(**args):
    dropout= args.get('dropout',0.5)
    kernel_size = args.get('kernel_size',3)
    lr = args.get('lr', np.exp(-args.get('llr',np.log(1./1e-3))))
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(kernel_size, kernel_size), strides=3, activation='relu', input_shape=(150,94,5)))
    model.add(Conv2D(32, kernel_size=(kernel_size,kernel_size), strides=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(dropout/2))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer = Adam(lr))
    
    return model.to_json()

def test_densenet(nb_classes = 3, img_dim = (150, 94, 5), depth = 10, nb_dense_block = 3, growth_rate = 12, dropout_rate= 0.00, nb_filter = 16, lr = 1e-3):
    densenet = DenseNet(nb_classes = nb_classes, img_dim = img_dim, depth = depth, nb_dense_block = nb_dense_block, growth_rate = growth_rate, dropout_rate = dropout_rate, nb_filter = nb_filter)
    optimizer = Adam(lr = lr)
    densenet.compile(loss='categorical_crossentropy', optimizer = optimizer)
    return densenet.to_json()

def test_pytorch_cnn(conv_layers=2, dense_layers=2, dropout=0.5, classes=3, in_channels=5):
    from PytorchCNN import CNN
    import getpass
    pytorch_cnn = CNN(conv_layers=conv_layers, dense_layers=dense_layers, dropout=dropout, classes=classes, in_channels=in_channels)
    PATH = "/tmp/{}/test_pytorch_cnn.torch".format(getpass.getuser()) # To be determined where is the best location to save it
    torch.save(pytorch_cnn, PATH) 
    return PATH

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--blocks", help = "Number of dense blocks", type=int, default=12)
    parser.add_argument("--growth", help = "Growth rate", type=int, default=12)
    parser.add_argument("--dropout", help = "Dropout rate", type=float, default = 0)
    parser.add_argument("--filters", help = "Number of filters", type = int, default = 16)
    parser.add_argument("--lr", help = "Initial learning rate", type = float, default = 1e-3)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    depth = args.blocks * 3 + 4
    print("Model depth = {}".format(depth))
    from keras.models import model_from_json
    
    # os.environ["CUDA_VISIBLE_DEVICES"]="0,3,4,5"
    # import setGPU

    
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    #sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=True))
    #sess.close()
    

    #model = test_densenet(depth = depth, growth_rate = args.growth, dropout_rate = args.dropout, nb_filter = args.filters, lr = args.lr)
    model = test_cnn()

    mlapi = mpi_learn_api( model = model,
                           train_pattern = '/bigdata/shared/LCDJets_Remake/train/04*.h5',
                           val_pattern = '/bigdata/shared/LCDJets_Remake/val/020*.h5',
                           check_file = True
                           )
   
    output = mlapi.train(N=1,
                trial_name = 'test',
                features_name = 'Images',
                labels_name = 'Labels',
                batch = 4,
                epoch = 10,
                verbose = True,
                loss = 'categorical_crossentropy',
                easgd = False,
                early_stopping = 5
                )
    #print(output)
