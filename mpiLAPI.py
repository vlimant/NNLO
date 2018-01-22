import os
import keras
import glob
import h5py
import hashlib
import time
from densenet import DenseNet
from keras.optimizers import Adam
from argparse import ArgumentParser
from subprocess import check_output
import tensorflow as tf

class mpi_learn_api:
    def __init__(self, **args):
        args['check'] = time.mktime(time.gmtime())
        hash = hashlib.md5(str(args).encode('utf-8')).hexdigest()
        self.json_file = '/tmp/%s.json'% hash
        #print("self.jsonfile = {}".format(self.json_file))
        if os.path.isfile( self.json_file ) :
            print("hash",hash,"cannot work")
            sys.exit(1)
        self.train_files = '/tmp/%s_train.list'%hash
        self.val_files = '/tmp/%s_val.list'%hash
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
        return check_output(com, shell=True)

def get_model(nb_classes = 3, img_dim = (150, 94, 5), depth = 40, nb_dense_block = 3, growth_rate = 12, dropout_rate= 0.00, nb_filter = 16, lr = 1e-3):
    densenet = DenseNet(nb_classes = nb_classes, img_dim = img_dim, depth = depth, nb_dense_block = nb_dense_block, growth_rate = growth_rate, dropout_rate = dropout_rate, nb_filter = nb_filter)
    optimizer = Adam(lr = lr)
    densenet.compile(loss='categorical_crossentropy', optimizer = optimizer)
    return densenet.to_json()

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
    
    os.environ["CUDA_VISIBLE_DEVICES"]="0,3,4,5"
    
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    #sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=True))

    model = get_model(depth = depth, growth_rate = args.growth, dropout_rate = args.dropout, nb_filter = args.filters, lr = args.lr)

    mlapi = mpi_learn_api( model = model,
                           train_pattern = '/bigdata/shared/LCDJets_Remake/train/04*.h5',
                           val_pattern = '/bigdata/shared/LCDJets_Remake/val/020*.h5',
                           check_file = True
                           )
   
    output = mlapi.train(N=3,
                trial_name = 'test',
                features_name = 'Images',
                labels_name = 'Labels',
                batch = 4,
                epoch = 2,
                verbose = True,
                loss = 'categorical_crossentropy',
                easgd = False,
                early_stopping = 5
                )
    print(output)
    #sess.close()
