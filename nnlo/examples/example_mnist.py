from nnlo.models.model_mnist_tf import make_mnist_model

def get_all():
    import socket,os,glob
    host = os.environ.get('HOST',os.environ.get('HOSTNAME',socket.gethostname()))

    if 'daint' in host:
        all_list = glob.glob('/scratch/snx3000/vlimant/data/mnist/*.h5')
    elif 'titan' in host:
        all_list = glob.glob('/ccs/proj/csc291/DATA/mnist/*.h5')
    else:
        all_list = glob.glob('/bigdata/shared/mnist/*.h5')
    if not all_list:
        all_list = glob.glob('mnist_*.h5')
    return all_list
    
def get_train():
    all_list = get_all()
    l = int( len(all_list)*0.70)
    train_list = all_list[:l]
    return train_list

def get_val():
    all_list = get_all()
    l = int( len(all_list)*0.70)
    val_list = all_list[l:]
    return val_list

def get_features():
    #return ('features', lambda x: x) ##example of data adaptor
    return 'features'

def get_labels():
    return 'labels'
    
