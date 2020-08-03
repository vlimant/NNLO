from nnlo.models.Models import make_cifar10_model

get_model = make_cifar10_model
def get_name():
    return 'cifar10'

def get_all():
    import socket,os,glob
    host = os.environ.get('HOST',os.environ.get('HOSTNAME',socket.gethostname()))

    all_list = glob.glob('mnist_*.h5')
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
    
