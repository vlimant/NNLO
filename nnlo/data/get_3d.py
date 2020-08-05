import os
import glob
import logging
try:
    import h5py
    pass
except:
    logging.info("import h5py failed")
import numpy as np
import sys

def get_data(datafile):
    #get data for training
    f=h5py.File(datafile,'r')
    y=f.get('target')
    X=np.array(f.get('ECAL'))
    y=(np.array(y[:,1]))
    X[X < 1e-4] = 0
    X = np.expand_dims(X, axis=-1)
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    y = y/100.
    ecal = np.squeeze(np.sum(X, axis=(1, 2, 3)))
    logging.info("X shape {}; y shape {}; ecal shape {}".format(str(X.shape)), str(y.shape), str(ecal.shape))

    f.close()
    return X, y, ecal

def main():
    dest='/data/shared/3DGAN/'
    import socket
    host = os.environ.get('HOST', os.environ.get('HOSTNAME',socket.gethostname()))
    if 'daint' in host:
        dest='/scratch/snx3000/vlimant/3DGAN/'
    if 'titan' in host:
        dest='/ccs/proj/csc291/DATA/3DGAN/'
    
    sub_split = int(sys.argv[1]) if len(sys.argv)>1 else 1
                                              
    for F in glob.glob('/bigdata/shared/LCD/NewV1/*scan/*.h5'):
        _,d,f = F.rsplit('/',2)
        if not 'Ele' in d: continue
        X = None
        if sub_split==1:
            nf = '%s/%s_%s.h5'%( dest,d,f)
            if os.path.isfile( nf) :
                continue
            logging.info("processing files {} into {}".format(F,nf))
            if X is None:
                X,y,ecal = get_data(F)
            o = h5py.File(nf,'w')
            o['X'] = X
            o.create_group("y")
            o['y']['a'] = np.ones(y.shape)
            o['y']['b'] = y
            o['y']['c'] = ecal
            o.close()        
        else:
            for sub in range(sub_split):
                nf = '%s/%s_%s_sub%s.h5'%(dest, d,f,sub)
                if os.path.isfile( nf) :
                    continue
                logging.info("processing files {} into {}".format(F,nf))
                if X is None:
                    X,y,ecal = get_data(F)
                    N = X.shape[0]
                    splits = [i*N/sub_split for i in range(sub_split)]+[-1]
                o = h5py.File(nf,'w')
                o['X'] = X[splits[sub]:splits[sub+1],...]
                o.create_group("y")
                o['y']['a'] = np.ones(y[splits[sub]:splits[sub+1],...].shape)
                o['y']['b'] = y[splits[sub]:splits[sub+1],...]
                o['y']['c'] = ecal[splits[sub]:splits[sub+1],...]
                o.close()
                X = None

    if sub_split == 1:
        sub_files = lambda f:not 'sub' in f
    else:
        sub_files = lambda f:'sub' in f
    
    open('train_3d.list','w').write( '\n'.join(filter(sub_files,glob.glob(dest+'/*.h5')[:-4])))
    open('test_3d.list','w').write( '\n'.join(filter(sub_files,glob.glob(dest+'/*.h5')[-4:])))
    
    open('train_small_3d.list','w').write( '\n'.join(filter(sub_files,glob.glob(dest+'/*.h5')[:-4])))
    open('test_small_3d.list','w').write( '\n'.join(filter(sub_files,glob.glob(dest+'/*.h5')[-4:])))
    
    open('train_7_3d.list','w').write( '\n'.join(filter(sub_files,glob.glob(dest+'/*.h5')[:7])))
    open('test_1_3d.list','w').write( '\n'.join(filter(sub_files,glob.glob(dest+'/*.h5')[-1:])))
    
if __name__ == '__main__':
    main()
