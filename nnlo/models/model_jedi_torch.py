import torch
import torch.nn as nn
from torch.autograd.variable import Variable
import itertools
import numpy as np

args_cuda = True
args_sumO = True

class GraphNet(nn.Module):
    def __init__(self, n_constituents, n_targets, params, hidden, De, Do, 
                 fr_activation=0, fo_activation=0, fc_activation=0, optimizer = 0, verbose = False):
        super(GraphNet, self).__init__()
        self.hidden = hidden
        self.P = len(params)
        self.N = n_constituents
        self.Nr = self.N * (self.N - 1)
        self.Dr = 0
        self.De = De
        self.Dx = 0
        self.Do = Do
        self.n_targets = n_targets
        self.fr_activation = fr_activation
        self.fo_activation = fo_activation
        self.fc_activation = fc_activation
        self.optimizer = optimizer
        self.verbose = verbose
        self.assign_matrices()

        self.sum_O = args_sumO
        self.Ra = torch.ones(self.Dr, self.Nr)
        self.fr1 = nn.Linear(2 * self.P + self.Dr, hidden).cuda()
        self.fr2 = nn.Linear(hidden, int(hidden/2)).cuda()
        self.fr3 = nn.Linear(int(hidden/2), self.De).cuda()
        self.fo1 = nn.Linear(self.P + self.Dx + self.De, hidden).cuda()
        self.fo2 = nn.Linear(hidden, int(hidden/2)).cuda()
        self.fo3 = nn.Linear(int(hidden/2), self.Do).cuda()
        if self.sum_O:
            self.fc1 = nn.Linear(self.Do *1, hidden).cuda()
        else:
            self.fc1 = nn.Linear(self.Do * self.N, hidden).cuda()
        self.fc2 = nn.Linear(hidden, int(hidden/2)).cuda()
        self.fc3 = nn.Linear(int(hidden/2), self.n_targets).cuda()

    def assign_matrices(self):
        self.Rr = torch.zeros(self.N, self.Nr)
        self.Rs = torch.zeros(self.N, self.Nr)
        receiver_sender_list = [i for i in itertools.product(range(self.N), range(self.N)) if i[0]!=i[1]]
        for i, (r, s) in enumerate(receiver_sender_list):
            self.Rr[r, i] = 1
            self.Rs[s, i] = 1
        self.Rr = Variable(self.Rr).cuda()
        self.Rs = Variable(self.Rs).cuda()

    def forward(self, x):
        Orr = self.tmul(x.float(), self.Rr)
        Ors = self.tmul(x.float(), self.Rs)
        B = torch.cat([Orr, Ors], 1)
        ### First MLP ###
        B = torch.transpose(B, 1, 2).contiguous()
        if self.fr_activation ==2:
            B = nn.functional.selu(self.fr1(B.view(-1, 2 * self.P + self.Dr)))
            B = nn.functional.selu(self.fr2(B))
            E = nn.functional.selu(self.fr3(B).view(-1, self.Nr, self.De))            
        elif self.fr_activation ==1:
            B = nn.functional.elu(self.fr1(B.view(-1, 2 * self.P + self.Dr)))
            B = nn.functional.elu(self.fr2(B))
            E = nn.functional.elu(self.fr3(B).view(-1, self.Nr, self.De))
        else:
            B = nn.functional.relu(self.fr1(B.view(-1, 2 * self.P + self.Dr)))
            B = nn.functional.relu(self.fr2(B))
            E = nn.functional.relu(self.fr3(B).view(-1, self.Nr, self.De))
        del B
        E = torch.transpose(E, 1, 2).contiguous()
        Ebar = self.tmul(E, torch.transpose(self.Rr, 0, 1).contiguous())
        del E
        C = torch.cat([x.float(), Ebar], 1)
        del Ebar
        C = torch.transpose(C, 1, 2).contiguous()
        ### Second MLP ###
        if self.fo_activation ==2:
            C = nn.functional.selu(self.fo1(C.view(-1, self.P + self.Dx + self.De)))
            C = nn.functional.selu(self.fo2(C))
            O = nn.functional.selu(self.fo3(C).view(-1, self.N, self.Do))
        elif self.fo_activation ==1:
            C = nn.functional.elu(self.fo1(C.view(-1, self.P + self.Dx + self.De)))
            C = nn.functional.elu(self.fo2(C))
            O = nn.functional.elu(self.fo3(C).view(-1, self.N, self.Do))
        else:
            C = nn.functional.relu(self.fo1(C.view(-1, self.P + self.Dx + self.De)))
            C = nn.functional.relu(self.fo2(C))
            O = nn.functional.relu(self.fo3(C).view(-1, self.N, self.Do))
        del C
        ## sum over the O matrix
        if self.sum_O:
            O = torch.sum( O, dim=1)
        ### Classification MLP ###
        if self.fc_activation ==2:
            if self.sum_O:
                N = nn.functional.selu(self.fc1(O.view(-1, self.Do * 1)))
            else:
                N = nn.functional.selu(self.fc1(O.view(-1, self.Do * self.N)))
            N = nn.functional.selu(self.fc2(N))       
        elif self.fc_activation ==1:
            if self.sum_O:
                N = nn.functional.elu(self.fc1(O.view(-1, self.Do * 1)))
            else:
                N = nn.functional.elu(self.fc1(O.view(-1, self.Do * self.N)))
            N = nn.functional.elu(self.fc2(N))
        else:
            if self.sum_O:
                N = nn.functional.relu(self.fc1(O.view(-1, self.Do * 1)))
            else:
                N = nn.functional.relu(self.fc1(O.view(-1, self.Do * self.N)))
            N = nn.functional.relu(self.fc2(N))
        del O
        #N = nn.functional.relu(self.fc3(N))
        N = self.fc3(N)
        return N

    def tmul(self, x, y):  #Takes (I * J * K)(K * L) -> I * J * L 
        x_shape = x.size()
        y_shape = y.size()
        return torch.mm(x.view(-1, x_shape[2]), y).view(-1, x_shape[1], y_shape[1])


def get_model(**args):

    nParticles = 150
    labels = ['j_g', 'j_q', 'j_w', 'j_z', 'j_t']
    params = ['j1_px', 'j1_py' , 'j1_pz' , 'j1_e' , 'j1_erel' , 'j1_pt' , 'j1_ptrel', 'j1_eta' , 'j1_etarel' ,
              'j1_etarot' , 'j1_phi' , 'j1_phirel' , 'j1_phirot', 'j1_deltaR' , 'j1_costheta' , 'j1_costhetarel']

    load = True
    if load:
        if args_sumO:
            x = [50, 14, 10, 2, 2, 2, 0]
        else:
            x = [10, 4, 14, 2, 2, 2, 0]
        ## load the best model for 150 particles
        args.setdefault('hidden', x[0])
        args.setdefault('De', x[1])
        args.setdefault('Do', x[2])
        args.setdefault('fr_act', x[3])
        args.setdefault('fo_act', x[4])
        args.setdefault('fc_act', x[5])

    mymodel = GraphNet(nParticles, len(labels), params,
                       hidden = args.get('hidden', 10),
                       De = args.get('De', 10),
                       Do = args.get('Do',10),
                       fr_activation=args.get('fr_act',0),
                       fo_activation=args.get('fo_act',0),
                       fc_activation=args.get('fc_act',0),
                       optimizer=0,# disabled
                       verbose=True)
    

    return mymodel

def get_name():
    return 'hls4ml-jedi'

def get_all():
    import os,glob

    if os.path.isdir('/scratch/snx3000/vlimant'):
        all_list = glob.glob('/scratch/snx3000/vlimant/data/mnist/jetImage*_150p_*.h5')
    elif os.path.isdir('/ccs/proj/csc291/DATA/hls-fml'):
        all_list = glob.glob('/ccs/proj/csc291/DATA/hls-fml/NEWDATA/*_150p_*.h5')
    elif os.path.isdir('/mnt/ceph/users/vlimant'):
        all_list = glob.glob('/mnt/ceph/users/vlimant/hls-fml/jetImage*_150p_*.h5')
    elif os.path.isdir('/simons/scratch/vlimant'):
        all_list = glob.glob('/simons/scratch/vlimant/JEDI/jetImage*_150p_*.h5')
    elif os.path.isdir('/gpfs/alpine/world-shared/'):
        all_list = glob.glob('/gpfs/alpine/world-shared/hep120/vlimant/JEDI/jetImage*_150p_*.h5')
    elif os.path.isdir('/storage/group/gpu/bigdata'):
        all_list = glob.glob('/storage/group/gpu/bigdata/hls-fml/NEWDATA/jetImage*_150p_*.h5')
    else:
        all_list = []
        
    return all_list

from skopt.space import Real, Integer, Categorical
get_model.parameter_range =     [
    Integer(10,100, name='hidden'),
    Integer(10,100, name='De'),
    Integer(10,100, name='Do'),
    Categorical([0,1,2], name='fr_act'),
    Categorical([0,1,2], name='fo_act'),
    Categorical([0,1,2], name='fc_act'),    
]

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
    #return 'X'
    return 'jetConstituentList', lambda x:np.swapaxes(x, 1, 2)

def get_labels():
    #return 'Y'
    return 'jets',lambda y:y[0:,-6:-1]


if __name__ == "__main__":
    import glob
    import h5py
    import numpy as np
    
    for f in glob.glob('/bigdata/shared/hls-fml/NEWDATA/jetImage*_150p_*.h5'):
        if 'JEDI' in f : continue
        fi = h5py.File(f)
        X = np.array(fi.get('jetConstituentList'))
        X = np.swapaxes(X, 1, 2)
        Y = np.array(fi.get('jets'))[0:,-6:-1]
        #Y = np.argmax(Y, axis=1) ## torch pre-processing
        fi.close()
        fo = h5py.File(f+'_JEDI.h5','w')
        fo['X'] = X
        fo['Y'] = Y
        fo.close()
        
