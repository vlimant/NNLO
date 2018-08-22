from threaded_skopt import dummy_func
import json
from threading import Thread
import os
import sys
import time

class fold(Thread):
    def __init__(self, opt, fold):
        Thread.__init__(self)
        self.opt = opt
        self.fold = fold
    def run(self):
        command = 'python '+' '.join(sys.argv)+' --fold %d'%self.fold
        print "running",command
        os.system( command )
    
def evaluate( o , fold = None):
    X = (o['par0'], o['par1'])
    value = dummy_func( X , fold = fold)
    res = {
        'result': value,
        'params' : o,
        'annotate' : 'a free comment'
    }
    if fold is not None:
        res['fold'] = fold
    dest = '%s.json'%o['hash'] if fold is None else '%s_f%d.json'%(o['hash'], fold)
    
    open(dest,'w').write(json.dumps(res))

def evaluate_folds( o , Nfolds , Nthreads=2):
    ## thats a dummy way of running folds sequentially
    #for f in range(Nfolds):
    #    evaluate( opt, fold = f )

    folds = []
    for f in range(Nfolds):
        folds.append( fold( opt, fold = f ) )

    ons = []
    while True:
        if len(ons) < Nthreads:
            ons.append( folds.pop(-1) )
            ons[-1].start()
            time.sleep(2)
            continue
        for f in ons:
            if not f.is_alive():
                ons.remove( f )
                break
        if len(folds) == 0:
            break
        
    ## read all back and make the average
    r = []
    for f in range(Nfolds):
        src = '%s_f%d.json'%(o['hash'], f)
        d = json.loads( open(src).read())
        r.append( d['result'] )
    import numpy as np

    ## that's the final expect answer
    dest = '%s.json'%o['hash']
    res = {
        'result': np.mean( r ),
        'params' : o,
    }
    print "The averaged value on hash",o['hash'],"is",res
    open(dest,'w').write(json.dumps(res))
    
        
if __name__ == "__main__":
    import sys
    ## convert blindy the parameters
    opt={}
    for i,_ in enumerate(sys.argv):
        k = sys.argv[i]
        if k.startswith('--'):
            v = sys.argv[i+1]
            try:
                opt[k[2:]] = float(v)
            except:
                opt[k[2:]] = v
    Nfolds = int(opt.pop('folds')) if 'folds' in opt else 1
    if 'fold' in opt:
        f = int(opt.pop('fold'))
        evaluate( opt, fold = f )
    elif Nfolds>1:
        ## need to spawn enough threads, and collect them all
        print "going for",Nfolds,"folds"
        evaluate_folds( opt, Nfolds = Nfolds )
    else:
        evaluate( opt )
                                                                                        
