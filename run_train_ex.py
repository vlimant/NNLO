from threaded_skopt import dummy_func
import json

def evaluate( o ):
    X = (o['par0'], o['par1'])
    value = dummy_func( X )
    open('%s.json'%o['hash'],'w').write(json.dumps(
        {
            'result': value,
            'params' : o,
            'annotate' : 'a free comment'
        }))
    
    

    
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
                
    evaluate( opt )
                                                                                        
