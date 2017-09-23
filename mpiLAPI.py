import os
import keras
import glob
import h5py
import hashlib
import time

class mpi_learn_api:
    def __init__(self, **args):
        args['check'] = time.mktime(time.gmtime())
        hash = hashlib.md5(str(args)).hexdigest()
        self.json_file = '/tmp/%s.json'% hash
        if os.path.isfile( self.json_file ) :
            print "hash",hash,"cannot work"
            sys.exit(1)
        self.train_files = '/tmp/%s_train.list'%hash
        self.val_files = '/tmp/%s_val.list'%hash

        open(self.json_file,'w').write(args['model'].to_json())
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
                f.close()
            except:
                print fn,"not usable"
                a_list.remove(fn)
        return a_list
    
    def train(self, **args):
        com = 'mpirun -n %d MPIDriver.py %s %s %s'%(
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
                                 'batch' : 100
                                 }.items():
            v = args.get(option,default)
            if type(v)==bool:
                com +=' --%s'%option.replace('_','-') if v else ''
            else:
                com+=' --%s %s'%(option.replace('_','-'), v)
        print com
        os.system( com )


if __name__ == "__main__":
    from keras.models import model_from_json
    model = model_from_json( open('cnn.json').read())
    tl = 'train.list'
    vl = 'val.list'

    #mlapi = mpi_learn_api( model = model, train_list = tl, val_list = vl)
    mlapi = mpi_learn_api( model = model,
                           train_pattern = '/data/shared/Delphes/np_datasets_new/3_way/MaxLepDeltaR_des/train/images/*_*.h5',
                           val_pattern = '/data/shared/Delphes/np_datasets_new/3_way/MaxLepDeltaR_des/val/images/*_*.h5',
                           #train_pattern = '/data/shared/JetImages/train/*_*.h5',
                           #val_pattern = '/data/shared/JetImages/val/*_*.h5',
                           check_file = True
                           )
    mlapi.train(N=8,
                trial_name = 'with_api',
                features_name = 'Images',
                labels_name = 'Labels'
                )
                
