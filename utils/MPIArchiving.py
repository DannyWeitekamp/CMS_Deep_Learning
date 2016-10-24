from .archiving import KerasTrial
from .batch import batchAssertArchived

import sys,os
import numpy as np
# import argparse
import json
import shlex
import subprocess
from mpi4py import MPI
from time import time,sleep

p = "/home/dweitekamp/mpi_learn/"
if(not p in sys.path):sys.path.append(p)
#print(sys.path)

#from mpi_learn.GPU import *
from mpi_learn.mpi.manager import MPIManager, get_device
from mpi_learn.train.algo import Algo
from mpi_learn.train.data import H5Data

class MPI_KerasTrial(KerasTrial):
    
    def execute(self, archiveTraining=True,
                    archiveValidation=True,
                    custom_objects={},
                    verbose=1,
                    numProcesses=2,
                    **kargs):
        print(kargs)
        if(not "isMPI_Instance" in kargs):
            comm = MPI.COMM_WORLD.Dup()
            print("Not MPI_Instance")
            loc = "/data/shared/Software/CMS_SURF_2016/utils/MPIKerasTrial_execute.py"
            print(self.archive_dir, self.hash())
            RunCommand = 'mpirun -np python %s %s %s %s' % (numProcesses, loc, self.archive_dir, self.hash())
            print(RunCommand)
            args = shlex.split(RunCommand)
            env=os.environ
            new_env = {k: v for k, v in env.iteritems() if "MPI" not in k}
            
            #print(env)
            #print(new_env)
            
            #print new_env
            p = subprocess.Popen(RunCommand,shell=True, env=new_env,stdout=subprocess.PIPE, stdin=subprocess.PIPE)
            p.wait()
            result="process myrank "+str(comm.Get_rank())+" got "+p.stdout.read()
            print result
            return
        
        
        load_weights = True
        synchronous = True
        masters = 1
        max_gpus = 2
        sync_every = 1
        MPIoptimizer = "adadelta"
        batch_size = 100

        print(self.train_procedure)
        if(not isinstance(self.train_procedure,list)): self.train_procedure = [self.train_procedure]
        if(not isinstance(self.val_procedure,list)): self.val_procedure = [self.val_procedure]
        
        train_dps = [DataProcedure.from_json(archive_dir,x) for x in self.train_procedure]
        val_dps = [DataProcedure.from_json(archive_dir,x) for x in self.val_procedure]
        print(train_dps)
        
        
        #print(self.train_procedure)
        if(not(isinstance(train_dps, list) and isinstance(train_dps[0], DataProcedure))):
            raise ValueError("Train procedure must be list of DataProcedures")
        if(not(isinstance(val_dps, list) and isinstance(val_dps[0], DataProcedure))):
            raise ValueError("Validation procedure must be list of DataProcedures")
        batchAssertArchived(train_dps)
        batchAssertArchived(val_dps)
        train_list = [dp.get_path() + "archive.h5" for dp in train_dps]
        val_list = [dp.get_path() + "archive.h5" for dp in val_dps]


        comm = MPI.COMM_WORLD.Dup()
        # We have to assign GPUs to processes before importing Theano.
        device = get_device( comm, masters, gpu_limit=max_gpus )
        print "Process",comm.Get_rank(),"using device",device
        os.environ['THEANO_FLAGS'] = "device=%s,floatX=float32" % (device)
        import theano

        # There is an issue when multiple processes import Keras simultaneously --
        # the file .keras/keras.json is sometimes not read correctly.  
        # as a workaround, just try several times to import keras.
        # Note: importing keras imports theano -- 
        # impossible to change GPU choice after this.
        for try_num in range(10):
            try:
                from keras.models import model_from_json
                import keras.callbacks as cbks
                break
            except ValueError:
                print "Unable to import keras. Trying again: %d" % try_num
                sleep(0.1)

        # We initialize the Data object with the training data list
        # so that we can use it to count the number of training examples
        data = H5Data( train_list, batch_size=batch_size, 
                features_name="X", labels_name="Y")
        if comm.Get_rank() == 0:
            validate_every = data.count_data()/batch_size
        callbacks = []
        #callbacks.append( cbks.ModelCheckpoint( '_'.join([
        #    model_name,args.trial_name,"mpi_learn_result.h5"]), 
        #    monitor='val_loss', verbose=1 ) )

        # Creating the MPIManager object causes all needed worker and master nodes to be created
        manager = MPIManager( comm=comm, data=data, num_epochs=self.nb_epoch, 
                train_list=train_list, val_list=val_list, num_masters=masters,
                synchronous=synchronous, callbacks=callbacks )
        # Process 0 defines the model and propagates it to the workers.
        if comm.Get_rank() == 0:
            model = self.compile(custom_objects=custom_objects)
            model_arch = model.to_json()
            #if args.easgd:
                #raise NotImplementedError("Not implemented")
                # algo = Algo(None, loss=args.loss, validate_every=validate_every,
                #         mode='easgd', elastic_lr=args.elastic_lr, sync_every=sync_every,
                #         worker_optimizer=args.worker_optimizer,
                #         elastic_force=args.elastic_force/(comm.Get_size()-1)) 
            #else:
            algo = Algo(MPIoptimizer, loss=self.loss, validate_every=validate_every,
                    sync_every=sync_every, worker_optimizer=self.optimizer) 
            print algo
            weights = model.get_weights()

            manager.process.set_model_info( model_arch, algo, weights )
            t_0 = time.time()
            #raise NotImplementedError("Don't start it just yet")
            histories = manager.process.train() 
            print(histories)
            delta_t = time.time() - t_0
            manager.free_comms()
            print "Training finished in %.3f seconds" % delta_t

            # Make output dictionary
            out_dict = { "args":"MOOO",
                         "history":histories,
                         "train_time":delta_t,
                         }
            #json_name = '_'.join([model_name,args.trial_name,"history.json"]) 
            #with open( json_name, 'w') as out_file:
            #    out_file.write( json.dumps(out_dict, indent=4, separators=(',',': ')) )
            #print "Wrote trial information to",json_name
            