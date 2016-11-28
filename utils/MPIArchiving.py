import sys,os
import numpy as np
import json
import shlex
import subprocess
from mpi4py import MPI
from time import time,sleep
import select
import importlib

from mpi_learn.mpi.manager import MPIManager, get_device
from mpi_learn.train.algo import Algo
from mpi_learn.train.data import H5Data


from .archiving import KerasTrial, DataProcedure
from .batch import batchAssertArchived

class MPI_KerasTrial(KerasTrial):
    
    def __init__(self,*args, **kargs):
        custom_objects = None
        if("custom_objects" in kargs):
            custom_objects = kargs["custom_objects"]
            del kargs["custom_objects"]
        #print(custom_objects)
        if(custom_objects != None): 
            self.setCustomObjects(custom_objects)
        else:
            self.custom_objects = {}
        #print(self.custom_objects)
        #raise ValueError()
        KerasTrial.__init__(self,*args,**kargs)

    def setCustomObjects(self,custom_objects):
        self.custom_objects = {name:obj.__module__ if hasattr(obj, "__module__") else obj for name, obj in custom_objects.items()}
    
    print("MOOPGS")
    def execute(self, archiveTraining=True,
                    archiveValidation=True,
                    verbose=1,
                    numProcesses=2):
        
        # print(kargs)
        # if(not "isMPI_Instance" in kargs):
        self.write()
        
        # comm = MPI.COMM_WORLD.Dup()
        # print("Not MPI_Instance")
        loc = "/data/shared/Software/CMS_SURF_2016/utils/MPIKerasTrial_execute.py"
        print(self.archive_dir, self.hash())
        RunCommand = 'mpirun -np %s python %s %s %s' % (numProcesses, loc, self.archive_dir, self.hash())
        print(RunCommand)

        args = shlex.split(RunCommand)
        env=os.environ
        new_env = {k: v for k, v in env.iteritems() if "MPI" not in k}
        
        p = subprocess.Popen("exec " + RunCommand,shell=True, env=new_env,stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            while True:
                reads = [p.stdout.fileno(), p.stderr.fileno()]
                ret = select.select(reads, [], [])
                for fd in ret[0]:
                    if fd == p.stdout.fileno():
                        read = p.stdout.readline()
                        sys.stdout.write(read)
                    if fd == p.stderr.fileno():
                        read = p.stderr.readline()
                        sys.stderr.write(read)
                if p.poll() != None:
                    break
        except Exception as e:
            print("KILLING THIS SHIT:",p.pid,os.getpgid(p.pid))
            p.kill()
            del p
            sys.exit()
        return
            
    def _execute_MPI(self,
                    comm=None,
                    masters=1,
                    easgd=True,
                    archiveTraining=True,
                    archiveValidation=True,
                    verbose=1):
        
            #return prep_func
        #print(self.custom_objects)
        #print(custom_objects)
        #print(Lorentz, Slice)
        #raise ValueError()
        load_weights = True
        synchronous = False
        sync_every = 1
        MPIoptimizer = "rmsprop"
        batch_size = 100

        if(comm == None):
            comm = MPI.COMM_WORLD.Dup()



        # if(not isinstance(self.train_procedure,list)): self.train_procedure = [self.train_procedure]
        # if(not isinstance(self.val_procedure,list)): self.val_procedure = [self.val_procedure]
        if(not(isinstance(self.train_procedure,list) and isinstance(self.train_procedure[0],DataProcedure))):
            raise ValueError("Trial attribute train_procedure: expected list of DataProcedures but got type %r" % type(self.train_procedure))

        if(not(isinstance(self.val_procedure,list) and isinstance(self.val_procedure[0],DataProcedure))):
            raise ValueError("Trial attribute val_procedure: expected list of DataProcedures but got type %r" % type(self.val_procedure))

        train_dps = [DataProcedure.from_json(self.archive_dir,x) for x in self.train_procedure]
        val_dps = [DataProcedure.from_json(self.archive_dir,x) for x in self.val_procedure]

        if(not(isinstance(train_dps, list) and isinstance(train_dps[0], DataProcedure))):
            raise ValueError("Train procedure must be list of DataProcedures")
        if(not(isinstance(val_dps, list) and isinstance(val_dps[0], DataProcedure))):
            raise ValueError("Validation procedure must be list of DataProcedures")
        batchAssertArchived(train_dps)
        batchAssertArchived(val_dps)
        train_list = [dp.get_path() + "archive.h5" for dp in train_dps]
        val_list = [dp.get_path() + "archive.h5" for dp in val_dps]
        print("Train List:", train_list)
        print("Val List:", val_list)

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


        custom_objects = {}
        for name, module in self.custom_objects.items():
            try:
                #my_module = importlib.import_module('os.path')
                custom_objects[name] = getattr(importlib.import_module(module), name)
                #exec("from " + module +  " import " + name)
            except Exception:
                raise ValueError("Custom Object %r does not exist in %r. \
                    For best results Custom Objects should be importable and not locally defined." % (str(name), str(module)))

        # We initialize the Data object with the training data list
        # so that we can use it to count the number of training examples

        data = H5Data( train_list, batch_size=batch_size, 
                features_name="X", labels_name="Y")
        if comm.Get_rank() == 0:
            validate_every = data.count_data()/batch_size
       
        callbacks = self._generateCallbacks(verbose=verbose)


        # Creating the MPIManager object causes all needed worker and master nodes to be created
        manager = MPIManager( comm=comm, data=data, num_epochs=self.nb_epoch, 
                train_list=train_list, val_list=val_list, num_masters=masters,
                synchronous=synchronous, callbacks=callbacks )
        # Process 0 defines the model and propagates it to the workers.
        if comm.Get_rank() == 0:
            model = self.compile(custom_objects=custom_objects)
            model_arch = model.to_json()
            if easgd:
                # raise NotImplementedError("Not implemented")
                algo = Algo(None, loss=self.loss, validate_every=validate_every,
                        mode='easgd', elastic_lr=1.0, sync_every=sync_every,
                        worker_optimizer='sgd',
                        elastic_force=0.9/(comm.Get_size()-1)) 
            else:
                algo = Algo(MPIoptimizer, loss=self.loss, validate_every=validate_every,
                        sync_every=sync_every, worker_optimizer=self.optimizer) 
            print algo
            weights = model.get_weights()

            manager.process.set_model_info( model_arch, algo, weights )
            t_0 = time()
            histories = manager.process.train() 
            delta_t = time() - t_0
            manager.free_comms()
            print "Training finished in %.3f seconds" % delta_t
            print(histories)

            
            
