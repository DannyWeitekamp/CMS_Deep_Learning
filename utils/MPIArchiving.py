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

MPI_INPUT_DEFAULTS = { "masters" : 1,
                       "workers" : 2,
                       "max_gpus" : 2,
                       "master_gpu" : False,
                       "synchronous" : False,
                       "master_optimizer" : "rmsprop",
                       #"worker_optimizer" : "rmsprop",
                       "sync_every" : 1,
                       "easgd" : False,
                       "elastic_force" : 0.9,
                       "elastic_lr" : 1.0,
                       "elastic_momentum" : 0.0

}

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

        for key,value in MPI_INPUT_DEFAULTS.items():
            if(key in kargs):
                setattr(self, key, kargs[key])
            else:
                setattr(self, key, value)

        #print(self.custom_objects)
        #raise ValueError()
        KerasTrial.__init__(self,*args,**kargs)

    def _remove_dict_defaults(self, d):
        del_keys = []
        for key in d:
            if(key in MPI_INPUT_DEFAULTS and MPI_INPUT_DEFAULTS[key] == d[key]):
                del_keys.append(key)
        for key in del_keys:
            del d[key]

        d = super(MPI_KerasTrial, self)._remove_dict_defaults(d)
        return d





    def setCustomObjects(self,custom_objects):
        self.custom_objects = {name:obj.__module__ if hasattr(obj, "__module__") else obj for name, obj in custom_objects.items()}
    
    # print("MOOPGS")

    def kill(self, p):
        print("Killing %r and related processes" % self.hash(),p.pid,os.getpgid(p.pid))
        p.kill()
        del p
        sys.exit()
    def execute(self, archiveTraining=True,
                    archiveValidation=True,
                    verbose=1,
                    # numProcesses=2
                    ):
        
        # print(kargs)
        # if(not "isMPI_Instance" in kargs):
        if(not self.is_complete()):
            self.write()
            
            # comm = MPI.COMM_WORLD.Dup()
            # print("Not MPI_Instance")
            loc = "/data/shared/Software/CMS_SURF_2016/utils/MPIKerasTrial_execute.py"
            print(self.archive_dir, self.hash())
            RunCommand = 'mpirun -np %s python %s %s %s --masters %s --max-gpus %s' % (self.workers + self.masters, loc, self.archive_dir, self.hash(), self.masters, self.max_gpus)
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
            except :
                self.kill(p)
            # except Exception as e:
                # self.kill(p)
                
        else:
            print("Trial %r Already Complete" % self.hash())
        self._history_to_record(['val_acc'])
        self.to_record( {'elapse_time' : self.get_history()['elapse_time']}, replace=True)
        # dct =  {'num_train' : self.samples_per_epoch,
        #             'num_validation' : num_val,
        #             'elapse_time' : self.get_history()['elapse_time'],
        #             # 'fit_cycles' : len(train_procs)
        #             }
            # self.to_record( dct, replace=True)
        return
            
    def _execute_MPI(self,
                    comm=None,
                    # masters=1,
                    # easgd=False,
                    archiveTraining=True,
                    archiveValidation=True,
                    verbose=1):
        
            #return prep_func
        #print(self.custom_objects)
        #print(custom_objects)
        #print(Lorentz, Slice)
        #raise ValueError()
        load_weights = True
        # synchronous = False
        # sync_every = 1
        # MPIoptimizer = "rmsprop"
        # batch_size = 100

        if(comm == None):
            comm = MPI.COMM_WORLD.Dup()



        # if(not isinstance(self.train_procedure,list)): self.train_procedure = [self.train_procedure]
        # if(not isinstance(self.val_procedure,list)): self.val_procedure = [self.val_procedure]
        if(not(isinstance(self.train_procedure,list))):
            raise ValueError("Trial attribute train_procedure: expected list of DataProcedures but got type %r" % type(self.train_procedure))
        if(not(isinstance(self.val_procedure,list))):
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
        # print("Train List:", train_list)
        # print("Val List:", val_list)

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
            except:
                raise ValueError("Custom Object %r does not exist in %r. \
                    For best results Custom Objects should be importable and not locally defined." % (str(name), str(module)))

        # We initialize the Data object with the training data list
        # so that we can use it to count the number of training examples

        data = H5Data( train_list, batch_size=self.batch_size, 
                features_name="X", labels_name="Y")
        num_train = data.count_data()
        


        if comm.Get_rank() == 0:
            validate_every = num_train/self.batch_size
       
        callbacks = self._generateCallbacks(verbose=verbose)


        # Creating the MPIManager object causes all needed worker and master nodes to be created
        manager = MPIManager( comm=comm, data=data, num_epochs=self.nb_epoch, 
                train_list=train_list, val_list=val_list, num_masters=self.masters,
                synchronous=self.synchronous, callbacks=callbacks, custom_objects=custom_objects )
        # Process 0 defines the model and propagates it to the workers.
        if comm.Get_rank() == 0:
            record = self.read_record()
            if(not "num_train" in record):
                self.to_record({"num_train": num_train})
            if(not "num_val" in record):
                val_data = H5Data( train_list, batch_size=self.batch_size, 
                features_name="X", labels_name="Y")
                self.to_record({"num_val": val_data.count_data()})

            print(custom_objects)
            model = self.compile(custom_objects=custom_objects)
            model_arch = model.to_json()
            if self.easgd:
                # raise NotImplementedError("Not implemented")
                algo = Algo(None, loss=self.loss, validate_every=validate_every,
                        mode='easgd', elastic_lr=1.0, sync_every=self.sync_every,
                        worker_optimizer='sgd',
                        elastic_force=0.9/(comm.Get_size()-1)) 
            else:
                algo = Algo(self.master_optimizer, loss=self.loss, validate_every=validate_every,
                        sync_every=self.sync_every, worker_optimizer=self.optimizer) 
            print algo
            weights = model.get_weights()

            manager.process.set_model_info( model_arch, algo, weights )
            t_0 = time()
            histories = manager.process.train() 
            delta_t = time() - t_0
            manager.free_comms()
            print "Training finished in %.3f seconds" % delta_t
            print(histories)

            
            
