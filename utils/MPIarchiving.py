from .archiving import KerasTrial
from .batch import batchAssertArchived

import sys,os
import numpy as np
# import argparse
import json
from mpi4py import MPI
from time import time,sleep

from mpi_tools.MPIManager import MPIManager, get_device
from Algo import Algo
from Data import H5Data

class MPI_KerasTrial(KerasTrial):
    def __init__(self,
                    archive_dir,
                    name = 'trial',
                    model=None,
                    train_procedure=None,
                    samples_per_epoch=None,
                    validation_split=0.0,
                    val_procedure=None,
                    nb_val_samples=None,

                    optimizer=None,
                    loss=None,
                    metrics=[],
                    sample_weight_mode=None,

                    batch_size=32,
                    nb_epoch=10,
                    callbacks=[],
                    
                    max_q_size=10,
                    nb_worker=1,
                    pickle_safe=False,

                    shuffle=True,
                    class_weight=None,
                    sample_weight=None
                ):
        KerasTrial.__init__(self,
                    archive_dir,
                    name = 'trial',
                    model=None,
                    train_procedure=None,
                    samples_per_epoch=None,
                    validation_split=0.0,
                    val_procedure=None,
                    nb_val_samples=None,

                    optimizer=None,
                    loss=None,
                    metrics=[],
                    sample_weight_mode=None,

                    batch_size=32,
                    nb_epoch=10,
                    callbacks=[],
                    
                    max_q_size=10,
                    nb_worker=1,
                    pickle_safe=False,

                    shuffle=True,
                    class_weight=None,
                    sample_weight=None)
    def execute(self, archiveTraining=True,
                    archiveValidation=True,
                    train_arg_decode_func=None,
                    val_arg_decode_func=None,
                    custom_objects={},
                    verbose=1):

        load_weights = True
        synchronous = True
        masters = 1
        max_gpus = 2
        sync_every = 1
        MPIoptimizer = "sgd"



        if(not(isinstance(self.val_procedure, list) == False and isinstance(self.val_procedure[0], DataProcedure))):
            raise ValueError("Validation procedure must be list of DataProcedures")
        if(not(isinstance(self.train_procedure, list) == False and isinstance(self.val_procedure[0], DataProcedure))):
            raise ValueError("Train procedure must be list of DataProcedures")
        batchAssertArchived(self.val_procedure)
        batchAssertArchived(self.train_procedure)
        train_list = [dp.get_path() + "archive.h5" for dp in self.train_procedure]
        val_list = [dp.get_path() + "archive.h5" for dp in self.val_procedure]


        comm = MPI.COMM_WORLD.Dup()
        # We have to assign GPUs to processes before importing Theano.
        device = get_device( comm, args.masters, gpu_limit=args.max_gpus )
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
        data = H5Data( train_list, batch_size=args.batch, 
                features_name=args.features_name, labels_name=args.labels_name )
        if comm.Get_rank() == 0:
            validate_every = data.count_data()/args.batch 
        callbacks = []
        callbacks.append( cbks.ModelCheckpoint( '_'.join([
            model_name,args.trial_name,"mpi_learn_result.h5"]), 
            monitor='val_loss', verbose=1 ) )

        # Creating the MPIManager object causes all needed worker and master nodes to be created
        manager = MPIManager( comm=comm, data=data, num_epochs=self.nb_epoch, 
                train_list=train_list, val_list=val_list, num_masters=masters,
                synchronous=synchronous, callbacks=callbacks )
        # Process 0 defines the model and propagates it to the workers.
        if comm.Get_rank() == 0:
            model = load_model(model_name, load_weights=load_weights)
            model_arch = model.to_json()
            if args.easgd:
                raise NotImplementedError("Not implemented")
                # algo = Algo(None, loss=args.loss, validate_every=validate_every,
                #         mode='easgd', elastic_lr=args.elastic_lr, sync_every=sync_every,
                #         worker_optimizer=args.worker_optimizer,
                #         elastic_force=args.elastic_force/(comm.Get_size()-1)) 
            else:
                algo = Algo(MPIoptimizer, loss=self.loss, validate_every=validate_every,
                        sync_every=sync_every, worker_optimizer=self.optimizer) 
            print algo
            weights = model.get_weights()

            manager.process.set_model_info( model_arch, algo, weights )
            t_0 = time()
            raise NotImplementedError("Don't start it just yet")
            histories = manager.process.train() 
            delta_t = time() - t_0
            manager.free_comms()
            print "Training finished in %.3f seconds" % delta_t

            # Make output dictionary
            out_dict = { "args":vars(args),
                         "history":histories,
                         "train_time":delta_t,
                         }
            json_name = '_'.join([model_name,args.trial_name,"history.json"]) 
            with open( json_name, 'w') as out_file:
                out_file.write( json.dumps(out_dict, indent=4, separators=(',',': ')) )
            print "Wrote trial information to",json_name
    else:
        print("Trial %r Already Complete" % self.hash())