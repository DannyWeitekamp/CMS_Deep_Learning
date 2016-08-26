#We can go into our root file and see what Trees are availiable
#%matplotlib inline
import sys, os
if __name__ == "__main__":
    username = "dweiteka"
    if(len(sys.argv) > 1):
        username = sys.argv[1]


import socket
if("daint" in socket.gethostname()):
    DELPHES_DIR = "/scratch/daint/" + username +  "/Delphes/"
    SOFTWAR_DIR = "/scratch/daint/" + username + "/"
    STORE_TYPE = "msg"
else:
    DELPHES_DIR = "/data/shared/Delphes/"
    SOFTWAR_DIR = "/data/shared/Software/"
    STORE_TYPE = "h5"

archive_dir = DELPHES_DIR+"CSCS_output/keras_archive/"

if(not SOFTWAR_DIR in sys.path):
    sys.path.append(SOFTWAR_DIR)
    if(not "daint" in socket.gethostname()):
        import deepconfig
        dc = deepconfig.deepconfig(gpu='gpu1', backend='theano')

import numpy as np
import pandas as pd
import ntpath
import glob
import itertools

from CMS_SURF_2016.utils.preprocessing import *
from CMS_SURF_2016.utils.callbacks import OverfitStopping, SmartCheckpoint
from CMS_SURF_2016.utils.archiving import *
from CMS_SURF_2016.utils.batch import batchAssertArchived, batchExecuteAndTestTrials
from CMS_SURF_2016.layers.lorentz import Lorentz, _lorentz
from CMS_SURF_2016.layers.slice import Slice



from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Flatten, Reshape, Activation, Dropout, Convolution2D, merge, Input, Flatten, Lambda, LSTM, Masking
from keras.engine.topology import Layer
from keras.callbacks import EarlyStopping
from keras.utils.visualize_util import plot
from keras.layers.advanced_activations import LeakyReLU

def findsubsets(S):
    '''Finds all subsets of a set S'''
    out = []
    for m in range(2, len(S)):
        out = out + [set(x) for x in itertools.combinations(S, m)]
    return out

#The observables taken from the table
observ_types = ['E/c', 'Px', 'Py', 'Pz', 'PT_ET','Eta', 'Phi', 'Charge', 'X', 'Y', 'Z',
                     'Dxy', 'Ehad', 'Eem', 'MuIso', 'EleIso', 'ChHadIso','NeuHadIso','GammaIso']
vecsize = len(observ_types)
epochs = 100
batch_size = 100

label_dir_pairs = \
            [   ("ttbar", DELPHES_DIR+"ttbar_lepFilter_13TeV/pandas_"+STORE_TYPE+"/"),
                ("wjet",  DELPHES_DIR+"wjets_lepFilter_13TeV/pandas_"+STORE_TYPE+"/"),
                ("qcd", DELPHES_DIR+"qcd_lepFilter_13TeV/pandas_"+STORE_TYPE+"/")
            ]
ldpsubsets = [sorted(list(s)) for s in findsubsets(label_dir_pairs)]
#Make sure that we do 3-way classification as well
ldpsubsets.append(label_dir_pairs)


def genModel(name,out_dim, depth, width, dense_activation="relu", dropout = 0.0,sphereCoords=True,weight_output=False):
    inputs = []
    mergelist = []
    for i, profile in enumerate(object_profiles):
        # print(o)
        inp = a = Input(shape=(profile.max_size, vecsize), name="input_"+str(i))
        inputs.append(inp)

        if(name == 'lorentz'):
            b1 = Lorentz(sphereCoords=sphereCoords, weight_output=weight_output ,name="lorentz_"+str(i))(a)
            b1 = Flatten(name="flatten1_"+str(i))(b1)
        elif(name == 'control_dense'):
            b1 = Slice('[:,0:4]',name='slice_1_'+str(i))(a)
            b1 = Flatten(name="4_flatten_"+str(i))(b1)
            b1 = Dense(4 * profile.max_size, activation='linear', name='4_dense_'+str(i))(b1)
        else:
            b1 = Slice('[:,0:4]',name='slice_1_'+str(i))(a)
            b1 = Flatten(name="flatten1_"+str(i))(b1)

        
        b2 = Slice('[:,4:]',name='slice_2_'+str(i))(a)
        b2 = Flatten(name="flatten_2_"+str(i))(b2)
        # b2 = Dense(10, activation='relu')(b2)
        mergelist.append(b1)
        mergelist.append(b2)
    # print(mergelist)
    a = merge(mergelist,mode='concat', name="merge")
    # a = Flatten()(a)
    for i in range(depth):
        a =  Dense(width, activation=dense_activation, name="dense_"+str(i))(a)
        if(dropout > 0.0):
            a =  Dropout(dropout, name="dropout_"+str(i))(a)
    dense_out = Dense(out_dim, activation='sigmoid', name='main_output')(a)
    model = Model(input=inputs, output=dense_out, name=name)
    return model


earlyStopping = EarlyStopping(verbose=1, patience=10)
trial_tups = []
for ldp in ldpsubsets:
    labels = [x[0] for x in ldp]
    for sort_on in ["PT_ET"]:
        for max_EFlow_size in [100]:#[100, 200]:
            
            object_profiles = [ObjectProfile("Electron",-1),
                                ObjectProfile("MuonTight", -1),
                                ObjectProfile("Photon", -1),
                                ObjectProfile("MissingET", 1),
                                ObjectProfile("EFlowPhoton",max_EFlow_size, sort_columns=[sort_on], sort_ascending=False), 
                                ObjectProfile("EFlowNeutralHadron",max_EFlow_size, sort_columns=[sort_on], sort_ascending=False), 
                                ObjectProfile("EFlowTrack",max_EFlow_size, sort_columns=[sort_on], sort_ascending=False)] 

            resolveProfileMaxes(object_profiles, ldp)

            dps, l = getGensDefaultFormat(archive_dir, (100000,20000,20000), 140000, \
                                 object_profiles,ldp,observ_types,megabytes=500, verbose=0)

            dependencies = batchAssertArchived(dps)
            train, num_train = l[0]
            val,   num_val   = l[1]
            test,  num_test  = l[2]
            max_q_size = l[3]
            print("MAXQ: ",max_q_size)

            
            for name in ['lorentz', 'not_lorentz', 'control_dense']:
                for sphereCoords in [False]:
                    for weight_output in [False, True]:
                        for depth in [2,3]:
                            for width in [10,25]:
                                for activation in ['relu']:
                                    for dropout in [0.0]:
                                        activation_name = activation if isinstance(activation, str) \
                                                            else activation.__name__
                                        model = genModel(name, len(labels), depth, width, activation, dropout, sphereCoords, weight_output)

                                        print(model.summary())

                                        trial = KerasTrial(archive_dir, name=name, model=model)

                                        trial.setTrain(train_procedure=train,
                                                       samples_per_epoch=num_train
                                                      )
                                        trial.setValidation(val_procedure=val,
                                                           nb_val_samples=num_val)
                                        trial.setCompilation(loss='binary_crossentropy',
                                                  optimizer='adam',
                                                  metrics=['accuracy']
                                                      )

                                        trial.setFit_Generator( 
                                                        nb_epoch=epochs,
                                                        callbacks=[earlyStopping],
                                                        max_q_size = max_q_size)

                                        trial.write()

                                        #print("EXECUTE: ", name,labels, depth, activation_name)
                                        #trial.execute(custom_objects={"Lorentz":Lorentz,"Slice": Slice},
                                        #             train_arg_decode_func=label_dir_pairs_args_decoder,
                                        #             val_arg_decode_func=label_dir_pairs_args_decoder)


                                        #trial.test(test_proc=test,
                                        #             test_samples=num_test,
                                        #             custom_objects={"Lorentz":Lorentz,"Slice": Slice},
                                        #            arg_decode_func = label_dir_pairs_args_decoder)
                                        print("WIGHHTE:", weight_output)

                                        trial.to_record({"labels": labels,
                                                         "depth": depth,
                                                         "width" : width,
                                                         "sort_on" : sort_on,
                                                         "activation": activation_name,
                                                         "weight_output": weight_output,
                                                         "dropout":dropout,
                                                         "max_EFlow_size": max_EFlow_size,
                                                         "sort_on" : sort_on,
                                                         "optimizer" : "adam"
                                                        })
                                        trial_tups.append((trial, test, num_test, dependencies))
batchExecuteAndTestTrials(trial_tups, time_str="24:00:00")

                
        

    