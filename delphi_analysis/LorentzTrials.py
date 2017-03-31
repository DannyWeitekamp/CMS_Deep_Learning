import sys

if __package__ is None:
    import sys, os
    sys.path.append(os.path.realpath("/data/shared/Software/"))
    sys.path.append(os.path.realpath("../"))
p = "/home/dweitekamp/mpi_learn/"
if(not p in sys.path):
    sys.path.append(p)

from CMS_Deep_Learning.utils.deepconfig import deepconfig
deepconfig("cpu", backend="theano")

from CMS_Deep_Learning.preprocessing.preprocessing import *
from CMS_Deep_Learning.storage.MPIArchiving import *
from CMS_Deep_Learning.postprocessing.analysistools import findsubsets
from CMS_Deep_Learning.layers.slice import Slice
from CMS_Deep_Learning.layers.lorentz import Lorentz

from keras.models import Model
from keras.layers import Dense, Dropout, merge, Input, Flatten
from keras.callbacks import EarlyStopping

#The observables taken from the table
DEFAULT_OBSV_TYPES = ['E/c', 'Px', 'Py', 'Pz', 'PT_ET','Eta', 'Phi', 'Charge', 'X', 'Y', 'Z',\
                     'Dxy', 'Ehad', 'Eem', 'MuIso', 'EleIso', 'ChHadIso','NeuHadIso','GammaIso']


DEFAULT_LABEL_DIR_PAIRS = \
            [   ("qcd", "/data/shared/Delphes/qcd_lepFilter_13TeV/pandas_h5/"),
                ("ttbar", "/data/shared/Delphes/ttbar_lepFilter_13TeV/pandas_h5/"),
                ("wjet", "/data/shared/Delphes/wjets_lepFilter_13TeV/pandas_h5/")
            ]

def genModel(name,out_dim, depth, width, vecsize, object_profiles, dense_activation="relu", output_activation='softmax',
             dropout = 0.0,sphereCoords=True,weight_output=False):
    inputs = []
    mergelist = []
    for i, profile in enumerate(object_profiles):
        # print(o)
        inp = a = Input(shape=(profile.max_size, vecsize), name="input_"+str(i))
        inputs.append(inp)

        if(name == 'lorentz'):
            b1 = Lorentz(sphereCoords=sphereCoords, weight_output=weight_output,
                         name="lorentz_"+str(i))(a)
            b1 = Flatten(name="flatten1_"+str(i))(b1)
        elif(name == 'lorentz_vsum'):
            b1 = Lorentz(sphereCoords=sphereCoords, weight_output=weight_output,
                         name="lorentz_" + str(i), sum_input=True)(a)
            b1 = Flatten(name="flatten1_" + str(i))(b1)
        elif(name == 'control_dense'):
            b1 = Slice('[:,0:4]',name='slice_1_'+str(i))(a)
            b1 = Flatten(name="4_flatten_"+str(i))(b1)
            b1 = Dense(4 * profile.max_size, activation='linear', name='4_dense_'+str(i))(b1)
        elif(name == 'control'):
            b1 = Slice('[:,0:4]',name='slice_1_'+str(i))(a)
            b1 = Flatten(name="flatten1_"+str(i))(b1)
        else:
            raise ValueError("Model name %r not understood." % name)

        if("_vsum" in name):
            b2 = a
        else:
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
    dense_out = Dense(out_dim, activation=output_activation, name='main_output')(a)
    model = Model(input=inputs, output=dense_out, name=name)
    return model

def runTrials(archive_dir,
                workers,
                observ_types=DEFAULT_OBSV_TYPES,
                label_dir_pairs=DEFAULT_LABEL_DIR_PAIRS,
                epochs = 30,
                batch_size = 100,
                patience = 8,
                num_val = 20000,
                num_train = 75000,
                output_activation = "softmax",
                loss='categorical_crossentropy',
                optimizer_options = ['rmsprop', 'adam'],
                sort_on="PT_ET",
                depth_options = [2,3],
                width_options = [10],
                activation_options = ['relu'],
                dropout_options = [0.0],
                weight_output_options = [True],
                sphereCoords_options = [False]
                ):
    vecsize = len(observ_types)
    ldpsubsets = [sorted(list(s)) for s in findsubsets(label_dir_pairs)]
    # Make sure that we do 3-way classification as well
    ldpsubsets.append(sorted(label_dir_pairs))
    # archive_dir = "/data/shared/Delphes/keras_archive/"

    earlyStopping = EarlyStopping(verbose=1, patience=patience)
    trial_tups = []
    print(archive_dir, workers)
    # Loop over all subsets
    print(ldpsubsets)
    for ldp in ldpsubsets:
        labels = [x[0] for x in ldp]

        object_profiles = [ObjectProfile("Electron", 8, pre_sort_columns=[sort_on], pre_sort_ascending=False),
                           ObjectProfile("MuonTight", 8, pre_sort_columns=[sort_on], pre_sort_ascending=False),
                           # ObjectProfile("Photon", -1),
                           ObjectProfile("MissingET", 1, pre_sort_columns=[sort_on], pre_sort_ascending=False),
                           ObjectProfile("EFlowPhoton", 100, pre_sort_columns=[sort_on], pre_sort_ascending=False),
                           ObjectProfile("EFlowNeutralHadron", 100, pre_sort_columns=[sort_on], pre_sort_ascending=False),
                           ObjectProfile("EFlowTrack", 100, pre_sort_columns=[sort_on], pre_sort_ascending=False)
                           ]
        dps, l = getGensDefaultFormat(archive_dir, (num_val, num_train), num_val + num_train, \
                                      object_profiles, ldp, observ_types,
                                      batch_size=batch_size, megabytes=100,
                                      verbose=0)

        dependencies = batchAssertArchived(dps, 4)
        val, _num_val = l[0]
        train, _num_train = l[1]
        max_q_size = l[2]

        val_dps = val.args[0]
        train_dps = train.args[0]
        # resolveProfileMaxes(object_profiles, ldp)
        for name in ['lorentz', 'lorentz_vsum', 'control', 'control_dense']:
            for optimizer in optimizer_options:
                for sphereCoords in sphereCoords_options:
                    for weight_output in weight_output_options:  # [False, True]:
                        for depth in depth_options:
                            for width in width_options:
                                for activation in activation_options:
                                    for dropout in dropout_options:
                                        # Weight output is really only for lorentz
                                        if (weight_output == True and name != 'lorentz' and len(weight_output_options) > 1): continue

                                        activation_name = activation if isinstance(activation, str) \
                                            else activation.__name__

                                        model = genModel(name, len(labels), depth, width, vecsize ,object_profiles,
                                                         dense_activation=activation, output_activation=output_activation,
                                                         dropout=dropout, sphereCoords=sphereCoords, weight_output=weight_output)

                                        trial = MPI_KerasTrial(archive_dir, name=name, model=model, workers=workers,
                                                               custom_objects={"Lorentz": Lorentz, "Slice": Slice})

                                        trial.set_train(train_procedure=train_dps,
                                                        samples_per_epoch=_num_train
                                                        )
                                        trial.set_validation(val_procedure=val_dps,
                                                             nb_val_samples=_num_val)

                                        trial.set_compilation(loss=loss,
                                                              optimizer=optimizer,
                                                              metrics=['accuracy']
                                                              )

                                        trial.set_fit_generator(
                                            epochs=epochs,
                                            callbacks=[earlyStopping],
                                            max_q_size=max_q_size)
                                        trial.write()

                                        trial.to_record({"labels": labels,
                                                         "depth": depth,
                                                         "width": width,
                                                         "sort_on": sort_on,
                                                         "activation": activation_name,
                                                         "weight_output": weight_output,
                                                         "dropout": dropout,
                                                         "optimizer": optimizer,
                                                         })
                                        trial_tups.append((trial, None, None, dependencies))
    for tup in trial_tups:
        tup[0].summary()
    for tup in trial_tups:
        tup[0].execute()
        # batchExecuteAndTestTrials(trial_tups)

if __name__ == '__main__':
    argv = sys.argv
    runTrials(argv[1], int(argv[2]))




                
        

    