import sys

if __package__ is None:
    import sys, os
    sys.path.append(os.path.realpath("/data/shared/Software/"))
    sys.path.append(os.path.realpath("../"))
# p = "/home/dweitekamp/mpi_learn/"
# if(not p in sys.path):
#     sys.path.append(p)

# from CMS_Deep_Learning.utils.deepconfig import deepconfig
# deepconfig("cpu", backend="theano")

from CMS_Deep_Learning.storage.batch import *
from CMS_Deep_Learning.preprocessing.preprocessing import *
# from CMS_Deep_Learning.storage.MPIArchiving import *
from CMS_Deep_Learning.postprocessing.analysistools import findsubsets

from keras.models import Model
from keras.layers import Dense, Dropout, merge, Input, LSTM, Masking
from keras.callbacks import EarlyStopping


#The observables taken from the table
DEFAULT_OBSV_TYPES = ['E/c', 'Px', 'Py', 'Pz', 'PT_ET','Eta', 'Phi',
                      "MaxLepDeltaEta", "MaxLepDeltaPhi",'MaxLepDeltaR', 'MaxLepKt', 'MaxLepAntiKt',
                      "METDeltaEta","METDeltaPhi",'METDeltaR', 'METKt', 'METAntiKt',
                      'Charge', 'X', 'Y', 'Z',
                      'Dxy', 'Ehad', 'Eem', 'MuIso', 'EleIso', 'ChHadIso','NeuHadIso','GammaIso', "ObjFt1","ObjFt2","ObjFt3"]


DEFAULT_LABEL_DIR_PAIRS = \
            [   ("qcd", "$DELPHES_DIR/qcd_lepFilter/pandas_h5/"),
                ("ttbar", "$DELPHES_DIR/ttbar_lepFilter/pandas_h5/"),
                ("wjet", "$DELPHES_DIR/wjets_lepFilter/pandas_h5/")

            ]
def genModel(name,object_profiles,out_dim, depth, vecsize
            ,lstm_activation="relu", lstm_dropout = 0.0, dropout=0.0,output_activation="softmax", single_list=False):
    inputs = []
    if(single_list):
        a = Input(shape=(sum([p.max_size for p in object_profiles]) , vecsize), name="input")
        inputs.append(a)
    else:
        mergelist = []
        for i, profile in enumerate(object_profiles):
            inp = a = Input(shape=(profile.max_size , vecsize), name="input_" + str(i))
            inputs.append(inp)
            mergelist.append(a)
        a = merge(mergelist, mode='concat', concat_axis=1, name="merge")
    for i in range(depth):
        a = Masking(mask_value=0.0)(a)
        a = LSTM(vecsize,
                 input_shape=(None,vecsize),
                 dropout_W=lstm_dropout,
                 dropout_U=lstm_dropout,
                 activation=lstm_activation,
                 name = "lstm_" +str(i))(a)
        if(dropout > 0.0):
            a =  Dropout(dropout, name="dropout_"+str(i))(a)
    dense_out = Dense(out_dim, activation=output_activation, name='main_output')(a)
    model = Model(input=inputs, output=dense_out, name=name)
    return model

def runTrials(archive_dir,
              workers,
              batchProcesses,
              delphes_dir=None,
              observ_types=DEFAULT_OBSV_TYPES,
              label_dir_pairs=DEFAULT_LABEL_DIR_PAIRS,
              nb_epoch = 30,
              batch_size = 100,
              patience = 8,
              num_val = 20000,
              num_train = 100000,
              output_activation = "softmax",
              loss='categorical_crossentropy',
              optimizer_options = ['rmsprop'],
              sortings = [("MaxLepDeltaPhi", False)],  #,("MaxLepDeltaEta", False), ("PT_ET", False), ("PT_ET", True),('MaxLepDeltaR', False), ('MaxLepKt',False), ('MaxLepAntiKt',False)],#, ('METDeltaR', False), ('METKt',False), ('METAntiKt',False),
                            #("METDeltaPhi", False), ("METDeltaEta", False)],
                single_list_options = [True]
              ):
    if(delphes_dir == None):
        split = list(archive_dir.split("/"))
        split = split[:split.index("Delphes")+1]
        delphes_dir = "/".join(split)
        
    os.environ["DELPHES_DIR"] = delphes_dir
    vecsize = len(observ_types)
    ldpsubsets = [sorted(list(s)) for s in findsubsets(label_dir_pairs)]
    #Make sure that we do 3-way classification as well
    ldpsubsets.append(sorted(label_dir_pairs))
    ldpsubsets = ldpsubsets[:1
                 ]
    #archive_dir = "/data/shared/Delphes/keras_archive/"

    earlyStopping = EarlyStopping(verbose=1, patience=patience)
    trial_tups = []
    print(archive_dir, workers)
    # Loop over all subsets
    print(ldpsubsets)
    for single_list in single_list_options:
    	for sort_on, sort_ascending in sortings:
            for ldp in ldpsubsets:
                labels = [x[0] for x in ldp]

                object_profiles = [
                    # ObjectProfile("Photon", -1, pre_sort_columns=["PT_ET"], pre_sort_ascending=False, sort_columns=[sort_on], sort_ascending=False, addColumns={"ObjType":3}),
                    ObjectProfile("EFlowPhoton", 100, pre_sort_columns=["PT_ET"], pre_sort_ascending=False,
                                  sort_columns=[sort_on], sort_ascending=sort_ascending, addColumns={"ObjFt1": -1, "ObjFt2": -1,"ObjFt3": -1}),
                    ObjectProfile("EFlowNeutralHadron", 100, pre_sort_columns=["PT_ET"], pre_sort_ascending=False,
                                  sort_columns=[sort_on], sort_ascending=sort_ascending, addColumns={"ObjFt1": -1, "ObjFt2": -1,"ObjFt3": 1}),
                    ObjectProfile("EFlowTrack", 100, pre_sort_columns=["PT_ET"], pre_sort_ascending=False,
                                  sort_columns=[sort_on], sort_ascending=sort_ascending, addColumns={"ObjFt1": -1, "ObjFt2": 1,"ObjFt3": -1}),
                    ObjectProfile("Electron", 8, pre_sort_columns=["PT_ET"], pre_sort_ascending=False,
                                  sort_columns=[sort_on],
                                  sort_ascending=sort_ascending, addColumns={"ObjFt1": -1, "ObjFt2": 1,"ObjFt3": 1}),
                    ObjectProfile("MuonTight", 8, pre_sort_columns=["PT_ET"], pre_sort_ascending=False,
                                  sort_columns=[sort_on],
                                  sort_ascending=sort_ascending, addColumns={"ObjFt1": 1, "ObjFt2": -1,"ObjFt3": -1}),
                    ObjectProfile("MissingET", 1, addColumns={"ObjFt1": 1, "ObjFt2": -1,"ObjFt3": 1}),]

                #resolveProfileMaxes(object_profiles, ldp)
                #print(archive_dir, (num_val, num_train), num_val+num_train, \
                #                             object_profiles, ldp, observ_types,)
		print(sort_on, labels, single_list)
                dps, l = getGensDefaultFormat(archive_dir, (num_val, num_train), num_val+num_train, \
                                              object_profiles, ldp, observ_types,
                                              single_list=single_list, sort_columns=[sort_on], sort_ascending=sort_ascending,
                                              batch_size=batch_size, megabytes=250,
                                              verbose=0)

                dependencies = batchAssertArchived(dps, batchProcesses)
                val, _num_val = l[0]
                train, _num_train = l[1]
                max_q_size = l[2]

                val_dps = val.args[0]
                train_dps = train.args[0]

                for name in ['LSTM']:
                    for optimizer in optimizer_options:
                        for depth in [1]:
                            for activation in ['tanh']:
                                for lstm_dropout in [0.0]:
                                    for dropout in [0.0]:
                                        activation_name = activation if isinstance(activation, str) \
                                            else activation.__name__

                                        model = genModel(name, object_profiles, len(labels), depth,vecsize, lstm_activation=activation,
                                                         lstm_dropout=lstm_dropout,dropout=dropout, output_activation=output_activation,
                                                        single_list=single_list)


                                        if(workers == 1):
                                            trial = KerasTrial(archive_dir, name=name, model=model,seed=0)
                                            t,v= train,val
                                        else:
                                            sys.path.append(("../../mpi_learn"))
                                            from CMS_Deep_Learning.storage.MPIArchiving import MPI_KerasTrial
                                            trial = MPI_KerasTrial(archive_dir, name=name, model=model, workers=workers,seed=0)
                                            t,v= train_dps,val_dps


                                        trial.set_train(train_procedure=t,  # train_dps,
                                                        samples_per_epoch=_num_train
                                                        )
                                        trial.set_validation(val_procedure=v,  # val_dps,
                                                             nb_val_samples=_num_val)

                                        trial.set_compilation(loss=loss,
                                                              optimizer=optimizer,
                                                              metrics=['accuracy']
                                                              )

                                        trial.set_fit_generator(
                                            nb_epoch=nb_epoch,
                                            callbacks=[earlyStopping],
                                            max_q_size=max_q_size)
                                        trial.write()

                                        trial_tups.append((trial, None, None, dependencies))

                                        trial.to_record({"labels": labels,
                                                         "depth": depth,
                                                         "sort_on": sort_on,
                                                         "sort_ascending": sort_ascending,
                                                         "activation": activation_name,
                                                         "dropout": dropout,
                                                         "lstm_dropout": lstm_dropout,
                                                         "query": None,
                                                         "patience": patience,
                                                         "single_list" : single_list,
                                                         #"useObjTypeColumn": True,
                                                         "output_activation": output_activation
                                                         # "Non_MPI" :True
                                                         })
    # trial_tups[0][0].summary()
    # trial_tups[0][0].execute()
    for tup in trial_tups:
        tup[0].summary()
    # for tup in trial_tups:
	# tup[0].summary()
     #    tup[0].execute()
    batchExecuteAndTestTrials(trial_tups, time_str="1:00:00")
    
if __name__ == '__main__':
    argv = sys.argv
    runTrials(argv[1], int(argv[2]), batchProcesses=int(argv[3]) if len(argv) >= 4 else 4)
