import sys,types,os,glob
os.environ["CUDA_VISIBLE_DEVICES"] = str(sys.argv[4])
from six import string_types
#import site
#sys.path = [site.USER_BASE] + sys.path
if __package__ is None:
    sys.path.append(os.path.realpath("/data/shared/Software/"))
    sys.path.append(os.path.realpath("../"))
print(sys.path)
#Prevent Theano from finding this for this front end script
#sys.modules['mpi4py']=None

# p = "/home/dweitekamp/mpi_learn/"
# if(not p in sys.path):
#     sys.path.append(p)

# from CMS_Deep_Learning.utils.deepconfig import deepconfig
# deepconfig("cpu", backend="theano")

from CMS_Deep_Learning.preprocessing.preprocessing import DataProcedure
from CMS_Deep_Learning.storage.archiving import KerasTrial
# from CMS_Deep_Learning.storage.MPIArchiving import *
from CMS_Deep_Learning.postprocessing.analysistools import findsubsets

from keras.models import Model
from keras.layers import Dense, Dropout, merge, Input, LSTM, Masking,GRU
from keras.callbacks import EarlyStopping
from keras import regularizers

def build_LSTM_model(name, input_width,out_width, depth,recurrent_width, lstm_activation="tanh", lstm_dropout = 0.0,
                     dropout=0.0, output_activation="softmax", single_list=False, l1_reg=0.0, **kargs):
    inputs = []
    a = Input(shape=(None , input_width), name="input")
    inputs.append(a)
    for i in range(depth):
        a = Masking(mask_value=0.0)(a)
        a = GRU(recurrent_width,
                 input_shape=(None, input_width),
                 dropout_W=lstm_dropout,
                 dropout_U=lstm_dropout,
                 activation=lstm_activation,
                 #implementation=2,
		 W_regularizer=regularizers.l1(l1_reg),
                 name = "gru_" +str(i))(a)
        if(dropout > 0.0):
            a =  Dropout(dropout, name="dropout_"+str(i))(a)
    dense_out = Dense(out_width, activation=output_activation, name='main_output')(a)
    model = Model(input=inputs, output=dense_out, name=name)
    return model

from CMS_Deep_Learning.preprocessing.pandas_to_numpy import PARTICLE_OBSERVS


def _readNumSamples(file_path):
    import h5py
    try:
        f = h5py.File(file_path, 'r')
        out = f["HLF"].len()
    except IOError as e:
        print("Something wrong with file %r" % file_path)
        raise e
    f.close()
    return out

def assert_dataset(data, nb_data=None, as_generator=False,**kargs):
    from CMS_Deep_Learning.preprocessing.pandas_to_numpy import size_from_meta,get_sizes_meta_dict
    if(isinstance(data, string_types)):
        data = glob.glob(os.path.abspath(data) + "/*.h5")
    data_dir = data[0].split("/")[-2]
    sizesDict = get_sizes_meta_dict(data_dir)
    actual_amount = sum([size_from_meta(x, sizesDict=sizesDict) for x in data])
    if(nb_data != None):
        if(nb_data > actual_amount):
            raise IOError("Not enough data in %r, requested %r"
                " but there are only %r samples" % (data_dir, nb_data,actual_amount) )
    else:
        nb_data = actual_amount
    if(as_generator):
        from CMS_Deep_Learning.preprocessing.preprocessing import gen_from_data
        data = DataProcedure(kargs['archive_dir'],False,func=gen_from_data, kargs={'lst':data, 'batch_size':kargs['batch_size']})
    return data, nb_data
    

def build_trial(name,
                model,
                train,
                val,
                archive_dir=None,
                nb_train=None,
                nb_val=None,
                workers=1,
                loss='categorical_crossentropy',
                optimizer= 'rmsprop',
                metrics = ['accuracy'],
                nb_epoch=10,
                callbacks=None,
                max_q_size=100,
                keys_to_record = [],
                **kargs):
    if isinstance(model, types.FunctionType):
        model = model(**kargs)
    if (workers == 1):
        trial = KerasTrial(archive_dir, name=name, model=model, seed=0)
        val, nb_val = assert_dataset(val, nb_data=nb_val, as_generator=True,archive_dir=archive_dir,**kargs)
        train, nb_train = assert_dataset(train, nb_train, as_generator=True,archive_dir=archive_dir,**kargs)
    else:
        print("USING MPI")
        p = "../../mpi_learn"
        if not p in sys.path:
            sys.path.append(p)
        from CMS_Deep_Learning.storage.MPIArchiving import MPI_KerasTrial
        trial = MPI_KerasTrial(archive_dir, name=name, model=model, workers=workers, seed=0, features_name="Particles",labels_name="Labels")
        val, nb_val = assert_dataset(val, nb_data=nb_val,archive_dir=archive_dir,**kargs)
        train, nb_train = assert_dataset(train, nb_train,archive_dir=archive_dir,**kargs)

    
    trial.set_train(train_procedure=train,  # train_dps,
                    samples_per_epoch=nb_train
                    )
    trial.set_validation(val_procedure=val,  # val_dps,
                         nb_val_samples=nb_val)

    trial.set_compilation(loss=loss,
                          optimizer=optimizer,
                          metrics=metrics
                          )

    trial.set_fit_generator(
        nb_epoch=nb_epoch,
        callbacks=callbacks,
        max_q_size=max_q_size)
    trial.write()

    trial.to_record({k:kargs[k] for k in keys_to_record})
    return  trial

def assert_write_datasets(sort_on,sort_ascending,dataset_dir='/bigdata/shared/Delphes/np_datasets', processes=1):
    from CMS_Deep_Learning.preprocessing.pandas_to_numpy import make_datasets
    dir = dataset_dir + "/" + sort_on + '_' + ('asc' if sort_ascending else 'dec')
    dir = os.path.abspath(dir)
    if(not os.path.exists(dir)):
        raise IOError("NOPE")
        sources = ['/bigdata/shared/Delphes/REDUCED_IsoLep/ttbar_lepFilter_13TeV','/bigdata/shared/Delphes/REDUCED_IsoLep/wjets_lepFilter_13TeV']
        make_datasets(sources, output_dir=dir, num_samples=120000, size=10000,
                      num_processes=processes, sort_on=sort_on, sort_ascending=sort_ascending,
                      v_split=20000, force=False)
    return dir
        

# def assertDirectoryStruct(ldps, sort_on, sort_ascending):
#     
    
def KFold(a,k,tokeep):
    omit = len(a)-tokeep
    print(omit)
    omittions = [ [int(float(len(a)-omit)/(k-1) * j) + i for i in range(omit)] for j in range(k)] 
    print(omittions)
    return [[x for i,x in enumerate(a) if not i in omittions[j]  ] for j in range(k)]


def trials_from_HPsweep(archive_dir,
              K_items,
              delphes_dir=None,
              nb_epoch = 5,
              batch_size = 100,
              patience = 8,
              output_activation = "softmax",
              loss='categorical_crossentropy',
              optimizer_options = ['adam'],
              sortings = [("MaxLepDeltaPhi", False) ,("MaxLepDeltaEta", False),
                         # ("PT_ET", False), ("PT_ET", True),
                         #[ ('MaxLepDeltaR', False), ('MaxLepDeltaR', True), ('random', False)],
                          ('MaxLepKt',False),('MaxLepKt',True)],
                         # ('MaxLepAntiKt',False),('MaxLepAntiKt',True),
                         # ('shuffle',False)],#, ('METDeltaR', False), ('METKt',False), ('METAntiKt',False),
                            #("METDeltaPhi", False), ("METDeltaEta", False)],
                n_train_files = [50],
                n_val_files = 5
              ):
    # if(delphes_dir == None):
    #     split = list(archive_dir.split("/"))
    #     delphes_dir = "/".join( split[:split.index("Delphes")+1] )
    #     
    # os.environ["DELPHES_DIR"] = delphes_dir
    # 
    earlyStopping = EarlyStopping(verbose=1, patience=patience)
    #print(archive_dir, workers)
    
    labels = ['qcd','ttbar', 'wjets']
    
    trials = []
    for sort_on, sort_ascending in sortings:
        # data_dir = assert_write_datasets(sort_on,sort_ascending)
        data_up_dir = '/bigdata/shared/Delphes/np_datasets/3_way/'
        data_dir = os.path.abspath(data_up_dir + sort_on + ("_asc" if sort_ascending else '_des'))
        
        if(sort_on in ['random','shuffled']): data_dir = os.path.abspath(data_up_dir + sort_on)
        assert os.path.exists(data_dir), "%r nope" % data_dir
        for optimizer in optimizer_options:
            for depth in [1]:
                
                for activation in ['tanh']:
                    activation_name = activation if isinstance(activation, string_types) \
                        else activation.__name__
                    for lstm_dropout in [0.0]:
#                        swit = int(sys.argv[5])
                        #l1_sets = [[0.01,0.025],[0.05,0.075],[0.1,0.25]]
                        #for data_dir in data_dirs:#l1_sets[swit]
                        for l1_reg in [0.0]:#l1_sets[swit]:

                            dropout = 0.0
                            for ntf in n_train_files:
                                g = sorted(glob.glob(data_dir + "/train/*.h5"))
                                train = g[:ntf] if ntf > 0 else g 
                                val = sorted(glob.glob(data_dir + "/val/*.h5")[:n_val_files])
                                for train in KFold(g[:ntf], int(K_items), 40):
                                    print("LEN TRAIN:",len(train))
                                    assert len(train) > 0, "nope. bad"
                                    def f(**kargs):
                                        return kargs
                                    inps = f(name='GRU', 
                                         input_width=len(PARTICLE_OBSERVS),
                                         out_width=3,
                                         depth=1,
                                         recurrent_width=50,
                                         lstm_activation=activation,
                                         lstm_dropout=lstm_dropout,
                                         dropout=dropout,
                                         train=train,#data_dir + "/train",
                                         val=val,#data_dir + "/val"t,
                                         archive_dir=archive_dir,
                                         workers=1,
                                         optimizer=optimizer,
                                         nb_epoch=100,
                                         batch_size=batch_size,
                                         callbacks=[earlyStopping],
                                         keys_to_record=['labels', 'depth', 'sort_on', 'sort_ascending','l1_reg','recurrent_width',
                                                         'activation', 'dropout', 'lstm_dropout',
                                                         'patience', "n_train_files"],
                                         sort_on=sort_on,
                                         sort_ascending=sort_ascending,
                                         activation=activation,
                                         labels=labels,
                                         n_train_files=40,#ntf,
                                         l1_reg=l1_reg,
                                         patience=patience
                                         )
                                    model = build_LSTM_model(**inps)
                                    inps["model"] = model
                                    trial = build_trial(**inps)
                            
                                
                                    trials.append(trial)
    return trials                            
                            

                                    
# trial_tups[0][0].summary()
    # trial_tups[0][0].execute()
    #batchExecuteAndTestTrials(trial_tups, time_str="1:00:00", use_mpi=workers > 1)
    
if __name__ == '__main__':
    argv = sys.argv
    #os.environ["CUDA_VISIBLE_DEVICES"] = str(argv[3])
    trials = trials_from_HPsweep(argv[1], int(argv[2]))
    trials = [trials[i] for i in range(int(argv[3]),len(trials),int(argv[2]))]
    for trial in trials:
        print(trial.summary())
    for trial in trials:
        print(trial.summary())
        trial.execute()
