#%matplotlib inline
import unittest
import tempfile

import sys, os
if __package__ is None:
    print("BOOOGALLOO")
    sys.path.append(os.path.realpath("/data/shared/Software/"))
# print(__package__)
# sys.exit()
if not os.path.realpath("../") in sys.path:
    sys.path.append(os.path.realpath("../"))

import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Merge
from keras.models import Sequential
from keras.utils.np_utils import to_categorical

from CMS_Deep_Learning.storage.archiving import *

from keras.callbacks import Callback
class SneakyStopping(Callback):
    def __init__(self, stop_epoch):
        self.stop_epoch = stop_epoch
        super(SneakyStopping, self).__init__()

    def on_train_begin(self, logs=None):
        self.curr = 0 
    def on_epoch_begin(self, epoch, logs=None):
        self.curr += 1
        if(self.curr > self.stop_epoch):
           raise Exception("SNEAKY HAS STOPPED THE TRAINING at epoch %r!" % epoch)
# plot(model, to_file='model3.png', show_shapes=True, show_layer_names=False)
# display(Image("model3.png"))


archive_dir = tempfile.mkdtemp()
other_tempFile = tempfile.mkdtemp()

# Define callback
earlystopping = EarlyStopping(patience=10, verbose=1)

def simpleModel(input_dim=24):
    
    # Make two input branches
    left_branch = Sequential()
    left_branch.add(Dense(10, input_dim=input_dim))  # , name=namer.get('dense')))
    right_branch = Sequential()
    right_branch.add(Dense(10, input_dim=input_dim))  # , name=namer.get('dense')))
    merged = Merge([left_branch, right_branch], mode='concat')  # , name=namer.get('merge'))
    
    # Make two layer dense model
    model = Sequential()
    model.add(merged)
    model.add(Dense(10, activation='softmax'))  # , name=namer.get('dense')))
    # model.add(Dense(10, activation='softmax'))  # , name=namer.get('dense')))
    return model

# Define a function for our DataProcedure. Note: it must return X,Y
def myGetXY(thousand, one, b=24, d=10):
    data_1 = np.random.random((thousand, b))
    data_2 = np.random.random((thousand, b))
    labels = np.random.randint(d, size=(thousand, one))
    labels = to_categorical(labels, d)
    X = [data_1, data_2]
    Y = labels
    return X, Y
    
# Define a list of two DataProcedures for the model to be fit on one after the other 
# We include as arguments to DataProcedures the function that generates our training data its arguments
data = [DataProcedure(archive_dir, True, myGetXY, 500, 1, b=24, d=10) for i in range(2)]
val_data = DataProcedure(archive_dir, True, myGetXY, 500, 1, b=24, d=10)
# for d in data+[val_data]:
#     d.get_data()
# from 
# for d in val_data:
#     d.get_data()

def myGen(dps, batch_size):
    if (isinstance(dps, list) == False): dps = [dps]
    for dp in dps:
        if (isinstance(dp, DataProcedure) == False):
            raise TypeError("Only takes DataProcedure got" % type(dp))
    while True:
        for i in range(0, len(dps)):
            X, Y = dps[i].get_data(verbose=0)
            if (isinstance(X, list) == False): X = [X]
            if (isinstance(Y, list) == False): Y = [Y]
            tot = Y[0].shape[0]
            assert tot == X[0].shape[0], "X shape: %r, Y shape: %r" % (X[0].shape[0], tot)
            for start in range(0, tot, batch_size):
                end = start + min(batch_size, tot - start)
                yield [x[start:end] for x in X], [y[start:end] for y in Y]
                
def removeEverything(m):
    trials = get_all_trials(archive_dir)
    for t in trials:
        t.summary()
        t.remove_from_archive()
    trials = get_all_trials(archive_dir)
    m.assertTrue(len(trials) == 0, "Not all filed Deleted")

    dps = get_all_data(archive_dir)
    for dp in dps:
        dp.summary()
        dp.remove_from_archive()

    dps = get_all_data(archive_dir)
    m.assertTrue(len(dps) == 0, "Not all data deleted")


class TestDelphesParser(unittest.TestCase):
    
    def testGeneral(self):
        # global myGetXY, myGen
        model = simpleModel()
        # Build our KerasTrial object and name it
        trial = KerasTrial(archive_dir, name="MyKerasTrial", model=model)
        # Set the training data
        trial.set_train(train_procedure=data)
        trial.set_validation(.2)
        # Set the compilation paramters
        trial.set_compilation(optimizer='rmsprop',
                              loss='categorical_crossentropy',
                              metrics=['accuracy']
                              )
        # Set the fit parameters
        trial.set_fit(callbacks=[earlystopping], nb_epoch=2)

        # If we like we can store information about the trial so we can keep track of what we've done
        trial.to_record({"stuff1": 100, "data_stuff": "Regular"})

        print(trial.hash())
        # trial.to_hashable()
        # raise ValueError()


        trial.execute(verbosity=0)
        # print()
        trial.summary()
        print("OK IT FINISHED!")

        self.assertIsNotNone(KerasTrial.find(archive_dir, trial.hash()))

        # Luckily no information was lost. We can still get the training history for the trial.
        history = trial.get_history()
        # plot_history([('myhistory', history)])

        test_pp = [DataProcedure(archive_dir, True, myGetXY, 500, 1, b=24, d=10) for i in range(2)]
        # test_X, test_Y = test_pp.getXY()
        # And even the model and weights are still intact
        model = trial.compile(loadweights=True)
        ev = trial.test(test_pp)
        loss = ev[0]
        accuracy = ev[1]

        print('\n')
        print("Test_Loss:", loss)
        print("Test_Accuracy:", accuracy)
        removeEverything(self)

    def testGenerator(self):
        # global myGetXY, myGen
        model = simpleModel()

        train_proc = DataProcedure(archive_dir, True, myGen, data, 100)

        # Build our KerasTrial object and name it
        trial = KerasTrial(archive_dir, name="MyKerasTrial", model=model)
        # Set the training data
        trial.set_train(train_procedure=train_proc,
                        samples_per_epoch=500)
        trial.set_validation(train_proc, nb_val_samples=100)
        # Set the compilation paramters
        trial.set_compilation(optimizer='rmsprop',
                              loss='categorical_crossentropy',
                              metrics=['accuracy']
                              )
        # Set the fit parameters
        trial.set_fit_generator(callbacks=[earlystopping], nb_epoch=2)

        # If we like we can store information about the trial so we can keep track of what we've done
        trial.to_record({"stuff1": 777, "data_stuff": "Generator"})

        # Execute the trial running fitting on each DataProcedure in turn 
        trial.execute(verbosity=0)
        print("OK IT FINISHED!")

        ev = trial.test(train_proc, 1000)
        loss = ev[0]
        accuracy = ev[1]

        print('\n')
        print("Test_Loss:", loss)
        print("Test_Accuracy:", accuracy)
        removeEverything(self)
    def testTrainingContinuation(self):
        # global myGetXY,myGen
        model = simpleModel()
        # Build our KerasTrial object and name it
        trial = KerasTrial(archive_dir, name="MyKerasTrial", model=model)
        # Set the training data
        trial.set_train(train_procedure=data)
        trial.set_validation(.2)
        # Set the compilation paramters
        trial.set_compilation(optimizer='rmsprop',
                              loss='categorical_crossentropy',
                              metrics=['accuracy']
                              )
        # Set the fit parameters
        sneakyStopping = SneakyStopping(2)
        print([sneakyStopping])
        trial.set_fit(callbacks=[sneakyStopping], nb_epoch=10)

        while trial.is_complete() == False:
            try:
                trial.execute()
            except Exception as e:
                print(e)
                continue

    def testGenHash(self):
        trials = []
        codes = []
        genCodes = []
        for i in range(10):
            model = simpleModel()
            trial = KerasTrial(archive_dir, name="MyKerasTrial", model=model, seed=i)
            trial.set_train(train_procedure=data)
            trial.set_validation(.2)
            trial.set_compilation(optimizer='rmsprop',
                                  loss='categorical_crossentropy',
                                  metrics=['accuracy']
                                  )
            trial.set_fit(nb_epoch=10)
            trials.append(trial)
            codes.append(trial.hash())
            genCodes.append(trial.gen_hash())
            print(trial.get_path())
            trial.write()
        # print(codes,genCodes)
        paths = sorted(Storable.get_all_paths(archive_dir))
        paths_indv = sorted([t.get_path() for t in trials]) #+ [data.]
        print(paths)
        print(paths_indv)
        self.assertListEqual(paths,paths_indv)
        #print("LENGTH:", paths, paths_indv)
        for path in paths:
             self.assertTrue(os.path.isdir(path))
            

        self.assertTrue(not False in [c == genCodes[0] for c in genCodes])
        self.assertTrue(len(codes) == len(set(codes)))
        
    def testEnvironmentVars(self):
        os.environ["TEST_ENV_VAR"] = archive_dir
        
        model = simpleModel()
        trial = KerasTrial("$TEST_ENV_VAR", name="MyKerasTrial", model=model, seed=i)
        trial.set_train(train_procedure=data)
        trial.set_validation(.2)
        trial.set_compilation(optimizer='rmsprop',
                              loss='categorical_crossentropy',
                              metrics=['accuracy']
                              )
        trial.set_fit(nb_epoch=10)
        hash1 = trial.hash()
        path1 = trial.get_path()

        os.environ["TEST_ENV_VAR"] = other_tempFile
        model = simpleModel()
        trial = KerasTrial("$TEST_ENV_VAR", name="MyKerasTrial", model=model, seed=i)
        trial.set_train(train_procedure=data)
        trial.set_validation(.2)
        trial.set_compilation(optimizer='rmsprop',
                              loss='categorical_crossentropy',
                              metrics=['accuracy']
                              )
        trial.set_fit(nb_epoch=10)
        hash2 = trial.hash()
        path2 = trial.get_path()
        
        print(hash1, hash2)
        print(path1, path2)
        self.assertTrue(hash1 == hash2)
        # self.assertFalse()

        
        
    
                
        


if __name__ == '__main__':
    unittest.main()