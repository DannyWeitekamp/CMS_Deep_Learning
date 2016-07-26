#%matplotlib inline
import sys, os
if __package__ is None:
	import sys, os
	sys.path.append(os.path.realpath("/data/shared/Software/"))
	sys.path.append(os.path.realpath("../.."))

from CMS_SURF_2016.utils.archiving import DataProcedure, KerasTrial, get_all_data, get_all_trials
from CMS_SURF_2016.utils.metrics import plot_history
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Reshape, Activation, Dropout, Convolution2D, Merge, Input
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical
import numpy as np


class LayerNamer:
    def __init__(self):
        self.i = 0
    def get(self,name='layer'):
        self.i += 1
        return str(name) + '_' + str(self.i)

namer = LayerNamer()

archive_dir = 'MyTrialDir/'

#Define callback
earlystopping = EarlyStopping(patience=10, verbose=1)

#Warning it's important to name all of your layers otherwise the hashing won't work

#Make two input branches
left_branch = Sequential()
left_branch.add(Dense(32, input_dim=784, name=namer.get('dense')))
right_branch = Sequential()
right_branch.add(Dense(32, input_dim=784, name=namer.get('dense')))
merged = Merge([left_branch, right_branch], mode='concat', name=namer.get('merge'))

#Make two layer dense model
model = Sequential()
model.add(merged)
model.add(Dense(10, activation='softmax', name=namer.get('dense')))
model.add(Dense(10, activation='softmax', name=namer.get('dense')))



# def getGen(dps, batch_size):
# 	return (myGen(dps,batch_size), 1000)



#Define a function for our DataProcedure. Note: it must return X,Y
def myGetXY(thousand, one, b=784, d=10):
	data_1 = np.random.random((thousand, b))
	data_2 = np.random.random((thousand, b))
	labels = np.random.randint(d, size=(thousand, one))
	labels = to_categorical(labels, d)
	X = [data_1, data_2]
	Y = labels
	return X, Y

#Define a list of two DataProcedures for the model to be fit on one after the other 
#We include as arguments to DataProcedures the function that generates our training data its arguments
data = [DataProcedure(archive_dir, True, myGetXY, 1000, 1, b=784, d=10) for i in range(2)]
val_data = DataProcedure(archive_dir, True,myGetXY, 1000, 1, b=784, d=10)

#Build our KerasTrial object and name it
trial = KerasTrial(archive_dir, name="MyKerasTrial", model=model)
#Set the training data
trial.setTrain(train_procedure=data)
trial.setValidation(.2)
#Set the compilation paramters
trial.setCompilation(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy']
              )
#Set the fit parameters
trial.setFit(callbacks = [earlystopping], nb_epoch=18)





trial.execute()
print("OK IT FINISHED!")





from keras.utils.visualize_util import plot
from IPython.display import Image, display
#Luckily no information was lost. We can still get the training history for the trial.
history = trial.get_history()
# plot_history([('myhistory', history)])

test_pp = [DataProcedure(archive_dir,True, myGetXY, 1000, 1, b=784, d=10) for i in range(2)]
# test_X, test_Y = test_pp.getXY()
#And even the model and weights are still intact
model = trial.compile(loadweights=True)
ev = trial.test(test_pp)
loss = ev[0]
accuracy = ev[1]

print('\n')
print("Test_Loss:",loss)
print("Test_Accuracy:",accuracy)

# plot(model, to_file='model3.png', show_shapes=True, show_layer_names=False)
# display(Image("model3.png"))





def myGen(dps, batch_size):
    if(isinstance(dps, list) == False): dps = [dps]
    for dp in dps:
        if(isinstance(dp, DataProcedure) == False):
            raise TypeError("Only takes DataProcedure got" % type(dp))
    while True:
        for i in range(0,len(dps)):            
            X,Y = dps[i].getData()
            if(isinstance(X,list) == False): X = [X]
            if(isinstance(Y,list) == False): Y = [Y]
            tot = Y[0].shape[0]
            assert tot == X[0].shape[0]
            for start in range(0, tot, batch_size):
                end = start+min(batch_size, tot-start)
                yield [x[start:end] for x in X], [y[start:end] for y in Y]

train_proc = DataProcedure(archive_dir,True,myGen,data,100)

#Build our KerasTrial object and name it
trial = KerasTrial(archive_dir, name="MyKerasTrial", model=model)
#Set the training data
trial.setTrain(train_procedure=train_proc,
				samples_per_epoch=1000)
trial.setValidation(train_proc, nb_val_samples=100)
#Set the compilation paramters
trial.setCompilation(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy']
              )
#Set the fit parameters
trial.setFit_Generator(callbacks = [earlystopping], nb_epoch=18)

#Execute the trial running fitting on each DataProcedure in turn 
trial.execute()
print("OK IT FINISHED!")

ev = trial.test(train_proc, 1000)
loss = ev[0]
accuracy = ev[1]

print('\n')
print("Test_Loss:",loss)
print("Test_Accuracy:",accuracy)



trials = get_all_trials(archive_dir)
for t in trials:
	t.summary()
	t.remove_from_archive()
trials = get_all_trials(archive_dir)
print("Deleted all trials?:", len(trials) == 0)

pps = get_all_data(archive_dir)
for p in pps:
	p.summary()
	p.remove_from_archive()

pps = get_all_data(archive_dir)
print("Deleted all data?:", len(pps) == 0)