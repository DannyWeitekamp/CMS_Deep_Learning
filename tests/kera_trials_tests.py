%matplotlib inline
import sys, os
if __package__ is None:
	import sys, os
	sys.path.append(os.path.realpath("/data/shared/Software/"))
from CMS_SURF_2016.utils.archiving import PreprocessingProcedure, KerasTrial, get_all_preprocessing, get_all_trials
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

trial_dir = 'MyTrialDir/'

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

#Define a function for our PreprocessingProcedure. Note: it must return X,Y
def myGetXY(thousand, one, b=784, d=10):
	data_1 = np.random.random((thousand, b))
	data_2 = np.random.random((thousand, b))
	labels = np.random.randint(d, size=(thousand, one))
	labels = to_categorical(labels, d)
	X = [data_1, data_2]
	Y = labels
	return X, Y

#Define a list of two preprocessing procedures for the model to be fit on one after the other 
#We include as arguments to PreprocessingProcedure the function that generates our training data its arguments
preprocessing = [PreprocessingProcedure(trial_dir, myGetXY, 1000, 1, b=784, d=10) for i in range(2)]

#Build our KerasTrial object and name it
trial = KerasTrial(trial_dir, name="MyKerasTrial", model=model)
#Set the preprocessing data
trial.setPreprocessing(preprocessing)
#Set the compilation paramters
trial.setCompilation(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy']
              )
#Set the fit parameters
trial.setFit(callbacks = [earlystopping], nb_epoch=19, batch_size=32, validation_split=.2)

#Execute the trial running fitting on each preprocessing procedure in turn 
trial.execute()
print("OK IT FINISHED!")

from keras.utils.visualize_util import plot
from IPython.display import Image, display
#Luckily no information was lost. We can still get the training history for the trial.
history = trial.get_history()
plot_history([('myhistory', history)])

test_pp = PreprocessingProcedure(trial_dir, myGetXY, 1000, 1, b=784, d=10)
test_X, test_Y = test_pp.get_XY()
#And even the model and weights are still intact
model = trial.compile(loadweights=True)
ev = model.evaluate(test_X, test_Y)
loss = ev[0]
accuracy = ev[1]

print('\n')
print("Test_Loss:",loss)
print("Test_Accuracy:",accuracy)

plot(model, to_file='model3.png', show_shapes=True, show_layer_names=False)
display(Image("model3.png"))

trials = get_all_trials(trial_dir)
for t in trials:
	t.summary()
	t.remove_from_archive()
trials = get_all_trials(trial_dir)
print("Deleted all trials?:", len(trials) == 0)

pps = get_all_preprocessing(trial_dir)
for p in pps:
	p.summary()
	p.remove_from_archive()

pps = get_all_preprocessing(trial_dir)
print("Deleted all preprocessing?:", len(pps) == 0)