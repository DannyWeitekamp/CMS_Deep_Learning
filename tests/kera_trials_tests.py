import sys, os
if __package__ is None:
	import sys, os
	sys.path.append(os.path.realpath("../../"))
from CMS_SURF_2016.utils.archiving import PreprocessingProcedure, KerasTrial, get_all_preprocessing, get_all_trials
from CMS_SURF_2016.utils.metrics import plot_history
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Reshape, Activation, Dropout, Convolution2D, Merge, Input
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical
import numpy as np


trial_dir = 'MyTrialDir/'

#Define callback
earlystopping = EarlyStopping(patience=10, verbose=1)

#Make two input branches
left_branch = Sequential()
left_branch.add(Dense(32, input_dim=784))
right_branch = Sequential()
right_branch.add(Dense(32, input_dim=784))
merged = Merge([left_branch, right_branch], mode='concat')

#Make two layer dense model
model = Sequential()
model.add(merged)
model.add(Dense(10, activation='softmax'))
model.add(Dense(10, activation='softmax'))

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

#To demonstrate that we will never rerun the same trial twice we get the trial's json string
js_str = trial.to_json()
#Then delete it
trial = None
#Reload it from that string
trial = KerasTrial.from_json(trial_dir, js_str)
#And then try to run it again. But this time it says that the trial is already done
trial.execute()


from keras.utils.visualize_util import plot
from IPython.display import Image, display
#Luckily no information was lost. We can still get the training history for the trial.
history = trial.get_history()
plot_history([('myhistory', history)])

#And even the model and weights are still intact
model = trial.get_model(loadweights=True)

plot(model, to_file='model3.png', show_shapes=True, show_layer_names=False)
display(Image("model3.png"))



# trial.summary()

# trials = get_trials_by_name('Duff*', trial_dir)
trials = get_all_trials(trial_dir)
print(trials)
for t in trials:
	t.summary()
	t.remove_from_archive()

pps = get_all_preprocessing(trial_dir)
for p in pps:
	p.summary()
	p.remove_from_archive()

pps = get_all_preprocessing(trial_dir)
for p in pps:
	p.summary()
# model = trial.compile()
# trial.fit(model, x_train, y_train)
# if(trial.is_complete() == False):
# 	trial.fit(model,X, Y)
# else:
	# print("Trial %r Already Complete" % trial.hash())
# train the model
# note that we are passing a list of Numpy arrays as training data
# since the model has 2 inputs



# x_train = np.random.random((100, 10))
# y_train = np.append(np.ones(50,dtype='float64'),np.zeros(50,dtype='float64'))



# print(trial.to_json())

# json_str = trial.to_json()
# print("JSON_STR:", json_str)
# hashcode1 = compute_hash(trial)
# print("Hashcode1:", hashcode1)
# hashcode2 = compute_hash(json_str)
# print("Hashcode2", hashcode2)
# blob_path1 = get_blob_path(trial, trial_dir)
# print("Blob_path1:", blob_path1)
# blob_path2 = get_blob_path(hashcode1, trial_dir)
# print("Blob_path2:", blob_path2)

# blob_path1 = get_blob_path(trial, trial_dir=trial_dir)
# print("Blob_path1:", blob_path1)
# blob_path2 = get_blob_path(hashcode1, trial_dir=trial_dir)
# print("Blob_path2:", blob_path2)




# print(trial.to_json())
# print(compute_hash(trial))


#USE THIS:::




# for a multi-input model with 10 classes:





# generate dummy data
