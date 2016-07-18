import sys, os
if __package__ is None:
	import sys, os
	sys.path.append(os.path.realpath("../../"))
from CMS_SURF_2016.utils.archiving import *
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Reshape, Activation, Dropout, Convolution2D, Merge, Input
from keras.callbacks import EarlyStopping
from CMS_SURF_2016.utils.callbacks import *
import numpy as np

trial_dir = 'MyTrialDir/'

earlystopping = EarlyStopping(patience=10, verbose=1)
# smartCheckpoint = SmartCheckpoint('moop')
# model = Sequential()
# model.add(Dense(10, input_dim=10))
# model.add(Dense(1, activation='sigmoid'))

left_branch = Sequential()
left_branch.add(Dense(32, input_dim=784))

right_branch = Sequential()
right_branch.add(Dense(32, input_dim=784))

merged = Merge([left_branch, right_branch], mode='concat')

model = Sequential()
model.add(merged)
model.add(Dense(10, activation='softmax'))
model.add(Dense(10, activation='softmax'))



import numpy as np
from keras.utils.np_utils import to_categorical

def myGetXY(thousand, one, b=784, d=10):
	data_1 = np.random.random((thousand, b))
	data_2 = np.random.random((thousand, b))
	# these are integers between 0 and 9
	labels = np.random.randint(d, size=(thousand, one))
	# we convert the labels to a binary matrix of size (1000, 10)
	# for use with categorical_crossentropy
	labels = to_categorical(labels, d)
	X = [data_1, data_2]
	Y = labels
	return X, Y

preprocessing = [PreprocessingProcedure(trial_dir, myGetXY, 1000, 1, b=784, d=10) for i in range(2)]

trial = KerasTrial(trial_dir, name="Duffles", model=model)
trial.setPreprocessing(preprocessing)
trial.setCompilation(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy']
              )
trial.setFit(callbacks = [earlystopping], nb_epoch=110, batch_size=32, validation_split=.2)
# write_trial(trial, trial_dir)
# trial.write()


# 
# X,Y = cache.get_XY(redo=True)

# print(X[0].shape, Y.shape)
# print(cache.to_json())
# print(cache.hash())


trial.execute()

js_str = trial.to_json()
# print(js_str)
trial = None
trial = KerasTrial.from_json(trial_dir, js_str)
trial.execute()
# trial.summary()

# trials = get_trials_by_name('Duff*', trial_dir)
trials = get_all_trials(trial_dir)
print(trials)
for t in trials:
	t.summary()


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
