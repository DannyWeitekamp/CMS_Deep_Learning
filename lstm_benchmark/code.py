
# coding: utf-8

# In[9]:

'''Compare LSTM implementations on the IMDB sentiment classification task.

consume_less='cpu' preprocesses input to the LSTM which typically results in
faster computations at the expense of increased peak memory usage as the
preprocessed input must be kept in memory.

consume_less='mem' does away with the preprocessing, meaning that it might take
a little longer, but should require less peak memory.

consume_less='gpu' concatenates the input, output and forget gate's weights
into one, large matrix, resulting in faster computation time as the GPU can
utilize more cores, at the expense of reduced regularization because the same
dropout is shared across the gates.

Note that the relative performance of the different `consume_less` modes
can vary depending on your device, your model and the size of your data.
'''
#NOOOOOOOOP
# from IPython.core import debugger
# from IPython import get_ipython
# get_ipython().magic(u'matplotlib inline')
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json


from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Embedding, Dense, LSTM
from keras.datasets import imdb
from keras.callbacks import History
from keras.callbacks import ModelCheckpoint

class SmartCheckpoint(ModelCheckpoint):
    def __checkMaxEpoch(self, epoch):
        if(epoch > self.max_epoch):
            self.model.stop_training = True

    def __init__(self, name, directory='', monitor='val_loss', verbose=0,
                 save_best_only=False, mode='auto', max_epoch = sys.maxint):
        self.name = name
        self.smartDir = directory + 'SmartCheckpoint/'
        self.checkpointFilename = self.smartDir + name + "_weights.hdf5"
        self.historyFilename = self.smartDir + name + "_history.json"
        self.max_epoch = max_epoch
        self.histDict = {}
        

     


        ModelCheckpoint.__init__(self, self.checkpointFilename,
                monitor, verbose, save_best_only, mode)

        # self.model.load_weights(self.checkpointFilename);

    # def on_epoch_begin(self, epoch, logs={}):

    def on_train_begin(self, logs={}):
        if not os.path.exists(self.smartDir):
            os.makedirs(self.smartDir)

        print('Load history?');
        try:
            self.histDict = json.load(open( self.historyFilename, "rb" ))
            print(True)
        except (IOError, EOFError):
            print(False)
            # history = History()
        # history.history = histDict
        self.epochOffset = self.histDict.get("last_epoch", 0);
        self.__checkMaxEpoch(self.max_epoch + self.epochOffset)
        print('Load weights?');
        # weightsloaded = False
        try:
            self.model.load_weights(self.checkpointFilename)
            # weightsloaded = True
            print(True)
        except IOError:
            print(False)

    def on_epoch_end(self, epoch, logs={}):
        epoch = epoch + self.epochOffset + 1
        ModelCheckpoint.on_epoch_end(self, epoch, logs)
        self.histDict["last_epoch"] = epoch
        self.histDict["epoch_" + str(epoch)] = logs.copy();
        json.dump(self.histDict,  open( self.historyFilename, "wb" ))

        self.__checkMaxEpoch(epoch + self.epochOffset)
        



name = "lstm_benchmark"
savedir = "hist"
namePath = savedir + "/" + name
max_features = 20000
max_length = 5
embedding_dim = 256
batch_size = 128
epochs = 5
modes = ['gpu']

print('Loading data...')
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features)
X_train = sequence.pad_sequences(X_train, max_length)
X_test = sequence.pad_sequences(X_test, max_length)

# Compile and train different models while meauring performance.
results = []
for mode in modes:
    print('Testing mode: consume_less="{}"'.format(mode))

    model = Sequential()
    model.add(Embedding(max_features, embedding_dim, input_length=max_length, dropout=0.2))
    model.add(LSTM(embedding_dim, dropout_W=0.2, dropout_U=0.2, consume_less=mode))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    start_time = time.time()
    # historyFilePath = namePath + "_" + mode + "_" + "history.best.p"
    # weightsfilepath = namePath + "_" + mode + "_" + "weights.best.hdf5"
   
    # histDict = None
    # print('Load history?');
    # try:
    #     histDict = json.load(open( historyFilePath, "rb" ))
    #     print(True)
    # except (IOError, EOFError):
    #     print(False)
    #     # history = History()
    # # history.history = histDict

    # print('Load weights?');
    # weightsloaded = False
    # try:
    #     model.load_weights(weightsfilepath)
    #     weightsloaded = True
    #     print(True)
    # except IOError:
    #     print(False)

    # model.load_weights('hist/lstm_benchmark_gpu_weights.h5')
    # print('Load weights?');
    # print(model.get_weights() != False);
   
    # print(model.get_weights());

    # model.stop_training = True
    # if(model.stop_training != True):
        
    checkpoint = SmartCheckpoint(name,
                                monitor='val_acc',
                                verbose=1,
                                save_best_only=True)
    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        nb_epoch=epochs,
                        validation_data=(X_test, y_test),
                        callbacks=[checkpoint])
    # histDict = history.history
        # model.save_weights()

    # print()
    
    # json.dump(histDict,  open( historyFilePath, "wb" ))
    average_time_per_epoch = (time.time() - start_time) / epochs

    #results.append((history, average_time_per_epoch))

# Compare models' accuracy, loss and elapsed time per epoch.
# plt.style.use('ggplot')
# ax1 = plt.subplot2grid((2, 2), (0, 0))
# ax1.set_title('Accuracy')
# ax1.set_ylabel('Validation Accuracy')
# ax1.set_xlabel('Epochs')
# ax2 = plt.subplot2grid((2, 2), (1, 0))
# ax2.set_title('Loss')
# ax2.set_ylabel('Validation Loss')
# ax2.set_xlabel('Epochs')
# ax3 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
# ax3.set_title('Time')
# ax3.set_ylabel('Seconds')
# for mode, result in zip(modes, results):
#     ax1.plot(result[0].epoch, result[0].history['val_acc'], label=mode)
#     ax2.plot(result[0].epoch, result[0].history['val_loss'], label=mode)
# ax1.legend()
# ax2.legend()
# ax3.bar(np.arange(len(results)), [x[1] for x in results],
#         tick_label=modes, align='center')
# plt.tight_layout()
# plt.show()



