''' 
callbacks.py
Contains custom callback objects.
Author: Danny Weitekamp
e-mail: dannyweitekamp@gmail.com
''' 


import os
import sys
import json
from keras.callbacks import History
from keras.callbacks import ModelCheckpoint
class SmartCheckpoint(ModelCheckpoint):
    ''' A smart checkpoint callback that automatically saves and loads training history, and weights
        based on the name given at instantiation. Creates a SmartCheckpoint directory that stores
        the data''' 

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
        self.histobj = History()

        histDict = {}
        try:
            histDict = json.load(open( self.historyFilename, "rb" ))
            print('Sucessfully loaded history at ' + self.historyFilename)
        except (IOError, EOFError):
            print('Failed to load history at ' + self.historyFilename)

        self.histobj.history = histDict

        ModelCheckpoint.__init__(self, self.checkpointFilename,
                monitor, verbose, save_best_only, mode)

        # self.model.load_weights(self.checkpointFilename);

    # def on_epoch_begin(self, epoch, logs={}):

    def on_train_begin(self, logs={}):
        histDict = self.histobj.history

        if not os.path.exists(self.smartDir):
            os.makedirs(self.smartDir)
        
        self.epochOffset = histDict.get("last_epoch", 0);
        self.__checkMaxEpoch(self.max_epoch + self.epochOffset)
        try:
            self.model.load_weights(self.checkpointFilename)
            # weightsloaded = True
            print('Sucessfully loaded weights at ' + self.checkpointFilename)
        except (IOError, EOFError):
            print('Failed to load weights at ' + self.checkpointFilename)


    def on_epoch_end(self, epoch, logs={}):
        histDict = self.histobj.history

        epoch = epoch + self.epochOffset + 1
        ModelCheckpoint.on_epoch_end(self, epoch, logs)
        histDict["last_epoch"] = epoch
        # print(histDict)
        # self.histDict["epoch_" + str(epoch)] = logs.copy();
        # self.epoch.append(epoch)
        for k, v in logs.items():
            if k not in histDict:
                histDict[k] = []
            histDict[k].append(v)
        # print(histDict)
        json.dump(histDict,  open( self.historyFilename, "wb" ))

        self.__checkMaxEpoch(epoch + self.epochOffset)


# class EarlyStopping(Callback):
#     '''Stop training when a monitored quantity has stopped improving.
#     # Arguments
#         monitor: quantity to be monitored.
#         patience: number of epochs with no improvement
#             after which training will be stopped.
#         verbose: verbosity mode.
#         mode: one of {auto, min, max}. In 'min' mode,
#             training will stop when the quantity
#             monitored has stopped decreasing; in 'max'
#             mode it will stop when the quantity
#             monitored has stopped increasing.
#     '''
#     def __init__(self, monitor='val_loss', patience=0, verbose=0, mode='auto'):
#         super(EarlyStopping, self).__init__()

#         self.monitor = monitor
#         self.patience = patience
#         self.verbose = verbose
#         self.wait = 0

#         if mode not in ['auto', 'min', 'max']:
#             warnings.warn('EarlyStopping mode %s is unknown, '
#                           'fallback to auto mode.' % (self.mode), RuntimeWarning)
#             mode = 'auto'

#         if mode == 'min':
#             self.monitor_op = np.less
#         elif mode == 'max':
#             self.monitor_op = np.greater
#         else:
#             if 'acc' in self.monitor:
#                 self.monitor_op = np.greater
#             else:
#                 self.monitor_op = np.less

#     def on_train_begin(self, logs={}):
#         self.wait = 0       # Allow instances to be re-used
#         self.best = np.Inf if self.monitor_op == np.less else -np.Inf

#     def on_epoch_end(self, epoch, logs={}):
#         current = logs.get(self.monitor)
#         if current is None:
#             warnings.warn('Early stopping requires %s available!' %
#                           (self.monitor), RuntimeWarning)

#         if self.monitor_op(current, self.best):
#             self.best = current
#             self.wait = 0
#         else:
#             if self.wait >= self.patience:
#                 if self.verbose > 0:
#                     print('Epoch %05d: early stopping' % (epoch))
#                 self.model.stop_training = True
#             self.wait += 1
