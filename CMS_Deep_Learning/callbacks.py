''' 
callbacks.py
Contains custom callback objects.
Author: Danny Weitekamp
e-mail: dannyweitekamp@gmail.com
''' 


import os
import sys
import json
import time
from keras.callbacks import History
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
class SmartCheckpoint(ModelCheckpoint):
    ''' A smart checkpoint callback that automatically saves and loads training history, and weights
        based on the name given at instantiation. Creates a SmartCheckpoint directory that stores
        the data''' 

    # def __checkMaxEpoch(self, epoch):
    #     if(epoch > self.max_epoch):
    #         self.model.stop_training = True

    def __init__(self, name, directory='', associated_trial=None, monitor='val_loss', verbose=0,
                 save_best_only=True, mode='auto'):
        self.name = name
        if(associated_trial != None):
            
            self.smartDir = associated_trial.get_path()
            self.checkpointFilename = self.smartDir + "weights.h5"
            self.historyFilename = self.smartDir + "history.json"
        else:
            self.smartDir = directory + 'SmartCheckpoint/'
            self.checkpointFilename = self.smartDir + name + "_weights.h5"
            self.historyFilename = self.smartDir + name + "_history.json"
        self.startTime = 0  
        # self.max_epoch = max_epoch
        self.histobj = History()

        histDict = {}
        try:
            histDict = json.load(open( self.historyFilename, "rb" ))
            print('Sucessfully loaded history at ' + self.historyFilename)
        except (IOError, EOFError):
            print('Failed to load history at ' + self.historyFilename)

        self.histobj.history = histDict

        self.elapse_time = histDict.get("elapse_time", 0)

        ModelCheckpoint.__init__(self, self.checkpointFilename,
                monitor, verbose, save_best_only, mode)

        metric_history = histDict.get(monitor, None)
        if(metric_history != None):
            best = metric_history[0]
            for metric in metric_history:
                if self.monitor_op(metric, self.best):
                    self.best = metric

    

    def on_train_begin(self, logs={}):
        self.startTime = time.clock()
        histDict = self.histobj.history

        if not os.path.exists(self.smartDir):
            os.makedirs(self.smartDir)
        
        self.epochOffset = histDict.get("last_epoch", 0);
        # self.__checkMaxEpoch(self.max_epoch + self.epochOffset)
        try:
            self.model.load_weights(self.checkpointFilename)
            print('Sucessfully loaded weights at ' + self.checkpointFilename)
        except (IOError, EOFError):
            print('Failed to load weights at ' + self.checkpointFilename)


    def on_epoch_end(self, epoch, logs={}):
        histDict = self.histobj.history

        epoch = epoch + self.epochOffset + 1
        ModelCheckpoint.on_epoch_end(self, epoch, logs)
        histDict["last_epoch"] = epoch
        for k, v in logs.items():
            if k not in histDict:
                histDict[k] = []
            histDict[k].append(v)
        json.dump(histDict,  open( self.historyFilename, "wb" ))

    def on_train_end(self, logs={}):
        histDict = self.histobj.history
        elapse = histDict.get("elapse_time", 0)

        #Elapse Time
        histDict["elapse_time"] = elapse + time.clock() - self.startTime 

        #Stops
        if(self.model.stop_training == True):
            stop = "callback"
        else:
            stop = "finished"
        stops = histDict.get("stops", [])
        stops.append( (stop,  histDict.get("last_epoch", 0)) )
        histDict['stops'] = stops

        json.dump(histDict,  open( self.historyFilename, "wb" ))
        # print("DONE!")
        # print(logs)
        # print(histDict)
        # self.__checkMaxEpoch(epoch + self.epochOffset)


class OverfitStopping(EarlyStopping):
    '''Stop training when a monitored quantity has stopped improving.
    # Arguments
        monitor: quantity to be monitored.
        patience: number of epochs with no improvement
            after which training will be stopped.
        verbose: verbosity mode.
        mode: one of {auto, min, max}. In 'min' mode,
            training will stop when the quantity
            monitored has stopped decreasing; in 'max'
            mode it will stop when the quantity
            monitored has stopped increasing.
    '''
    def __init__(self, monitor='val_loss', comparison_monitor="loss", max_percent_diff=.1, patience=0, verbose=0, mode='auto'):
        self.comparison_monitor = comparison_monitor
        self.max_percent_diff = max_percent_diff
        EarlyStopping.__init__(self,monitor, patience=patience, verbose=verbose, mode=mode)

    def on_train_begin(self, logs={}):
        self.wait = 0       # Allow instances to be re-used
       # self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        current_comp = logs.get(self.comparison_monitor)
        signed_percent_difference = (current - current_comp)/((current + current_comp)/2)

        if current is None:
            warnings.warn('Overfit stopping requires %s available!' %
                          (self.monitor), RuntimeWarning)
        if current_comp is None:
            warnings.warn('Overfit stopping requires %s available!' %
                          (self.comparison_monitor), RuntimeWarning)

        if self.monitor_op(signed_percent_difference, self.max_percent_diff):
            self.best = current
            self.wait = 0
        else:
            if self.wait >= self.patience:
                if self.verbose > 0:
                    print('Epoch %05d: overfit stopping' % (epoch))
                self.model.stop_training = True
            self.wait += 1
