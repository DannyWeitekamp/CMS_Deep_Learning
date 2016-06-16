import os
import sys
import json
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
        self.histobj = History()
        ModelCheckpoint.__init__(self, self.checkpointFilename,
                monitor, verbose, save_best_only, mode)

        # self.model.load_weights(self.checkpointFilename);

    # def on_epoch_begin(self, epoch, logs={}):

    def on_train_begin(self, logs={}):
        if not os.path.exists(self.smartDir):
            os.makedirs(self.smartDir)
        histDict = {}
        
        # print('Load history?');
        try:
            histDict = json.load(open( self.historyFilename, "rb" ))
            print('Sucessfully loaded history at ' + self.historyFilename)
        except (IOError, EOFError):
            print('Failed to load history at ' + self.historyFilename)

        self.histobj.history = histDict

        # print(histDict)
            # history = History()
        # history.history = histDict
        self.epochOffset = histDict.get("last_epoch", 0);
        self.__checkMaxEpoch(self.max_epoch + self.epochOffset)
        # print('Load weights?');
        # weightsloaded = False
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