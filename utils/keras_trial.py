import json
import hashlib
from keras.models import model_from_json
from keras.engine.training import Model
from CMS_SURF_2016.utils.preprocessing import preprocessFromPandas_label_dir_pairs
from CMS_SURF_2016.utils.callbacks import *
from keras.models import model_from_json
from keras.callbacks import *
import os
import copy

class KerasTrial():
    def __init__(self,
                    trial_dir,
                    name = 'trial',
    				model=None,
                    label_dir_pairs=None,
                    num_samples=None,
                    object_profiles=None,
                    observ_types=None,
                    optimizer=None,
                    loss=None,
                    metrics=[],
                    sample_weight_mode=None,
                    batch_size=32,
                    nb_epoch=10,
                    verbose=1,
                    callbacks=[],
                    validation_split=0.0,
                    validation_data=None,
                    shuffle=True,
                    class_weight=None,
                    sample_weight=None
                ):
    	

        self.trial_dir = trial_dir
        self.name = name
        self.hashcode = None
        self.setModel(model)

        self.setPreprocessing(label_dir_pairs=label_dir_pairs,
                                num_samples=num_samples,
                                object_profiles=object_profiles,
                                observ_types=observ_types)
    	# #Preprocessing
    	# self.label_dir_pairs=label_dir_pairs
     #    self.num_samples=num_samples
     #    self.object_profiles=object_profiles
     #    self.observ_types=observ_types

        self.setCompilation(optimizer=optimizer,
                                loss=loss,
                                metrics=metrics,
                                sample_weight_mode=sample_weight_mode)


        self.setFit(    batch_size=batch_size,
                        nb_epoch=nb_epoch,
                        verbose=verbose,
                        callbacks=callbacks,
                        validation_split=validation_split,
                        validation_data=validation_data,
                        shuffle=shuffle,
                        class_weight=class_weight,
                        sample_weight=sample_weight)
        # #Compilation
        # self.optimizer=optimizer
        # self.loss=loss
        # self.metrics=metrics
        # self.sample_weight_mode=sample_weight_mode

        # #Fit
        # self.batch_size=batch_size
        # self.nb_epoch=nb_epoch
        # self.verbose=verbose
        # self.callbacks=callbacks
        # self.validation_split=validation_split
        # self.validation_data=validation_data
        # self.shuffle=shuffle
        # self.class_weight=class_weight
        # self.sample_weight=sample_weight

    def setModel(self, model):
        #Model
        self.model = model
        if(isinstance(model, Model)):
            self.model = model.to_json()

    def setPreprocessing(self,
    				label_dir_pairs=None,
                    num_samples=None,
                    object_profiles=None,
                    observ_types=None):
    	#Preprocessing
    	self.label_dir_pairs=label_dir_pairs
        self.num_samples=num_samples
        self.object_profiles=object_profiles
        self.observ_types=observ_types

    def setCompilation(self,
    				optimizer,
                    loss,
                    metrics=[],
                    sample_weight_mode=None):

        metrics.sort()
    	#Compilation
        self.optimizer=optimizer
        self.loss=loss
        self.metrics=metrics
        self.sample_weight_mode=sample_weight_mode

    def setFit(self,
                batch_size=32,
                nb_epoch=10,
                verbose=1,
                callbacks=[],
                validation_split=0.0,
                validation_data=None,
                shuffle=True,
                class_weight=None,
                sample_weight=None):
    	#Fit
        strCallbacks = []
        for c in callbacks:
            if(isinstance(c, SmartCheckpoint) == False):
                strCallbacks.append(encodeCallback(c))
        callbacks = strCallbacks

        # print("MOOOO MOO:", callbacks)

        self.batch_size=batch_size
        self.nb_epoch=nb_epoch
        self.verbose=verbose
        self.callbacks=callbacks
        self.validation_split=validation_split
        self.validation_data=validation_data
        self.shuffle=shuffle
        self.class_weight=class_weight
        self.sample_weight=sample_weight

    def to_JSON(self):
        encoder = TrialEncoder()
        return encoder.encode(self)
        # return json.dumps(self, default=lambda o: o.__dict__, 
            # sort_keys=True, indent=4)

    def hash(self, rehash=False):
        if(self.hashcode == None):
            self.hashcode = compute_hash(self.to_JSON())
    	return self.hashcode

    def preprocess(self):
        return preprocessFromPandas_label_dir_pairs(
                label_dir_pairs=self.label_dir_pairs,
                num_samples=self.num_samples,
                object_profiles=self.object_profiles,
                observ_types=self.observ_types)
    def compile(self):
        model = model_from_json(self.model)
        model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=self.metrics,
            sample_weight_mode=self.sample_weight_mode)
        return model

    def fit(self, model, x_train, y_train, index_store=["val_acc"]):
        callbacks = []
        # print(self.callbacks)
        for c in self.callbacks:
            if(c != None):
                callbacks.append(decodeCallback(c))
        monitor = 'val_acc'
        if(self.validation_split == 0.0 or self.validation_data == None):
            monitor = 'acc'
        callbacks.append(SmartCheckpoint('weights', associated_trial=self,
                                             monitor=monitor,
                                             verbose=1,
                                             save_best_only=True,
                                             mode='auto'))
        print callbacks
        model.fit(x_train, y_train,
            batch_size=self.batch_size,
            nb_epoch=self.nb_epoch,
            verbose=self.verbose,
            callbacks=callbacks,
            validation_split=self.validation_split,
            validation_data=self.validation_data,
            shuffle=self.shuffle,
            class_weight=self.class_weight,
            sample_weight=self.sample_weight)

        history_path = self.get_path()+"history.json"
        if(os.path.exists(history_path)):
            histDict = json.load(open( history_path, "rb" ))
            dct = {} 
            for x in index_store:
                dct[x] = max(histDict[x])
            self.to_index(dct)
           
           

    def exectute(self):
    	x_train, y_train = self.preprocess()
        model = self.compile()
        self.fit(model, x_train, y_train)

    def get_path(self):
        json_str = self.to_JSON()
        hashcode = compute_hash(json_str)
        return get_blob_path(hashcode=hashcode, trial_dir=self.trial_dir)

    def to_index(self, dct, append=False, replace=True):
        # if(isinstance(keys, list) == False): keys = [keys]
        # if(isinstance(values, list) == False): values = [values]

        index = read_index(self.trial_dir)
        hashcode = self.hash()
        trial_dict = index.get(hashcode, {})
        for key in dct:
            if(append == True):
                if((key in trial_dict) == True):
                    x = trial_dict[key]
                    if(isinstance(x, list) == False):
                        x = [x]
                    if(replace == True):
                        x = set(x)
                        x.add(dct[key])
                        x = list(x)
                    else:
                        x.append(dct[key])
                    trial_dict[key] = x
                else:
                    trial_dict[key] = dct[key]
            else:
                if(replace == True or (key in trial_dict) == False):
                    trial_dict[key] = dct[key]
        index[hashcode] = trial_dict
        write_index(index, self.trial_dir)  



class TrialEncoder(json.JSONEncoder):
    def __init__(self):
        json.JSONEncoder.__init__(self,sort_keys=True, indent=4)
    def default(self, obj):
        d = obj.__dict__
        d = copy.deepcopy(d)
        if('name' in d): del d['name']
        if('trial_dir' in d): del d['trial_dir']
        if('hashcode' in d): del d['hashcode']
        return d



# class TrialDecoder(json.JSONDecoder):
  
#     def object_hook(self, dct):
#         d = obj.__dict__
#         d = copy.deepcopy(d)
#         if('name' in d): del d['name']
#         if('trial_dir' in d): del d['trial_dir']
#         return d

def encodeCallback(c):
    d = {}
    if(isinstance(c, EarlyStopping)):
        d['monitor'] = c.monitor
        d['patience'] = c.patience
        d['verbose'] = c.verbose
        d['mode'] = 'auto'
        if(isinstance(c, OverfitStopping)):
             d['type'] = "OverfitStopping"
             d['comparison_monitor'] = c.comparison_monitor
             d['max_percent_diff'] = c.max_percent_diff
        else:
             d['type'] = "EarlyStopping"
    return d




def decodeCallback(d):
    # if(d == None):

    if(d['type'] == "OverfitStopping"):
        return OverfitStopping(  monitor=d['monitor'],
                                comparison_monitor=d['comparison_monitor'],
                                max_percent_diff=d['max_percent_diff'],
                                patience=d['patience'],
                                verbose=d['verbose'],
                                mode =d['mode'])
    elif(d['type'] == "EarlyStopping"):
        return EarlyStopping(   monitor=d['monitor'],
                                patience=d['patience'],
                                verbose=d['verbose'],
                                mode =d['mode'])



def compute_hash(inp):
    json_str = inp
    if(isinstance(inp, KerasTrial)):
        json_str = inp.to_JSON()
    h = hashlib.sha1()
    h.update(json_str)
    return h.hexdigest()

def split_hash(hashcode):
    return hashcode[:5], hashcode[5:]

def get_blob_path(*args, **kwargs):
    def _helper(a):
        if(isinstance(a, KerasTrial)):
            trial = a
            return split_hash(trial.hash())
        elif(isinstance(a, str)):
            hashcode = a
            return split_hash(hashcode)
        else:
            raise ValueError("Unknown datatype at 1st argument")
    if(len(args) == 2):
        blob_dir, blob = _helper(args[0])
        trial_dir = args[1]
    elif(len(args) <= 1):
        if('trial_dir' in kwargs):
            trial_dir = kwargs['trial_dir']
        else:
            raise ValueError("Trial Directory was not specified")
        if(len(args) == 1):
            blob_dir, blob = _helper(args[0])
        elif(len(args) == 0):
            if 'json_str' in kwargs:
                hashcode = compute_hash(kwargs['json_str'])
            elif 'hashcode' in kwargs:
                hashcode = kwargs['hashcode']
            else:
                raise ValueError("No hashcode or trial specified")
            blob_dir, blob = split_hash(hashcode)
    else:
        raise ValueError("Too Many arguments")
            
    
    # if(isinstance(args[0], KerasTrial)):
    #     trial = args[0]
    #     blob_dir, blob = split_hash(trial.hash())
    # else:
    #     blob_dir, blob = split_hash(hashcode)
       
    # blob_dir, blob = split_hash(trial.hash())
    blob_path = trial_dir + "blobs/" +  blob_dir + '/' + blob + "/"
    return blob_path

def is_complete(trial, trial_dir):
    blob_path = get_blob_path(trial, trial_dir)
    history_path = blob_path+"history.json"
    if(os.path.exists(history_path)):

        histDict = json.load(open( history_path, "rb" ))
        if(len(histDict.get('stops', [])) > 0):
            return True
        else:
            return False
    else:
        return False
def read_index(trial_dir):
    try:
        index = json.load(open( trial_dir + 'index.json', "rb" ))
        print('Sucessfully loaded index.json at ' + trial_dir)
    except (IOError, EOFError):
        index = {}
        print('Failed to load index.json  at ' + trial_dir)
    return index

def write_index(index,trial_dir):
    try:
        json.dump(index,  open( trial_dir + 'index.json', "wb" ))
        print('Sucessfully wrote index.json at ' + trial_dir)
    except (IOError, EOFError):
        print('Failed to write index.json  at ' + trial_dir)

def write_to_index(key, value):

    index = read_index(trial_dir)
    trial_dict = index.get(hashcode, {})
    trial_dict['name'] = name
    index[hashcode] = trial_dict
    write_index(index, trial_dir)

def write_trial(trial, trial_dir):
    json_str = trial.to_JSON()
    hashcode = compute_hash(json_str)
    blob_path = get_blob_path(hashcode=hashcode, trial_dir=trial_dir)
    if not os.path.exists(blob_path):
        os.makedirs(blob_path)
    trial_path = blob_path + "trial.json"
    try:
        f = open(trial_path, 'w')
        f.write(json_str)
        print('Sucessfully wrote index.json at ' + trial_dir)
    except (IOError, EOFError):
        print('Failed to write index.json  at ' + trial_dir)
    f.close()
    trial.to_index({'name' : trial.name}, append=True)
    # index = read_index(trial_dir)
    # trial_dict = index.get(hashcode, {})
    # trial_dict['name'] = name
    # index[hashcode] = trial_dict
    # write_index(index, trial_dir)




# def batchExecuteTrials(trials):
#     for trial in trials:
#         if(isinstance(trial, KerasTrial) == False):
#             raise ValueError("Cannot execute trail of type %r." % (type(trial))
        
            
        
