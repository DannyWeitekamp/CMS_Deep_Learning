# from __future__ import unicode_literals
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
import h5py
import re
import shutil
from CMS_SURF_2016.layers.lorentz import Lorentz, _lorentz
from CMS_SURF_2016.layers.slice import Slice

class Storable( object ):
    """An object that we can hash, archive as a json String, and reconstitute"""
    def __init__(self):
        '''Initialize the Storable'''
        self.hashcode = None
    def hash(self, rehash=False):
        '''Compute the hashcode for the Storable from its json string'''
        if(self.hashcode is None):
            self.hashcode = compute_hash(self.to_json())
        return self.hashcode
    def get_path(self):
        '''Gets the archive (blob) path from its hash'''
        json_str = self.to_json()
        hashcode = compute_hash(json_str)
        return get_blob_path(hashcode=hashcode, archive_dir=self.archive_dir) 
    def to_json( self ):
        '''Must implement a function that returns the json string corresponding to the Storable'''
        raise NotImplementedError( "Should have implemented to_json" )
    def write( self ):
        '''Must implement a function that write the Storable's json sring to its archive (blob) path'''
        raise NotImplementedError( "Should have implemented write" )
    def remove_from_archive(self):
        '''Removes all the data that the Storable has archived in its archive path'''
        folder = self.get_path()
        blob_dir, blob = split_hash(self.hash()) 
        parentfolder = self.archive_dir + "blobs/" +  blob_dir + '/'
        try:
            if(os.path.isdir(folder)):
                shutil.rmtree(folder)
            if(os.path.isdir(parentfolder) and os.listdir(parentfolder) == []):
                shutil.rmtree(parentfolder)
        except Exception as e:
            print(e)

    @staticmethod
    def find_by_hashcode( hashcode, archive_dir ):
        '''Must implement function that find a Storable by its hashcode'''
        raise NotImplementedError( "Should have implemented find_by_hashcode" )

class DataProcedure(Storable):
    '''A wrapper for archiving the results of data grabbing and preprocessing functions of the type X,Y getXY where are X is the training
        data and Y contains the labels/targets for each entry'''
    def __init__(self, archive_dir, func,  *args, **kargs):
        Storable.__init__(self)
        self.archive_dir = archive_dir
        self.func = func.__name__
        self.func_module = func.__module__
        self.args = args
        self.kargs = kargs
        self.encoder = json.JSONEncoder(sort_keys=True, indent=4, default=lambda x: x.__dict__)
        self.X = None
        self.Y = None

    def set_encoder(self, encoder):
        '''Set the json encoder for the procedure in case its arguements are not json encodable'''
        self.encoder = encoder


    def to_json(self):
        '''Returns the json string for the Procedure with only its essential characteristics'''
        d = self.__dict__
        d = copy.deepcopy(d)
        #Don't hash on verbose or verbosity if they are in the function
        if("verbose" in d.get("kargs", [])): del d['kargs']["verbose"]
        if("verbosity" in d.get("kargs", [])): del d['kargs']["verbosity"]

        del d["archive_dir"]
        if('encoder' in d): del d["encoder"]
        if('decoder' in d): del d["decoder"]
        del d["hashcode"]
        del d["X"]
        del d["Y"]
        return self.encoder.encode(d)

    # def hash(self, rehash=False):
    #     if(self.hashcode == None):
    #         self.hashcode = compute_hash(self.to_json())
    #     return self.hashcode

    def write(self, verbose=0):
        '''Write the json string for the procedure to its directory'''
        json_str = self.to_json()
        hashcode = compute_hash(json_str)
        blob_path = self.get_path()
        write_object(blob_path, 'procedure.json', json_str, verbose=verbose)

    def is_archived(self):
        '''Returns True if this procedure is already archived'''
        blob_path = get_blob_path(self, self.archive_dir)
        data_path = blob_path+"archive.h5"
        if(os.path.exists(data_path)):
            return True
        else:
            return False

    def archive(self):
        '''Store the DataProcedure in a directory computed by its hashcode'''
        if((self.X is None or self.Y is None) == False):
            blob_path = self.get_path()
            if( os.path.exists(blob_path) == False):
                os.makedirs(blob_path)
            if( os.path.exists(blob_path + 'procedure.json') == False):
                self.write()
            X = self.X
            Y = self.Y

            if(isinstance(self.X, list) == False): X = [X]
            if(isinstance(self.Y, list) == False): Y = [Y]
            h5f = h5py.File(self.get_path() + 'archive.h5', 'w')
            h5f.create_group("X")
            for i, x in enumerate(X):
                h5f.create_dataset('X/'+str(i), data=x)
            h5f.create_group("Y")
            for i, y in enumerate(Y):
                h5f.create_dataset('Y/'+str(i), data=y)
            
            h5f.close()
            data_archive = read_dataArchive(self.archive_dir)

            #TODO: this is a really backward way of doing this
            jstr = self.to_json()
            d = json.loads(jstr)

            # print(d)
            proc_dict = {}
            proc_dict['func'] = d['func']
            proc_dict['module'] = d['func_module']
            proc_dict['args'] = d['args']
            proc_dict['kargs'] = d['kargs']
            data_archive[self.hash()] = proc_dict

            write_dataArchive(data_archive, self.archive_dir)
            # def read_json_obj(directory, filename, verbose=0):
                
        else:
            raise ValueError("Cannot archive DataProcedure with NoneType X or Y")
        # self.to_index({'name' : self.name}, append=True)

    def remove_from_archive(self):
        '''Removes the DataProcedure from the data_archive and destroys its blob directory'''
        data_archive = read_dataArchive(self.archive_dir)
        if(self.hash() in  data_archive): del data_archive[self.hash()] 
        write_dataArchive(data_archive, self.archive_dir)

        Storable.remove_from_archive(self)

    def get_XY(self, archive=True, redo=False, verbose=1):
        '''Apply the DataProcedure returning X,Y from the archive or generating them from func'''
        if(self.is_archived() and redo == False):
            h5f = None
            try:
                h5f = h5py.File(self.get_path() + 'archive.h5', 'r')
                self.X = []
                X_group = h5f['X']
                keys = list(X_group.keys())
                keys.sort()
                for key in keys:
                    self.X.append(X_group[key][:])

                self.Y = []
                Y_group = h5f['Y']
                keys = list(Y_group.keys())
                keys.sort()
                for key in keys:
                    self.Y.append(Y_group[key][:])

                h5f.close()
                if(verbose >= 1): print("DataProcedure results %r read from archive" % self.hash())
            except:
                if(h5f != None): h5f.close()
                if(verbose >= 1): print("Failed to load archive %r running from scratch" % self.hash())
                return self.get_XY(archive=archive, redo=True)
        else:
            prep_func = self.get_func(self.func, self.func_module)
            self.X, self.Y = prep_func(*self.args, **self.kargs)
            # print("WRITE:", self.X.shape, self.Y.shape)
            if(archive == True): self.archive()
        return self.X, self.Y

    def get_summary(self):
        '''Get the summary for the DataProcedure as a string'''
        str_args = ','.join([str(x) for x in self.args])
        str_kargs = ','.join([str(x) + "=" + str(self.kargs[x]) for x in self.kargs])
        arguments = ','.join([str_args, str_kargs])
        return self.func_module + "." + self.func +"(" + arguments + ")"
    def summary(self):
        '''Print a summary'''
        print("-"*50)
        print("DataProcedure (%r)" % self.hash())
        print("    " + self.get_summary())
        print("-"*50) 

    @staticmethod
    def get_func(name, module):
        '''Get a function from its name and module path'''
        # print("from " + module +  " import " + name + " as prep_func")
        try:
            exec("from " + module +  " import " + name + " as prep_func")
        except ImportError:
            # try:
            #     exec('prep_func = ' + name)
            # except Exception:
            raise ValueError("DataProcedure function %r does not exist in %r. \
                For best results functions should be importable and not locally defined." % (str(name), str(module)))
        return prep_func

    @staticmethod
    def from_json(archive_dir ,json_str, arg_decode_func=None):  
        '''Get a DataProcedure object from its json string'''
        d = json.loads(json_str)
        func = None
        temp = lambda x: 0
        try:
            func = DataProcedure.get_func(d['func'], d['func_module'])
        except ValueError:
            func = temp
        args, kargs = d['args'], d['kargs']
        if(arg_decode_func != None):
            # print('arg_decode_func_ENABLED:', arg_decode_func.__name__)
            args, kargs = arg_decode_func(*args, **kargs)
        dp = DataProcedure(archive_dir,  func, *args, **kargs)
        if(func == temp):
            dp.func = d['func']
            dp.func_module = d['func_module']
        return dp

    @staticmethod
    def find_by_hashcode( hashcode, archive_dir, verbose=0 ):
        '''Returns the archived DataProcedure with the given hashcode or None if one is not found'''
        path = get_blob_path(hashcode, archive_dir) + 'procedure.json'
        try:
            f = open( path, "rb" )
            json_str = f.read()
            f.close()
            # print(json_str)
            out = DataProcedure.from_json(archive_dir,json_str)
            if(verbose >= 1): print('Sucessfully loaded procedure.json at ' + archive_dir)
        except (IOError, EOFError):
            out = None
            if(verbose >= 1): print('Failed to load procedure.json  at ' + archive_dir)
        return out




class KerasTrial(Storable):
    '''An archivable object representing a machine learning trial in keras'''
    def __init__(self,
                    archive_dir,
                    name = 'trial',
    				model=None,
                    train_procedure=None,
                    optimizer=None,
                    loss=None,
                    metrics=[],
                    sample_weight_mode=None,
                    batch_size=32,
                    nb_epoch=10,
                    callbacks=[],
                    validation_split=0.0,
                    validation_data=None,
                    shuffle=True,
                    class_weight=None,
                    sample_weight=None
                ):
    	
        Storable.__init__(self)
        if(archive_dir[len(archive_dir)-1] != "/"):
            archive_dir = archive_dir + "/"
        self.archive_dir = archive_dir
        self.name = name
        self.setModel(model)

        self.setTrain(train_procedure=train_procedure)

        self.setCompilation(optimizer=optimizer,
                                loss=loss,
                                metrics=metrics,
                                sample_weight_mode=sample_weight_mode)


        self.setFit(    batch_size=batch_size,
                        nb_epoch=nb_epoch,
                        callbacks=callbacks,
                        validation_split=validation_split,
                        validation_data=validation_data,
                        shuffle=shuffle,
                        class_weight=class_weight,
                        sample_weight=sample_weight)
       

    def setModel(self, model):
        '''Set the model used by the trial (either the object or derived json string)'''
        self.model = model
        self.compiled_model = None
        if(isinstance(model, Model)):
            self.model = model.to_json()


    def setTrain(self,
                   train_procedure=None):
        '''Sets the training data for the trial'''
        if(train_procedure != None):
            if(isinstance(train_procedure, list) == False):
                train_procedure = [train_procedure]
            l = []
            for p in train_procedure:
                if(isinstance(p, DataProcedure)):
                    l.append(p.to_json())
                else:
                    l.append(p)
            train_procedure = l    
        else:
            train_procedure = None
        self.train_procedure = train_procedure

    def setCompilation(self,
    				optimizer,
                    loss,
                    metrics=[],
                    sample_weight_mode=None):
        '''Sets the compilation arguments for the trial'''
        metrics.sort()
        self.optimizer=optimizer
        self.loss=loss
        self.metrics=metrics
        self.sample_weight_mode=sample_weight_mode

    def setFit(self,
                batch_size=32,
                nb_epoch=10,
                callbacks=[],
                validation_split=0.0,
                validation_data=None,
                shuffle=True,
                class_weight=None,
                sample_weight=None):
        '''Sets the fit arguments for the trial'''
    	#Fit
        strCallbacks = []
        for c in callbacks:
            if(isinstance(c, SmartCheckpoint) == False):
                if(isinstance(c, Callback) == True):
                    strCallbacks.append(encodeCallback(c))
                else:
                    strCallbacks.append(c)
        callbacks = strCallbacks

        self.batch_size=batch_size
        self.nb_epoch=nb_epoch
        # self.verbose=verbose
        self.callbacks=callbacks
        self.validation_split=validation_split
        self.validation_data=validation_data
        self.shuffle=shuffle
        self.class_weight=class_weight
        self.sample_weight=sample_weight

    def to_json(self):
        '''Converts the trial to a json string '''
        encoder = TrialEncoder()
        return encoder.encode(self)
    

    # def preprocess(self):
    #     return preprocessFromPandas_label_dir_pairs(
    #             label_dir_pairs=self.label_dir_pairs,
    #             num_samples=self.num_samples,
    #             object_profiles=self.object_profiles,
    #             observ_types=self.observ_types)
    def compile(self, loadweights=False, redo=False, custom_objects={}):
        '''Compiles the model set for this trial'''
        if(self.compiled_model is None or redo): 
            model = self.get_model(loadweights=loadweights, custom_objects=custom_objects)#model_from_json(self.model)
            model.compile(
                optimizer=self.optimizer,
                loss=self.loss,
                metrics=self.metrics,
                sample_weight_mode=self.sample_weight_mode)
            self.compiled_model = model
        else:
            model = self.compiled_model
        return model

    def fit(self, model, x_train, y_train, index_store=["val_acc"], verbose=1):
        '''Runs model.fit(x_train, y_train) for the trial using the arguments passed to trial.setFit(...)'''
        callbacks = []
        # print(self.callbacks)
        for c in self.callbacks:
            if(c != None):
                callbacks.append(decodeCallback(c))
        monitor = 'val_acc'
        if(self.validation_split == 0.0 or self.validation_data is None):
            monitor = 'acc'
        callbacks.append(SmartCheckpoint('weights', associated_trial=self,
                                             monitor=monitor,
                                             verbose=verbose,
                                             save_best_only=True,
                                             mode='auto'))
        model.fit(x_train, y_train,
            batch_size=self.batch_size,
            nb_epoch=self.nb_epoch,
            # verbose=self.verbose,
            callbacks=callbacks,
            validation_split=self.validation_split,
            validation_data=self.validation_data,
            shuffle=self.shuffle,
            class_weight=self.class_weight,
            sample_weight=self.sample_weight)

        histDict = self.get_history()
        if(histDict != None):
            dct = {} 
            for x in index_store:
                dct[x] = max(histDict[x])
            self.to_index(dct)

    def write(self, verbose=0):
        '''Writes the model's json string to its archive location''' 
        json_str = self.to_json()
        hashcode = compute_hash(json_str)
        blob_path = self.get_path()
        write_object(blob_path, 'trial.json', json_str, verbose=verbose)

        self.to_index({'name' : self.name}, append=True)
                 

    def execute(self, archiveTraining=True, arg_decode_func=None, custom_objects={}):
        '''Executes the trial, fitting on the X, and Y for training for each given DataProcedure in series'''
    	if(self.train_procedure is None):
            raise ValueError("Cannot execute trial without DataProcedure")
        if(self.is_complete() == False):
            model = self.compile(custom_objects=custom_objects)
            train_procs = self.train_procedure
            if(isinstance(train_procs, list) == False): train_procs = [train_procs]
            # print(train_procs)
            totalN = 0
            for p in train_procs:
                proc = DataProcedure.from_json(self.archive_dir,p, arg_decode_func=arg_decode_func)
                X, Y = proc.get_XY(archive=archiveTraining)
                if(isinstance(X, list) == False): X = [X]
                if(isinstance(Y, list) == False): Y = [Y]
                totalN += Y[0].shape[0]
                self.fit(model,X, Y)
            self.write()

            # if(self.validation_split != 0.0):
            dct =  {'num_train' : totalN*(1.0-self.validation_split),
                    'num_validation' : totalN*(self.validation_split),
                    'elapse_time' : self.get_history()['elapse_time'],
                    'fit_cycles' : len(train_procs)
                    }
            self.to_index( dct, replace=True)
        else:
            print("Trial %r Already Complete" % self.hash())
    def test(self,test_proc, archiveTraining=True, custom_objects={}):
        model = self.compile(custom_objects=custom_objects)
        if(isinstance(test_proc, DataProcedure) == False):
            proc = DataProcedure.from_json(self.archive_dir,test_proc, arg_decode_func=arg_decode_func)
        else:
            proc = test_proc
        X, Y = proc.get_XY(archive=archiveTraining)
        if(isinstance(X, list) == False): X = [X]
        if(isinstance(Y, list) == False): Y = [Y]
        metrics = model.evaluate(X, Y)
        self.to_index({'test_loss' : metrics[0], 'test_acc' :  metrics[1], 'num_test' : Y[0].shape[0]}, replace=True)
        return metrics

    def to_index(self, dct, append=False, replace=True):
        '''Pushes a dictionary of values to the archive index ('index' like in a book not a list) for this trial'''
        index = read_index(self.archive_dir)
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
        write_index(index, self.archive_dir) 

    def get_index_entry(self, verbose=0):
        '''Get the dictionary containing all the index values for this trial  ('index' like in a book not a list)'''
        index = read_index(self.archive_dir, verbose=verbose)
        return index[self.hash()]

    def get_from_index(self, keys, verbose=0):
        '''Get a value from the index  ('index' like in a book not a list)'''
        indexDict = self.get_index_entry(verbose=verbose)
        if(isinstance(keys, list)):
            out = []
            for key in keys:
                out.append(indexDict.get(key, None))
        else:
            out = indexDict.get(keys, None)
        return out

    def get_history(self, verbose=0):
        '''Get the training history for this trial'''
        # history_path = self.get_path()+"history.json"
        history = read_json_obj(self.get_path(), "history.json")
        if(history == {}):
            history = None
        return history

    def get_model(self, loadweights=False,custom_objects={}):
        '''Gets the model, optionally with the best set of weights'''
        model = model_from_json(self.model, custom_objects=custom_objects)
        if(loadweights): model.load_weights(self.get_path()+"weights.h5")
        return model

    def is_complete(self):
        '''Return True if the trial has completed'''
        blob_path = get_blob_path(self, self.archive_dir)
        history_path = blob_path+"history.json"
        if(os.path.exists(history_path)):

            histDict = json.load(open( history_path, "rb" ))
            if(len(histDict.get('stops', [])) > 0):
                return True
            else:
                return False
        else:
            return False


    def summary(self,
                showName=False,
                showDirectory=False,
                showIndex=True,
                showTraining=True,
                showCompilation=True,
                showFit=True,
                showModelPic=False,
                showNoneType=False,
                squat=True):
        '''Print a summary of the trial
            #Arguments:
                showName=False,showDirectory=False, showIndex=True, showTraining=True, showCompilation=True, showFit=True,
                 showModelPic=False, showNoneType=False -- Control what data is printed
                squat=True -- If False shows data on separate lines
        '''
        indent = "    "     
        d = self.__dict__
        def _listIfNotNone(keys):
            l = []
            for key in keys:
                if(showNoneType == False):
                    val = d.get(key, None)
                    if(val != None):
                        # print(indent*2 + )
                        l.append(str(key) + "=" + str(val))
            return l
        if(squat):
            sep = ", "         
        else:
            sep = "\n" + indent*2 

        print("-"*50)
        print("TRIAL SUMMARY (" + self.hash() + ")" )
        if(showDirectory):print(indent + "Directory: " + self.archive_dir)
        if(showName):  print(indent + "Name: " + self.name)
            # n = self.get_from_index(['name'])
           

        if(showIndex):
            print(indent + "Index_Info:")
            index = self.get_index_entry()
            indexes = []
            for key in index:
                indexes.append(str(key) + " = " + str(index[key]))
            print(indent*2 + sep.join(indexes))

        if(showTraining):
            print(indent + "Training:")
            preps = []
            for s in self.train_procedure:
                p = DataProcedure.from_json(self.archive_dir, s)
                # str_args = ','.join([str(x) for x in p.args])
                # str_kargs = ','.join([str(x) + "=" + str(p.kargs[x]) for x in p.kargs])
                # arguments = ','.join([str_args, str_kargs])
                # preps.append(p.func_module + "." + p.func +"(" + arguments + ")")
                preps.append(p.get_summary())
            print(indent*2 + sep.join(preps))

        if(showCompilation):
            print(indent + "Compilation:")
            comps = _listIfNotNone(["optimizer", "loss", "metrics", "sample_weight_mode"])
            print(indent*2 + sep.join(comps))

        if(showFit):
            print(indent + "Fit:")
            fits = _listIfNotNone(["batch_size", "nb_epoch", "verbose", "callbacks",
                                     "validation_split", "validation_data", "shuffle",
                                     "class_weight", "sample_weight"])
            print(indent*2 + sep.join(fits))

        # if(showModelPic):

        print("-"*50)

    def remove_from_archive(self):
        '''Remove the trial from the index and destroys its archive including the trial.json, weights.h5 and history.json'''
        index = read_index(self.archive_dir)
        if(self.hash() in  index): del index[self.hash()] 
        write_index(index, self.archive_dir)

        Storable.remove_from_archive(self)


    @staticmethod
    def from_json(archive_dir,json_str, name='trial'):
        '''Reconsitute a KerasTrial object from its json string'''
        d = json.loads(json_str)
        # print(d['callbacks'])
        trial = KerasTrial(
                archive_dir,
                name = name,
                model = d.get('model', None),
                train_procedure=d.get('train_procedure', None),
                optimizer=d.get('optimizer', None),
                loss=d.get('loss', None),
                metrics=d.get('metrics', []),
                sample_weight_mode=d.get('sample_weight_mode', None),
                batch_size=d.get('batch_size', 32),
                nb_epoch=d.get('nb_epoch', 10),
                callbacks=d.get('callbacks', []),
                validation_split=d.get('validation_split', 0.0),
                validation_data=d.get('validation_data', None),
                shuffle=d.get('shuffle', True),
                class_weight=d.get('class_weight', None),
                sample_weight=d.get('sample_weight', None))
        return trial

    @staticmethod
    def find_by_hashcode( hashcode, archive_dir, verbose=0 ):
        '''Returns the archived KerasTrial with the given hashcode or None if one is not found'''
        path = get_blob_path(hashcode, archive_dir) + 'trial.json'
        try:
            f = open( path, "rb" )
            json_str = f.read()
            f.close()
            # print(json_str)
            out = KerasTrial.from_json(archive_dir,json_str)
            if(verbose >= 1): print('Sucessfully loaded trial.json at ' + archive_dir)
        except (IOError, EOFError):
            out = None
            if(verbose >= 1): print('Failed to load trial.json  at ' + archive_dir)
        return out

class TrialEncoder(json.JSONEncoder):
    '''A json encoder for KerasTrials. Doesn't store name,archive_dir,hashcode etc since they don't affect how it functions'''
    def __init__(self):
        json.JSONEncoder.__init__(self,sort_keys=True, indent=4)
    def default(self, obj):
        temp = obj.compiled_model
        obj.compiled_model = None
        d = obj.__dict__
        d = copy.deepcopy(d)
        if('name' in d): del d['name']
        if('archive_dir' in d): del d['archive_dir']
        if('hashcode' in d): del d['hashcode']
        if('compiled_model' in d): del d['compiled_model']
        obj.compiled_model = temp
        return d


#TODO: Stopping Callbacks can't infer mode -> only auto works
def encodeCallback(c):
    '''Encodes callbacks so that they can be decoded later'''
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
    '''Decodes callbacks into usable objects'''
    # if(d == None):
    # print(d)
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
    '''Computes a SHA1 hash string from a json string or Storable'''
    json_str = inp
    if(isinstance(inp, Storable)):
        json_str = inp.to_json()
    h = hashlib.sha1()
    h.update(json_str)
    return h.hexdigest()

def split_hash(hashcode):
    '''Splits a SHA1 hash string into two strings. One with the first 5 characters and another with the rest'''
    return hashcode[:5], hashcode[5:]

def get_blob_path(*args, **kwargs):
    '''Blob path (archive location) from either (storable,archive_dir), (hashcode, archive_dir), or
        (json_str=?, archive_dir=?)'''
    def _helper(a):
        if(isinstance(a, Storable)):
            return split_hash(a.hash())
        elif(isinstance(a, str) or isinstance(a, unicode)):
            hashcode = a
            return split_hash(hashcode)
        else:
            raise ValueError("Unknown datatype at 1st argument")
    if(len(args) == 2):
        blob_dir, blob = _helper(args[0])
        archive_dir = args[1]
    elif(len(args) <= 1):
        if('archive_dir' in kwargs):
            archive_dir = kwargs['archive_dir']
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

    blob_path = archive_dir + "blobs/" +  blob_dir + '/' + blob + "/"
    return blob_path


def read_dataArchive(archive_dir, verbose=0):
    '''Returns the data archive read from the trial directory'''
    return read_json_obj(archive_dir, 'data_archive.json')
def write_dataArchive(data_archive, archive_dir, verbose=0):
    '''Writes the data archive to the trial directory'''
    write_json_obj(data_archive, archive_dir, 'data_archive.json')

def read_index(archive_dir, verbose=0):
    '''Returns the index read from the trial directory'''
    return read_json_obj(archive_dir, 'index.json')
#     try:
#         index = json.load(open( archive_dir + 'index.json', "rb" ))
#         if(verbose >= 1): print('Sucessfully loaded index.json at ' + archive_dir)
#     except (IOError, EOFError):
#         index = {}
#         if(verbose >= 1): print('Failed to load index.json  at ' + archive_dir)
#     return index

def write_index(index,archive_dir, verbose=0):
    '''Writes the index to the trial directory'''
    write_json_obj(index, archive_dir, 'index.json')
#     try:
#         json.dump(index,  open( archive_dir + 'index.json', "wb" ))
#         if(verbose >= 1): print('Sucessfully wrote index.json at ' + archive_dir)
#     except (IOError, EOFError):
#         if(verbose >= 1): print('Failed to write index.json  at ' + archive_dir)


def read_json_obj(directory, filename, verbose=0):
    '''Return a json object read from the given directory'''
    try:
        obj = json.load(open( directory + filename, "rb" ))
        if(verbose >= 1): print('Sucessfully loaded ' + filename +'  at ' + directory)
    except (IOError, EOFError):
        obj = {}
        if(verbose >= 1): print('Failed to load '+ filename +'  at ' + directory)
    return obj

def write_json_obj(obj,directory, filename, verbose=0):
    '''Writes a json object to the given directory'''
    try:
        json.dump(obj,  open( directory + filename, "wb" ))
        if(verbose >= 1): print('Sucessfully wrote ' + filename +'  at ' + directory)
    except (IOError, EOFError):
        if(verbose >= 1): print('Failed to write '+ filename +'  at ' + directory)


# def write_to_index(key, value):
#     '''Writes a value to the index'''

#     index = read_index(archive_dir)
#     trial_dict = index.get(hashcode, {})
#     trial_dict['name'] = name
#     index[hashcode] = trial_dict
#     write_index(index, archive_dir)


def write_object(directory, filename, data, verbose=0):
    '''Writes an object from the given data with the given filename in the given directory'''
    if not os.path.exists(directory):
        os.makedirs(directory)
    path = directory + filename
    try:
        f = open(path, 'w')
        f.write(data)
        if(verbose >= 1): print('Sucessfully wrote %r at %r' + (filename, directory))
    except (IOError, EOFError):
        if(verbose >= 1): print('Failed to write %r at %r' + (filename, directory))
    f.close()





#Reading Trials

def get_all_data(archive_dir):
    '''Gets all the DataProcedure in the data_archive'''
    return get_data_by_function('.', archive_dir)

def get_data_by_function(func, archive_dir):
    '''Gets a list of DataProcedure that use a certain function'''
    data_archive = read_dataArchive(archive_dir)
    out = []
    if(isinstance(func, str)):
        func_name = func
        func_module = None
    else:
        func_name = func.__name__
        func_module = func.__module__

    # print(func_name, func_module)
    # print(len(data_archive))
    for key in data_archive:
        t_func = data_archive[key].get("func", 'unknown')
        t_module = data_archive[key].get("func_module", 'unknown')
        # print(t_func, t_module)
        # print(data_archive)
        # print(t_name, name.decode("UTF-8"))
        # print([re.match(name, x) for x in t_name])
        if(re.match(func_name, t_func) != None and (func_module is None or re.match(func_module, t_module) != None)):
            # blob_path = get_blob_path(key, archive_dir)
            dp = DataProcedure.find_by_hashcode(key, archive_dir)
            if(dp != None):
                out.append(dp)

    return out


def get_all_trials(archive_dir):
    '''Get all the trials listed in the index'''
    return get_trials_by_name('.', archive_dir)

def get_trials_by_name(name, archive_dir):
    '''Get all the trials with a particluar name or that match a given regular expression'''
    index = read_index(archive_dir)
    out = []
    for key in index:
        t_name = index[key].get("name", 'unknown')
        # print(t_name, name.decode("UTF-8"))
        if(isinstance(t_name, list) == False):
            t_name = [t_name]
        # print([re.match(name, x) for x in t_name])
        if True in [re.match(name, x) != None for x in t_name]:
            # blob_path = get_blob_path(key, archive_dir)
            dp = KerasTrial.find_by_hashcode(key, archive_dir)
            if(dp != None):
                out.append(dp)

    return out

