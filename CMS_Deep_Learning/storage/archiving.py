''' 
archive.py
A system for archiving keras trials and input data
Author: Danny Weitekamp
e-mail: dannyweitekamp@gmail.com
'''

import copy
import glob
import hashlib
import re
import shutil
import types

import h5py
from keras.callbacks import *
from keras.engine.training import Model
from keras.models import model_from_json

from CMS_Deep_Learning.callbacks import *


class Storable( object ):
    """An object that we can hash, archive as a json String, and reconstitute"""
    def __init__(self):
        '''Initialize the Storable'''
        def recurseStore(x):
            if(isinstance(x,Storable)):
                return x.to_json()
            else:
                return x.__dict__ 
                
        self.encoder = json.JSONEncoder(sort_keys=True, indent=4, default=recurseStore)
        self.hashcode = None
    def hash(self, rehash=False):
        '''Compute the hashcode for the Storable from its json string'''
        if(self.hashcode is None):
            self.hashcode = compute_hash(self.to_hashable())
        return self.hashcode
    def get_path(self):
        '''Gets the archive (blob) path from its hash'''
        hashcode = self.hash()
        return get_blob_path(hashcode=hashcode, archive_dir=self.archive_dir) 
    def to_json( self ):
        '''Must implement a function that returns the json string corresponding to the Storable'''
        raise NotImplementedError( "Should have implemented to_json" )
    def to_hashable( self ):
        '''Must implement a function that returns a hashable string corresponding to the Storable'''
        raise NotImplementedError( "Should have implemented to_hashable" )
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


    def read_record(self, verbose=0):
        '''Get the dictionary containing all the record values for this trial '''
        record = read_json_obj(self.get_path(), "record.json",verbose=verbose)
        if(record == None): record = {}
        return record

    def write_record(self, obj, verbose=0):
        '''Write the dictionary containing all the record values for this trial '''
        if(not isinstance(obj, dict)):
            raise TypeError("obj must be type dict, but got %r" % type(obj))
        write_json_obj(obj, self.get_path(),'record.json',verbose=verbose)

    def remove_from_record(self, key, verbose=0):
        '''Remove a key from the record. Returns 1 if sucessfully removed 0 if does not exist'''
        record = self.read_record(verbose)
        if(key in record):
            del record[key]
            self.write_record(record)
            return 1
        return 0


    @staticmethod
    def get_all_paths(archive_dir):
        '''Get a list of all the blob paths of the Storables in the given archive_dir'''
        if(archive_dir[-1] != "/"):
            archive_dir += "/"
        directories = glob.glob(archive_dir + "blobs/*")
        paths = []
        for d in directories:
            p = glob.glob(d + "/*")
            paths += [path +"/" for path in p]
        return paths

    @classmethod
    def get_all_records(cls,archive_dir,verbose=0):
        '''Get a dicionary of all the records in the archive_dir keyed by their hashcodes'''
        paths = cls.get_all_paths(archive_dir)
        records = {}
        for path in paths:
            s = path.split('/')
            hashcode = s[-3] + s[-2]
            assert len(hashcode) > len(s[-3]) and len(hashcode) > len(s[-2])
            record = read_json_obj(path, "record.json",verbose=verbose)
            if(record != {}):
                records[hashcode] = record
        return records

    @classmethod
    def find_by_hashcode(cls, archive_dir,hashcode, verbose=0 ):
        '''Returns the archived DataProcedure with the given hashcode or None if one is not found'''
        name = cls.__name__
        if(issubclass(cls,KerasTrial)):
            name = "trial"
        elif(issubclass(cls,DataProcedure)):
            name  = "procedure"
        path = get_blob_path(hashcode, archive_dir) + name +'.json'
        try:
            f = open( path, "rb" )
            json_str = f.read()
            f.close()
            out = cls.from_json(archive_dir,json_str)
            out.hashcode = hashcode
            if(verbose >= 1): print('Sucessfully loaded ' + name + '.json at ' + path)
        except (IOError, EOFError):
            out = None
            if(verbose >= 1): print('Failed to load ' + name + '.json at '  + path)
        return out

class DataProcedure(Storable):
    '''A wrapper for archiving the results of data grabbing and preprocessing functions of the type X,Y getData where are X is the training
        data and Y contains the labels/targets for each entry'''
    def __init__(self, _archive_dir,archive_getData, func,  *args, **kargs):
        Storable.__init__(self)
        if(isinstance(_archive_dir, str) == False and isinstance(_archive_dir, unicode) == False):
            raise TypeError("_archive_dir must be str, but got %r" %type(_archive_dir))
        if(isinstance(archive_getData, bool) == False):
            raise TypeError("archive_getData must be bool, but got %r" % type(archive_getData))
        if(isinstance(func, types.FunctionType) == False):
            raise TypeError("func must be function, but got %r" % type(func))

        self.archive_dir = _archive_dir
        self.func = func.__name__
        self.func_module = func.__module__
        self.args = args
        self.kargs = kargs
        self.archive_getData = archive_getData


    def set_encoder(self, encoder):
        '''Set the json encoder for the procedure in case its arguements are not json encodable'''
        self.encoder = encoder

    def _genJsonableDict(self):
        d = self.__dict__
        d = copy.deepcopy(d)

        d["class_name"] = self.__class__.__name__

        del d["archive_dir"]
        if('encoder' in d): del d["encoder"]
        if('decoder' in d): del d["decoder"]
        del d["hashcode"]
        return d

    def to_hashable(self):
        '''Gets a string that uniquely defines an object as far as its model, complilation, and training parameters are concerned'''

        d = self._genJsonableDict()
        #Don't hash on verbose or verbosity if they are in the function
        if("verbose" in d.get("kargs", [])): del d['kargs']["verbose"]
        if("verbosity" in d.get("kargs", [])): del d['kargs']["verbosity"]

        return self.encoder.encode(d)

    def to_json(self):
        '''Returns the json string for the Procedure with only its essential characteristics'''
        d = self._genJsonableDict()

        return self.encoder.encode(d)


    def write(self, verbose=0):
        '''Write the json string for the procedure to its directory'''
        json_str = self.to_json()
        blob_path = self.get_path()
        write_object(blob_path, 'procedure.json', json_str, verbose=verbose)

    def is_archived(self):
        '''Returns True if this procedure is already archived'''
        blob_path = self.get_path()
        data_path = blob_path+"archive.h5"
        if(os.path.exists(data_path)):
            return True
        else:
            return False

    def archive(self, X, Y):
        '''Store the DataProcedure in a directory computed by its hashcode'''
        if((not X is None) and (not Y is None)):
            blob_path = self.get_path()
            if( os.path.exists(blob_path) == False):
                os.makedirs(blob_path)
            if( os.path.exists(blob_path + 'procedure.json') == False):
                self.write()

            if(isinstance(X, list) == False): X = [X]
            if(isinstance(Y, list) == False): Y = [Y]
            h5f = h5py.File(self.get_path() + 'archive.h5', 'w')
            h5f.create_group("X")
            for i, x in enumerate(X):
                h5f.create_dataset('X/'+str(i), data=x)
            h5f.create_group("Y")
            for i, y in enumerate(Y):
                h5f.create_dataset('Y/'+str(i), data=y)
            
            h5f.close()

            #TODO: this is a really backward way of doing this
            jstr = self.to_json()
            d = json.loads(jstr)

            record_dict = {}
            record_dict['func'] = d['func']
            record_dict['module'] = d['func_module']
            record_dict['args'] = d['args']
            record_dict['kargs'] = d['kargs']

            self.write_record(record_dict)
                
        else:
            raise ValueError("Cannot archive DataProcedure with NoneType X or Y")
        

    def getData(self, archive=True, redo=False, verbose=1):
        '''Apply the DataProcedure returning X,Y from the archive or generating them from func'''

        if(self.is_archived() and redo == False):
            h5f = None
            try:
                h5f = h5py.File(self.get_path() + 'archive.h5', 'r')
                X = []
                X_group = h5f['X']
                keys = list(X_group.keys())
                keys.sort()
                for key in keys:
                    X.append(X_group[key][:])


                Y = []
                Y_group = h5f['Y']
                keys = list(Y_group.keys())
                keys.sort()
                for key in keys:
                    Y.append(Y_group[key][:])

                h5f.close()
                out = (X, Y)
                if(verbose >= 1): print("DataProcedure results %r read from archive" % self.hash())
            except Exception as e:
                print(e)
                if(h5f != None): h5f.close()
                if(verbose >= 1): print("Failed to load archive %r running from scratch" % self.hash())
                return self.getData(archive=archive, redo=True, verbose=verbose)
        else:
            prep_func = self.get_func(self.func, self.func_module)

            out = prep_func(*self.args, **self.kargs)
            

            if(isinstance(out, tuple)):
                if(len(out) == 2):
                    if( (isinstance(out[0], list) or isinstance(out[0], np.ndarray)) 
                        and (isinstance(out[1], list) or isinstance(out[1], np.ndarray))):
                        if(self.archive_getData == True or archive == True):
                            self.archive(out[0], out[1])
                            if(verbose >= 1): print("ARCHIVE SUCCESSFUL %r" % self.hash())
                else:
                    raise ValueError("getData returned too many arguments expected 2 got %r" % len(out))
            elif(isinstance(out, types.GeneratorType)):
                    self.archive_getData = False
                    archive = False
            else:
                raise ValueError("getData did not return (X,Y) or Generator got types (%r,%r)"
                                    % ( type(out[0]), type(out[1]) ))
            
        return out

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
        try:
            exec("from " + module +  " import " + name + " as prep_func")
        except ImportError:
            raise ValueError("DataProcedure function %r does not exist in %r. \
                For best results functions should be importable and not locally defined." % (str(name), str(module)))
        return prep_func

    @classmethod
    def from_json(cls, archive_dir ,json_str, arg_decode_func=None):  
        '''Get a DataProcedure object from its json string'''
        d = json.loads(json_str)
        func = None
        temp = lambda x: 0
        try:
            func = cls.get_func(d['func'], d['func_module'])
        except ValueError:
            func = temp
        args, kargs = d['args'], d['kargs']
        for i,arg in enumerate(args):
            islist = True
            if(isinstance(arg, list) == False):
                islist = False
                arg = [arg]
            for j ,a in enumerate(arg):
                if(isinstance(a, str) or isinstance(a, unicode)):
                    try:
                        obj = json.loads(a)
                    except ValueError as e:
                        continue
                    if(isinstance(obj, dict)):
                        # print(type(a), type(obj))
                        if(obj.get('class_name', None) == "DataProcedure"):
                            arg[j] = DataProcedure.from_json(archive_dir, a,arg_decode_func)
            if(islist == False):
                arg = arg[0]
            args[i] = arg
        if(arg_decode_func != None):
            # print('arg_decode_func_ENABLED:', arg_decode_func.__name__)
            args, kargs = arg_decode_func(*args, **kargs)

        archive_getData = d['archive_getData']
        dp = cls(archive_dir, archive_getData, func, *args, **kargs)
        if(func == temp):
            dp.func = d['func']
            dp.func_module = d['func_module']
        return dp

    @staticmethod
    def get_all_paths(archive_dir):
        '''Gets the blob paths of all of the DataProcedures in the archive_dir'''
        paths = Storable.get_all_paths(archive_dir)
        paths = [p for p in paths if os.path.isfile(p + "procedure.json")]
        return paths

# INPUT_DEFAULTS ={
#                 "metrics" : [],
#                 "sample_weight_mode" : None,

#                 "validation_split" : 0.0,
#                 "batch_size" : 32,
#                 "nb_epoch" : 10,
#                 "callbacks" : [],

#                 "shuffle" : True,
#                 "class_weight" : None,
#                 "sample_weight" : None,

#                 "nb_val_samples" : None,
#                 "max_q_size" : 10,
#                 "nb_worker" :1,
#                 "pickle_safe" : False
#                 }
INPUT_DEFAULTS = {
    "model":None,
    "train_procedure":None,
    "samples_per_epoch":None,
    "validation_split":0.0,
    "val_procedure":None,
    "nb_val_samples":None,

    "optimizer":None,
    "loss":None,
    "metrics":[],
    "sample_weight_mode":None,

    "batch_size":32,
    "nb_epoch":10,
    "callbacks":[],
    
    "max_q_size":10,
    "nb_worker":1,
    "pickle_safe":False,

    "shuffle":True,
    "class_weight":None,
    "sample_weight":None,
}

REQUIRED_INPUTS = set(["optimizer", "loss"])


class KerasTrial(Storable):
    '''An archivable object representing a machine learning trial in keras'''
    def __init__(self,
                    archive_dir,
                    name = 'trial',
    				# model=None,
        #             train_procedure=None,
        #             samples_per_epoch=None,
        #             validation_split=0.0,
        #             val_procedure=None,
        #             nb_val_samples=None,

        #             optimizer=None,
        #             loss=None,
        #             metrics=[],
        #             sample_weight_mode=None,

        #             batch_size=32,
        #             nb_epoch=10,
        #             callbacks=[],
                    
        #             max_q_size=10,
        #             nb_worker=1,
        #             pickle_safe=False,

        #             shuffle=True,
        #             class_weight=None,
        #             sample_weight=None,
                    **kargs
                ):
    	
        Storable.__init__(self)
        if(archive_dir[len(archive_dir)-1] != "/"):
            archive_dir = archive_dir + "/"
        self.archive_dir = archive_dir
        self.name = name
        for key in INPUT_DEFAULTS:
            if(not key in kargs):
                kargs[key] = INPUT_DEFAULTS[key]

        self.setModel(kargs["model"])

        self.setTrain(train_procedure=kargs["train_procedure"],samples_per_epoch=kargs["samples_per_epoch"])

        self.setValidation( validation_split=kargs["validation_split"],
                            val_procedure=kargs["val_procedure"],
                            nb_val_samples=kargs["nb_val_samples"])

        self.setCompilation(optimizer=kargs["optimizer"],
                                loss=kargs["loss"],
                                metrics=kargs["metrics"],
                                sample_weight_mode=kargs["sample_weight_mode"])


        self.setFit(    batch_size=kargs["batch_size"],
                        nb_epoch=kargs["nb_epoch"],
                        callbacks=kargs["callbacks"],
                        shuffle=kargs["shuffle"],
                        class_weight=kargs["class_weight"],
                        sample_weight=kargs["sample_weight"])

        self.setFit_Generator(  nb_epoch=kargs["nb_epoch"],
                                callbacks=kargs["callbacks"],
                                class_weight=kargs["class_weight"],
                                max_q_size=kargs["max_q_size"],
                                nb_worker=kargs["nb_worker"],
                                pickle_safe=kargs["pickle_safe"])
    

    def setModel(self, model):
        '''Set the model used by the trial (either the object or derived json string)'''
        self.model = model
        self.compiled_model = None
        if(isinstance(model, Model)):
            self.model = model.to_json()

    def _prep_procedure(self, procedure, name='train'):
        '''A helper function that makes sure that DataProcedures are stored as json strings for easy serialization'''
        if(procedure != None):
            if(isinstance(procedure, list) == False):
                procedure = [procedure]
            l = []
            for p in procedure:
                if(isinstance(p, DataProcedure)):
                    l.append(p.to_json())
                elif(isinstance(p, str) or isinstance(p, unicode)):
                    l.append(p)
                else:
                    raise TypeError("%r_procedure must be DataProcedure, but got %r" % (name,type(p)))
            return l
        else:
            return None

    def setTrain(self,
                   train_procedure=None, samples_per_epoch=None):
        '''Sets the training data for the trial'''
        self.train_procedure = self._prep_procedure(train_procedure, 'train')
        self.samples_per_epoch = samples_per_epoch
    
    def setValidation(self,
                  val_procedure=None, validation_split=0.0, nb_val_samples=None):
        '''Sets the training data for the trial'''
    
        if(isinstance(val_procedure, float)):
            validation_split = val_procedure
            val_procedure = None
        if(isinstance(validation_split, float) == False):
            raise TypeError("validation_split must have type float, but got %r" % type(validation_split))
        if((isinstance(nb_val_samples, int) or nb_val_samples is None) == False):
            raise TypeError("nb_val_samples must have type int, but got %r" % type(nb_val_samples))

        self.validation_split = validation_split
        self.val_procedure = self._prep_procedure(val_procedure, 'val')
        self.nb_val_samples = nb_val_samples

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
        self.callbacks=callbacks

        self.shuffle=shuffle
        self.class_weight=class_weight
        self.sample_weight=sample_weight

    def setFit_Generator(self,
                nb_epoch=10,
                callbacks=[],
                class_weight={},
                max_q_size=10,
                nb_worker=1,
                pickle_safe=False):
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

        self.nb_epoch=nb_epoch
        self.callbacks=callbacks
 
        self.class_weight=class_weight
        self.max_q_size=max_q_size
        self.nb_worker=nb_worker
        self.pickle_safe=pickle_safe


    def _json_dict_helper(self):
        '''A helper function that generates a dictionary of values from a DataProcedure, omitting data that should not be stored or hashed on'''
        temp = self.compiled_model
        self.compiled_model = None
        d = self.__dict__
        d = copy.deepcopy(d)
        if('name' in d): del d['name']
        if('archive_dir' in d): del d['archive_dir']
        if('hashcode' in d): del d['hashcode']
        if('compiled_model' in d): del d['compiled_model']
        self.compiled_model = temp
        return d

    def _remove_dict_defaults(self, d):
        '''Removes default values from trial for forward compatabilty'''
        del_keys = []
        for key in d:
            if(key in INPUT_DEFAULTS and INPUT_DEFAULTS[key] == d[key]):
                del_keys.append(key)
        for key in del_keys:
            del d[key]
        return d
    def to_hashable(self):
        '''Converts the trial to a hashable string '''
        temp = self.model

        #Doesn't hash on the names of layers since they are random, and doesn't hash on keras_version
            #for forward and backward compatability.
        names = [s.replace('"name": ', "" ) for s in re.findall(r'"name": "[^"]*"', self.model)]
        for name in names:
            self.model = self.model.replace(name, "@")
        self.model = re.sub(r'"keras_version": "[^"]*"', "", self.model)
        

        #Doesn't hash on anything that is its default value for forward compatability. For example
            # if a parameter is added in a new version of keras and takes its default value then we 
            # should not hash on it since otherwise the user will find that trails created in the 
            # new version will hash differently than trials run with older versions of keras that 
            # did not have the extra parameter, even though the user didn't change anything.
        d = self._json_dict_helper()
        d = self._remove_dict_defaults(d)
        json_str = self.encoder.encode(d)
        self.model = temp
        return json_str

    def to_json(self):
        '''Converts the trial to a json string '''
        d = self._json_dict_helper()
        return self.encoder.encode(d)

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

    def _generateCallbacks(self, verbose):
        '''A helper function that Generates a SmartCheckpoint used for storing the training history of a trial, plus more'''
        callbacks = []
        for c in self.callbacks:
            if(c != None):
                callbacks.append(decodeCallback(c))
        monitor = 'val_acc'
        if(self.validation_split == 0.0 and self.val_procedure is None):
            monitor = 'acc'
        callbacks.append(SmartCheckpoint('weights', associated_trial=self,
                                             monitor=monitor,
                                             verbose=verbose,
                                             save_best_only=True,
                                             mode='auto'))
        return callbacks

    def _history_to_record(self, record_store):
        '''A helper function that Adds the important parts of the training history to the record'''
        histDict = self.get_history()
        if(histDict != None):
            dct = {} 
            for x in record_store:
                if(histDict.get(x, None) is None):
                    continue
                dct[x] = max(histDict[x])
            self.to_record(dct)

    def fit(self, model, x_train, y_train, record_store=["val_acc"], verbose=1):
        '''Runs model.fit(x_train, y_train) for the trial using the arguments passed to trial.setFit(...)'''
        
        callbacks = self._generateCallbacks(verbose)

        model.fit(x_train, y_train,
            batch_size=self.batch_size,
            nb_epoch=self.nb_epoch,
            verbose=verbose,
            callbacks=callbacks,
            validation_split=self.validation_split,
            shuffle=self.shuffle,
            class_weight=self.class_weight,
            sample_weight=self.sample_weight)
        self._history_to_record(record_store)
       

    def fit_generator(self, model, generator, validation_data=None, record_store=["val_acc"] ,verbose=1):
        '''Runs model.fit_generator(gen, samples) for the trial using the arguments passed to trial.setFit_Generator(...)'''
        callbacks = self._generateCallbacks(verbose)

        model.fit_generator(generator, self.samples_per_epoch,
                    nb_epoch=self.nb_epoch,
                    verbose=verbose,
                    callbacks=callbacks,
                    validation_data=validation_data,
                    nb_val_samples=self.nb_val_samples,
                    class_weight=self.class_weight,
                    max_q_size=self.max_q_size,
                    nb_worker=self.nb_worker,
                    pickle_safe=self.pickle_safe)
        self._history_to_record(record_store)


    def write(self, verbose=0):
        '''Writes the model's json string to its archive location''' 
        json_str = self.to_json()
        blob_path = self.get_path()
        write_object(blob_path, 'trial.json', json_str, verbose=verbose)

        self.to_record({'name' : self.name}, append=True)
                 

    def execute(self, archiveTraining=True, archiveValidation=True, train_arg_decode_func=None, val_arg_decode_func=None, custom_objects={}, verbose=1):
        '''Executes the trial, fitting the traing data in each DataProcedure in series'''
    	if(self.train_procedure is None):
            raise ValueError("Cannot execute trial without DataProcedure")
        if(self.is_complete() == False):
            model = self.compile(custom_objects=custom_objects)
            train_procs = self.train_procedure
            if(isinstance(train_procs, list) == False): train_procs = [train_procs]
            # print(train_procs)
            num_train = 0
            num_val = None
            if(self.val_procedure != None):
                if(len(self.val_procedure) != 1):
                    raise ValueError("val_procedure must be single procedure, but got list")
                val_proc = DataProcedure.from_json(self.archive_dir,self.val_procedure[0], arg_decode_func=val_arg_decode_func)
                val = val_proc.getData(archive=archiveValidation)
                num_val = self.nb_val_samples
            else:
                val = None

            for p in train_procs:
                train_proc = DataProcedure.from_json(self.archive_dir,p, arg_decode_func=train_arg_decode_func)

                train = train_proc.getData(archive=archiveTraining)

                if(isinstance(train, types.GeneratorType)):
                    self.fit_generator(model,train, val, verbose=verbose)
                    num_train += self.samples_per_epoch
                elif(isinstance(train, tuple)):
                    if(isinstance(val,  types.GeneratorType)):
                        raise ValueError("Fit() cannot take generator for validation_data. Try fit_generator()")
                    X,Y = train
                    if(isinstance(X, list) == False): X = [X]
                    if(isinstance(Y, list) == False): Y = [Y]
                    num_train += Y[0].shape[0]
                    self.fit(model,X, Y, verbose=verbose)
                else:
                    raise ValueError("Traning DataProcedure returned useable type %r" % type(train))
            self.write()

            if(num_val == None):
                num_val = num_train*(self.validation_split)

            dct =  {'num_train' : int(num_train*(1.0-self.validation_split)),
                    'num_validation' : num_val,
                    'elapse_time' : self.get_history()['elapse_time'],
                    'fit_cycles' : len(train_procs)
                    }
            self.to_record( dct, replace=True)
        else:
            print("Trial %r Already Complete" % self.hash())


    def test(self,test_proc, test_samples=None, redo=False, archiveTraining=True, custom_objects={}, max_q_size=None, nb_worker=1, pickle_safe=False, arg_decode_func=None):
        if(max_q_size == None):
            # print("USING max_q_size: %r" % self.max_q_size)
            #sys.stdout.flush()
            max_q_size = self.max_q_size

        record_loss, record_acc = tuple(self.get_from_record(['test_loss', 'test_acc'] ))
        if(redo == True or record_loss == None or record_acc == None):
            model = self.compile(loadweights=True,custom_objects=custom_objects)
            if(isinstance(test_proc, list) == False): test_proc = [test_proc]

            assert len(test_proc) > 0, "test_proc is empty: %r" % test_proc

            sum_metrics = []
            for p in test_proc:
                if(isinstance(p, str) or isinstance(p, unicode)):
                    p = DataProcedure.from_json(self.archive_dir,p, arg_decode_func=arg_decode_func)
                elif(isinstance(p, DataProcedure) == False):
                     raise TypeError("test_proc expected DataProcedure, but got %r" % type(p))

                test_data = p.getData(archive=archiveTraining)
                n_samples = 0
                if(isinstance(test_data, types.GeneratorType)):
                    metrics = model.evaluate_generator(test_data, test_samples,
                                                        max_q_size=max_q_size,
                                                        nb_worker=nb_worker,
                                                        pickle_safe=pickle_safe)
                    n_samples = test_samples
                else:
                    X,Y = test_data
                    if(isinstance(X, list) == False): X = [X]
                    if(isinstance(Y, list) == False): Y = [Y]
                    metrics = model.evaluate(X, Y)
                    n_samples = Y[0].shape[0]
                if(sum_metrics == []):
                    sum_metrics = metrics
                else:
                    sum_metrics = [sum(x) for x in zip(sum_metrics, metrics)]
            metrics = [x/len(test_proc) for x in sum_metrics]
            self.to_record({'test_loss' : metrics[0], 'test_acc' :  metrics[1], 'num_test' : n_samples}, replace=True)
            print("Test Complete %r" % metrics)
            return metrics
        else:
            print("Test %r already Complete with test_loss: %0.4f, test_acc: %0.4f" % (self.hash(), record_loss, record_acc))
            return [record_loss, record_acc]


    @staticmethod
    def get_all_paths(archive_dir):
        '''Get a list of all the blob paths of the KerasTrials in the given archive_dir'''
        paths = Storable.get_all_paths(archive_dir)
        paths = [p for p in paths if os.path.isfile(p + "trial.json")]
        return paths


    def to_record(self, dct, append=False, replace=True):
        '''Pushes a dictionary of values to the archive record for this trial'''
        if(not isinstance(dct, dict)):
            raise TypeError("obj must be type dict, but got %r" % type(dct))
 
        trial_dict = self.read_record()
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
  
        self.write_record(trial_dict)
    

    def get_from_record(self, keys, verbose=0):
        '''Get a value from the record '''
        recordDict = self.read_record(verbose=verbose)
        if(isinstance(keys, list)):
            out = []
            for key in keys:
                out.append(recordDict.get(key, None))
        else:
            out = recordDict.get(keys, None)
        return out

    def get_history(self, verbose=0):
        '''Get the training history for this trial'''
        history = read_json_obj(self.get_path(), "history.json")
        if(history == {}):
            history = None
        return history

    def get_model(self, loadweights=False,custom_objects={}):
        '''Gets the model, optionally with the best set of weights'''
        model = model_from_json(self.model, custom_objects=custom_objects)
        if(loadweights):
            model.load_weights(self.get_path()+"weights.h5")
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
                showRecord=True,
                showTraining=False,
                showValidation=False,
                showCompilation=False,
                showFit=False,
                showModelPic=False,
                showNoneType=False,
                squat=True):
        '''Print a summary of the trial
            #Arguments:
                showName=False,showDirectory=False, showRecord=True, showTraining=True, showCompilation=True, showFit=True,
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
        def _getPairsFromKeys(record,keys):
            p_keys = [key for key in record]
            out = []
            for key in keys:
                if key in p_keys:
                    out.append( (key, record[key]) )
                    del record[key]
            return out


        if(showRecord):
            print(indent + "Record_Info:")
            record = self.read_record()
            
            if(record != None):
                groups = [  _getPairsFromKeys(record, ["name","elapse_time","fit_cycles"]),
                            _getPairsFromKeys(record, ["test_acc","val_acc", "acc", "test_loss", "val_loss", "loss"]) ,
                            _getPairsFromKeys(record, ["num_train","num_validation", "num_test"])
                        ]


                for group in groups:
                    records = []
                    for key, value in group:
                        if(key == "elapse_time"):
                            m, s = divmod(value, 60)
                            h, m = divmod(m, 60)
                            records.append(str(key) + " = " + "%d:%02d:%02d" % (h, m, s))
                        elif(re.match(".*_(acc|loss)", key) != None):
                            records.append(str(key) + " = " + "%.4f" % value)
                        else:
                            pass

                            records.append(str(key) + " = " + json.dumps(value))
                    print(indent*2 + sep.join(records))
                records = []
                for key in record:
                    records.append(str(key) + " = " + str(record[key]))
                records.sort()
                print(indent*2 + sep.join(records))
            else:
                print(indent*2 + "No record. Not stored in archive.")

        if(showTraining):
            print(indent + "Training:")
            preps = []
            for s in self.train_procedure:
                p = DataProcedure.from_json(self.archive_dir, s)
                preps.append(p.get_summary())
            print(indent*2 + sep.join(preps))
            if(self.samples_per_epoch != None):
                print(indent*2 + "samples_per_epoch = %r" % self.samples_per_epoch)

        if(showValidation):
            print(indent + "Validation:")
            if(self.val_procedure == None):
                print(indent*2 + "validation_split = %r" % self.validation_split)
            else:
                preps = []
                for s in self.val_procedure:
                    p = DataProcedure.from_json(self.archive_dir, s)
                    preps.append(p.get_summary())
                print(indent*2 + sep.join(preps))
                if(self.nb_val_samples != None):
                    print(indent*2 + "nb_val_samples = %r" % self.nb_val_samples)

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
        print("-"*50)


    @classmethod
    def from_json(cls,archive_dir,json_str, name='trial'):
        '''Reconsitute a KerasTrial object from its json string'''
        kargs = json.loads(json_str)
        for key in INPUT_DEFAULTS:
            if(not key in kargs):
                kargs[key] = INPUT_DEFAULTS[key]

        trial = cls(
                archive_dir,
                name = name,
                **kargs
                # model = d.get('model', None),
                # train_procedure=d.get('train_procedure', None),
                # samples_per_epoch=d.get('samples_per_epoch', None),
                # validation_split=d.get('validation_split', INPUT_DEFAULTS['validation_split']),
                # val_procedure=d.get('val_procedure', None),
                # nb_val_samples=d.get('nb_val_samples', INPUT_DEFAULTS['nb_val_samples']),

                # optimizer=d.get('optimizer', None),
                # loss=d.get('loss', None),
                # metrics=d.get('metrics', []),
                # sample_weight_mode=d.get('sample_weight_mode', INPUT_DEFAULTS['sample_weight_mode']),
                # batch_size=d.get('batch_size', INPUT_DEFAULTS['batch_size']),
                # nb_epoch=d.get('nb_epoch', INPUT_DEFAULTS['nb_epoch']),
                # callbacks=d.get('callbacks', INPUT_DEFAULTS['callbacks']),

                # max_q_size=d.get('max_q_size', INPUT_DEFAULTS['max_q_size']),
                # nb_worker=d.get('nb_worker',  INPUT_DEFAULTS['nb_worker']),
                # pickle_safe=d.get('pickle_safe', INPUT_DEFAULTS['pickle_safe']),

                # shuffle=d.get('shuffle', INPUT_DEFAULTS['shuffle']),
                # class_weight=d.get('class_weight',  INPUT_DEFAULTS['class_weight']),
                # sample_weight=d.get('sample_weight', INPUT_DEFAULTS['sample_weight'])
                )
        return trial
        

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
    hashable_str = inp
    if(isinstance(inp, Storable)):
        hashable_str = inp.to_hashable()
    h = hashlib.sha1()
    h.update(hashable_str)
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
                print("THIS HAPPENS")
                raise ValueError("using json_str is depricated")
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
    if not os.path.exists(directory):
        os.makedirs(directory)
    try:
        json.dump(obj,  open( directory + filename, "wb" ))
        if(verbose >= 1): print('Sucessfully wrote ' + filename +'  at ' + directory)
    except (IOError, EOFError):
        if(verbose >= 1): print('Failed to write '+ filename +'  at ' + directory)



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

def get_all_data(archive_dir,verbose=0):
    '''Gets all the DataProcedure in the data_archive'''
    return get_data_by_function('.', archive_dir,verbose=verbose)

def get_data_by_function(func, archive_dir,verbose=0):
    '''Gets a list of DataProcedure that use a certain function'''
    record = DataProcedure.get_all_records(archive_dir)
    out = []
    if(isinstance(func, str)):
        func_name = func
        func_module = None
    else:
        func_name = func.__name__
        func_module = func.__module__

    for key in record:
        t_func = record[key].get("func", 'unknown')
        t_module = record[key].get("func_module", 'unknown')
        if(re.match(func_name, t_func) != None and (func_module is None or re.match(func_module, t_module) != None)):
            dp = DataProcedure.find_by_hashcode(archive_dir,key,verbose=verbose)
            if(dp != None):
                out.append(dp)

    return out


def get_all_trials(archive_dir, verbose=0):
    '''Get all the trials listed in the trial_record'''
    return get_trials_by_name('.', archive_dir, verbose=verbose)

def get_trials_by_name(name, archive_dir, verbose=0):
    '''Get all the trials with a particluar name or that match a given regular expression'''
    record = KerasTrial.get_all_records(archive_dir)
    out = []
    for key in record:
        t_name = record[key].get("name", 'unknown')
        if(isinstance(t_name, list) == False):
            t_name = [t_name]
        if True in [re.match(name, x) != None for x in t_name]:
            trial = KerasTrial.find_by_hashcode(archive_dir,key, verbose=verbose)
            if(trial != None):
                out.append(trial)
    return out

