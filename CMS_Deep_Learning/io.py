import os,sys, types
import numpy as np
import h5py
import glob
import copy
import itertools
from six import string_types,reraise
from CMS_Deep_Learning.storage.archiving import DataProcedure,KerasTrial

#-----------------------------IO Utils----------------------------------------
def load_hdf5_dataset(data):
    """ based off - https://github.com/duanders/mpi_learn -- train/data.py
        Converts an HDF5 structure to nested lists of databases which can be
        copied to get numpy arrays or lists of numpy arrays.
        
        :param data: and h5py Group or Dataset"""
    if isinstance(data, h5py.Group):
        sorted_keys = sorted(data.keys())
        data = [data[key] for key in sorted_keys]
    return data


def retrieve_data(data, data_keys, just_length=False, assert_list=False, prep_func=None, verbose=0):
    '''Grabs raw data from a DataProcedure or file

        :param data: the data to get the raw verion of. If not str or DataProcedure returns itself
        :type data: DataProcedure or str<path> or other
        :param data_keys: The names of the keys in the hdf5 store to get the data from. Can be nested as in
                            [["HCAL", "ECAL"], "target"]
        :type data_keys: list of str
        :param just_length: If True just return the length of the data instead of the data itself
        :type just_length: bool
        :param assert_list: Whether or not the data should always be nested in a list even if there is only
                            one numpy array.
        :type assert_list: bool
        :param prep_func: a function that takes in the tuple of outputs and returns some light transformation
                        on them, for example reshaping or padding.
        :type prep_func: function
        :param verbose: whether or not to print out extra internal information
        :type verbose: int
        
        :returns: The raw data as numpy.ndarray

        '''
    assert prep_func == None or isinstance(prep_func, types.FunctionType), \
        "prep_func must be function type but got %r" % type(prep_func)
    # Applies prep_func if it does exists
    f_ret = lambda x: prep_func(x) if prep_func != None else x

    ish5 = isinstance(data, h5py.File)
    if (isinstance(data, DataProcedure)):
        return f_ret(data.get_data(data_keys=data_keys, verbose=verbose))
    elif (isinstance(data, string_types) or ish5):
        f_path = os.path.abspath(data) if not ish5 else data.filename
        h5_file = h5py.File(f_path, 'r') if not ish5 else data
        it = data_keys if isinstance(data_keys,(list,tuple)) else [data_keys]
        #for data_key in it:
        if isinstance(data_keys, (list,tuple)):
            # Get Recursively if keys are list
            out = [retrieve_data(h5_file, data_keys=data_key, just_length=just_length, assert_list=False)
                   for data_key in data_keys]
        else:
            # Grab directly from the HDF5 store
            data_key = data_keys
            try:
                data = h5_file[data_key]
            except KeyError:
                raise KeyError("No such key %r in H5 store %r." % (data_key, f_path))
            dataset = load_hdf5_dataset(data)
            if (just_length):
                nxt = len(dataset) if not isinstance(dataset, list) else [len(x) for x in dataset]
            else:
                nxt = dataset[:]
            nxt = [nxt] if (assert_list and not isinstance(nxt, list)) else nxt
            out = nxt

        return f_ret(out)
    else:
        return f_ret(data)

# ---------------------------------------------------------------------


# --------------------------SIZE UTILS-------------------------------
def nb_samples_from_h5(file_path):
    '''Get the number of samples contained in any .h5 file; numpy or pandas.

    :param file_path: The file_path
    :returns: The number of samples.
    '''
    try:
        f = d = h5py.File(file_path, 'r')
    except IOError as e:
        raise IOError(str(e) + " at %r" % file_path)
    try:
        while not isinstance(d, h5py.Dataset):
            keys = d.keys()
            d = d['axis1' if 'axis1' in keys else keys[0]]
        out = d.len()
    except IOError as e:
        raise IOError(str(e) + " at %r" % file_path)
    finally:
        f.close()
    return out


def get_sizes_meta_dict(directory, verbose=0):
    '''Returns a dictionary of the number of sample points contained in each .h5 file in a directory

    :param directory: The directory where the .h5 files are
    :returns: the sizes dictionary
    '''
    from CMS_Deep_Learning.storage.archiving import read_json_obj
    if (not os.path.isdir(directory)):
        split = os.path.split(directory)
        directory = "/".join(split[:-1])
    sizesDict = read_json_obj(directory, "sizesMetaData.json", verbose=verbose)
    return sizesDict


def size_from_meta(filename, sizesDict=None, zero_errors=True, verbose=0):
    '''Quickly resolves the number of entries in a file from metadata, making sure to update the metadata if necessary

    :param filename: The path the the file.
    :param sizesDict: (optional) the sizes dictionary gotten from get_sizes_meta_dict, if not passed will find it anyway.
    :returns: The number of samples in the file
     '''
    from CMS_Deep_Learning.storage.archiving import write_json_obj
    if (sizesDict == None):
        sizesDict = get_sizes_meta_dict(filename)
    modtime = os.path.getmtime(filename)
    if (not filename in sizesDict or sizesDict[filename][1] != modtime):
        try:
            file_total_events = nb_samples_from_h5(filename)
        except IOError as e:
            if (zero_errors):
                file_total_events = 0
            else:
                reraise(IOError, e)
        sizesDict[filename] = (file_total_events, modtime)
        if (not os.path.isdir(filename)):
            split = os.path.split(filename)
            directory = "/".join(split[:-1])
        write_json_obj(sizesDict, directory, "sizesMetaData.json", verbose=verbose)
    return sizesDict[filename][0]


# -------------------------------------------------------------

#-----------------------------GENERATOR------------------------

def _size_set(x, s=None):
    '''A helper method that contructs a set of the number of samples in a nested list of data samples
        if all is well the set should only have one element (i.e. all of the datasets
        agree on the number of samples)'''
    if (s == None): s = set([])
    if (isinstance(x, (list, tuple))):
        for y in x:
            _size_set(y, s)
    else:
        s.add(x.shape[0])
    return s


def gen_from_data(lst, batch_size, data_keys=["Particles", "Labels"],prep_func=None, verbose=1):
    '''Gets a generator that generates data of **batch_size** from a list of .h5 files or DataProcedures,
        or a directory containing .h5 files.

        :param lst: a list of .h5 filepaths and/or DataProcedures or a directory path
        :type lst: str or list
        :param batch_size: The number of samples to grab at each call to next()
        :type batch_size: int
        :param data_keys: The keys to draw from in the .h5 files. (order matters)
        :type data_keys: list of str
        :param prep_func: a function that takes in the tuple of outputs and returns some light transformation
                        on them, for example reshaping or padding.
        :param prep_func: function
        :param verbose: whether or not to print out info to the user.
        :type verbose: int
        :returns: a generator that runs through the given data
    '''
    if (isinstance(lst, string_types) and os.path.isdir(lst)):
        path = os.path.abspath(lst) + "/*.h5"
        lst = sorted(glob.glob(path))
        if(len(lst) == 0): raise IOError("No .h5 files found in directory %r" % path)
    if (isinstance(lst, DataProcedure) or isinstance(lst, string_types)): lst = [lst]
    for d in lst:
        if (not (isinstance(d, DataProcedure) or (isinstance(d, string_types) and os.path.exists(d))) ):
            if(isinstance(d, string_types)):
                raise IOError("No such file %r" % d)
            else:
                raise TypeError("List elements should be existing file path or DataProcedure but got %r" % type(d))
            

    while True:
        for i, elmt in enumerate(lst):
            flat_out = flatten(assert_list(retrieve_data(elmt, data_keys=data_keys, prep_func=prep_func, verbose=verbose)),inplace=True)
            tot_set = _size_set(flat_out)  
            assert len(tot_set) == 1, "datasets (i.e Particle,Labels,HLF) do not have same number of elements"
            tot = list(tot_set)[0]
            for start in range(0, tot, batch_size):
                end = start + min(batch_size, tot - start)
                # yield tuple([[x[start:end] for x in X] for X in out])
                yield restructure([x[start:end] for x in flat_out], data_keys)
                
#--------------------------------------------------------------

#---------------------UTILS-------------------------------
def repr_structure(x):
    '''Returns a string summary of a nested list of numpy arrays'''
    if isinstance(x,list):
        return "[" + ",".join([repr_structure(y) for y in x]) + "]"
    elif isinstance(x,tuple):
        return "(" + ",".join([repr_structure(y) for y in x]) + ")"
    elif(isinstance(x,np.ndarray)):
        return "<" + str(x.shape) + ">"
    else:
        return str(x)

def assert_list(x,seqtypes=(list, tuple)):
    '''Makes the input a list if it is not already'''
    return list(x) if isinstance(x,seqtypes) else [x]


def flatten(items, seqtypes=(list, tuple),inplace=False):
    '''Flattens an arbitrary nesting of lists'''
    if(not inplace):
        items = copy.deepcopy(items)
    items = assert_list(items,seqtypes=seqtypes)
    for i, x in enumerate(items):
        while i < len(items) and isinstance(items[i], seqtypes):
            items[i:i + 1] = items[i]
    return items


def restructure(flattened, data_keys, seqtypes=(list, tuple)):
    '''Structures a flattened list into the nested list structure given in data_keys'''
    if (not isinstance(data_keys, seqtypes)):
        return flattened[0] if isinstance(flattened, seqtypes) else flattened
    pos = 0
    out = []
    for key in data_keys:
        k = len(flatten(key)) if isinstance(key, seqtypes) else 1
        out.append(restructure(flattened[pos:pos + k], key))
        pos += k
    return out

def first_elmt(x,seqtypes=(list, tuple)):
    '''Gets the first non-list element of any set of nested lists'''
    if isinstance(x,seqtypes):
        return first_elmt(x[0],seqtypes=seqtypes)
    else:
        return x
    
#--------------------------------------------------------


#-----------------------------Iterator------------------------


if (sys.version_info[0] > 2):
    from inspect import signature
    getNumParams = lambda f: len(signature(f).parameters)
else:
    from inspect import getargspec

    getNumParams = lambda f: len(getargspec(f)[0])


class DataIterator:
    '''A tool for retrieving inputs, labels,prediction values and functions of data.
        Unlike gen_from_data aggregates data from multiple files together into a single list. 

        :param data: A generator, list of DataProcedures and/or file paths, or a directory path in which to find the data
        :type data: lst or str
        :param num_samples: If using a generator, must specify now many samples to read
        :type num_samples: uint
        :param data_keys: Which keys to grab from the data_store, these will be the first outputs of the iterator
        :type data_keys: list of str
        :param input_keys: The key in the source hdf5 store that corresponds to the input data
        :type input_keys: str
        :param label_keys: The key in the source hdf5 store that corresponds to the label data
        :type label_keys: str
        :param accumulate: An accumulator function built from CMS_Deep_Learning.postprocessing.metrics.build_accumulator
            the output of the accumulate function will follow any specified data_key data in the output
        :type accumulate: function
        :param prediction_model: A compiled model from which to get the predictions. If specified then predictions are returned
        :type prediction_model: Model
        :Output format: (data_keys outputs...,acummilate, predictions)
    '''

    def __init__(self, data, num_samples=None, data_keys=[], input_keys=["X"], label_keys=["Y"], accumulate=None,
                 prediction_model=None):
        self.num_samples = num_samples
        self.accumulate = accumulate
        self.prediction_model = prediction_model
        self.data_keys = data_keys
        self.input_keys = input_keys
        self.label_keys = label_keys

        # Make sure the data is some kind of list 
        if (not isinstance(data, list)):
            if (os.path.exists(os.path.abspath(data)) and os.path.isdir(os.path.abspath(data))):
                path = os.path.abspath(data) + "/*.h5"
                data = sorted(glob.glob(path))
                if (len(data) == 0): raise IOError("No .h5 files found in directory %r" % path)
            else:
                data = [data]

        # Resolve the full set of data that needs to be read
        if (self.accumulate != None): self.num_params = getNumParams(self.accumulate)
        self.x_required = self.prediction_model != None or self.accumulate != None
        self.y_required = self.accumulate != None and self.num_params > 1

        self.input_index, self.label_index = -1, -1
        if (not isinstance(self.data_keys, (list, tuple))):
            self.key_singluar, self.union_keys = True, [self.data_keys[:]]
        else:
            self.key_singluar, self.union_keys = False, self.data_keys[:]
        if (self.x_required):
            if (not self.input_keys in self.union_keys):
                self.union_keys.append(self.input_keys)
            self.input_index = self.union_keys.index(self.input_keys)
        if (self.y_required):
            if (not self.label_keys in self.union_keys):
                self.union_keys.append(self.label_keys)
            self.label_index = self.union_keys.index(self.label_keys)

        # Peek at the first part of the data
        if (isinstance(data[0], DataProcedure) or isinstance(data[0], string_types)):
            first_data = self._retrieve_data(data[0], self.union_keys)
            if (not isinstance(first_data, types.GeneratorType)):
                first_data = assert_list(first_data)
        else:
            first_data = data[0]

        # If it is a generator undo the peeking with chain
        if (isinstance(first_data, types.GeneratorType)):
            peek = next(first_data)
            data = itertools.chain([peek], first_data)
            first_data = peek

        self.data = data
        if (len(self.union_keys) != len(first_data)):
            raise ValueError("union_keys %r do not match data size of %r" % \
                             (self.union_keys, len(first_data)))

    def _retrieve_data(self, *args, **kwargs):
        '''Just a helper method for error better error handling retrieve_data.'''
        try:
            out = retrieve_data(*args, **kwargs)
        except KeyError as e:
            raise KeyError(str(e).replace("'", "") + str(" If these key names are unfamiliar please try setting " +
                                                         "input_keys=, label_keys= in the calling method. (e.g. input_keys=[['ECAL', 'HCAL']] )"))
        return out

    def length(self, verbose=0):
        '''Finds the length of the iterator if taken as a single list'''
        if (self.num_samples == None):
            num_samples = 0
            for d in self.data:
                lengths = flatten(self._retrieve_data(d, self.union_keys, just_length=True, verbose=verbose),inplace=True)
                assert len(set(lengths)) == 1, "Collection lengths mismatch %r, with lengths %r" % \
                                               (flatten(self.union_keys), flatten(self.union_keys))
                num_samples += lengths[0]
            self.num_samples = num_samples
        return self.num_samples

    def _assert_raw(self, d, verbose=0):
        '''Makes sure that the data is raw and not a string or DataProcdedure'''
        if (isinstance(d, DataProcedure) or isinstance(d, string_types)):
            d = self._retrieve_data(d, data_keys=self.union_keys)
        return d

    def as_list(self, verbose=0):
        '''Return the data as a list of lists/numpy arrays'''
        pos = 0
        samples_outs = [None] * len(self.union_keys)

        # Just make sure that self.num_samples is resolved
        self.length()

        acc_out = [None] * self.length(verbose=verbose) if (self.accumulate != None) else None
        pred_out = [None] * self.length(verbose=verbose) if (self.prediction_model != None) else None

        # Loop through the data, compute predictions and accum and put it in a list
        nb_data = len(self.data)
        for chunk_nb ,d in enumerate(self.data):
            if(verbose > 0):
                percent = float(chunk_nb)/nb_data
                sys.stdout.write('\r')
                sys.stdout.write("[%-20s] %r/%r " % ('=' * int(20 * percent), chunk_nb,nb_data) )                                 
                sys.stdout.flush()
            
            if (pos >= self.num_samples):
                break
            out = self._assert_raw(d, verbose=verbose)
            L = first_elmt(out).shape[0]
            for i, Z in enumerate(out):
                if (isinstance(Z, tuple)): Z = list(Z)
                if (not isinstance(Z, list)): Z = [Z]
                if (i == self.input_index): X = Z
                if (i == self.label_index): Y = Z

                flat_Z = flatten(Z)
                if (samples_outs[i] == None):
                    samples_outs[i] = [[None] * self.length(verbose=verbose) for _ in
                                       range(len(flat_Z))]
                for j, z in enumerate(flat_Z):
                    Zj_out = samples_outs[i][j]
                    for k in range(L):
                        Zj_out[pos + k] = z[k]
            if (self.prediction_model != None):
                pred = self.prediction_model.predict_on_batch(X)
                for j in range(L):
                    pred_out[pos + j] = pred[j]

            if (self.accumulate != None):
                if (self.num_params == 1):
                    acc = self.accumulate(X)
                else:
                    acc = self.accumulate(X, Y)
                for j in range(L):
                    acc_out[pos + j] = acc[j]
            pos += L
        out = []
        for key in assert_list(self.data_keys):
            Z_out = samples_outs[self.union_keys.index(key)]
            if (Z_out != None):
                for j, zo in enumerate(Z_out):
                    Z_out[j] = np.array(zo)
                Z_out = restructure(Z_out, key)
                out.append(Z_out)
        if (pred_out != None):
            out.append(np.array(pred_out))
        if (acc_out != None):
            out.append(np.array(acc_out))
        return out[0] if len(out) == 1 and self.key_singluar else tuple(out)

    def next(self):
        raise NotImplementedError("Need to actually make this an iterator")

    '''
    def _listNext():
        for p in self.proc:
            X,Y = p.getData()
            pred = self.prediction_model.predict_on_batch(X) if self.prediction_model != None else None
            acc = self.accumulate(X) if self.accumulate != None else None
            for  in
                yield next(self.proc)
        return StopIteration()
    '''

    def __iter__(self):
        raise NotImplementedError("Need to actually make this an iterator")
        return self


class TrialIterator(DataIterator):
    '''A tool for retrieving inputs, labels,prediction values and functions of data from a KerasTrial instance

        :param trial: A KerasTrial from which the model, and data can be assumed
        :type trial: KerasTrial
        :param data_type: 'val' or 'train'
        :type data_type: str
        :param data_keys: Which keys to grab from the data_store, these will be the first outputs of the iterator
        :type data_keys: list of str
        :param input_keys: The key in the source hdf5 store that corresponds to the input data
        :type input_keys: str
        :param label_keys: The key in the source hdf5 store that corresponds to the label data
        :type label_keys: str
        :param accumulate: An accumulator function built from CMS_Deep_Learning.postprocessing.metrics.build_accumulator
            the output of the  accumulate function will follow any specified data_key data in the output
        :type  accumulate: function
        :param return_prediction: Whether or not to return predictions
        :type return_prediction: bool
        :param custom_objects: A dictionary of objects created by the user keyed by their name. For example custom layers.
        :type custom_objects: dict of classes
        :Output format: (data_keys outputs..., accumulate, predictions)
    '''

    def __init__(self, trial, data_type="val", data_keys=[], input_keys="X", label_keys="Y", accumulate=None,
                 return_prediction=False, custom_objects={}):
        if (data_type == "val"):
            data = trial.get_val()
            num_samples = trial.nb_val_samples
        elif (data_type == "train"):
            data = trial.get_train()
            num_samples = trial.samples_per_epoch
        else:
            raise ValueError("data_type must be either val or train but got %r" % data_type)

        model = None
        if (return_prediction):
            model = trial.compile(loadweights=True, custom_objects=custom_objects)
        DataIterator.__init__(self, data, num_samples=num_samples, data_keys=data_keys,
                              input_keys=input_keys, label_keys=label_keys,
                              accumulate=accumulate, prediction_model=model)

#--------------------------------------------------------------------

#--------------------------------SIMPLE GRAB-----------------------------------

REQ_DICT = {"predictions": [['trial'], ['model', 'data'], ['model', 'X']],
            "characteristics": [['trial', 'accumulate'], ['model', 'data', 'accumulate'], ['model', 'X', 'accumulate']],
            "X": [['trial'], ['data']],
            "Y": [['trial'], ['data']],
            "model": [['trial']],
            "num_samples": [['trial']]}
ITERATOR_REQS = ['predictions', 'characteristics', 'X', 'Y', 'num_samples']


def assertModel(model, weights=None, loss='categorical_crossentropy', optimizer='rmsprop', custom_objects={}):
    '''Asserts that the inputs create a valid keras model and returns that model

        :param model: a keras Model or the path to a model .json
        :type model: str or Model
        :param weights: the model weights or path to the stored weights
        :type weights: str or weights
        :param loss: the loss function to compile the model with
        :type loss: str
        :param optimizer: the optimizer to compile the model with
        :type optimizer: str
        :param custom_objects: A dictionary of objects created by the user keyed by their name. For example custom layers.
        :type custom_objects: dict of classes
        :returns: A compiled model
        '''
    from keras.engine.training import Model
    from keras.models import model_from_json
    import os, sys
    '''Takes a model and weights, path and weights, json_sting and weights, or compiled model
        and outputs a compiled model'''
    if (loss == None): loss = 'categorical_crossentropy'
    if (optimizer == None): optimizer = 'rmsprop'

    if (isinstance(model, string_types)):
        if (os.path.exists(model)):
            model_str = open(model, "r").read()
        else:
            model_str = model
        model = model_from_json(model_str, custom_objects=custom_objects)
    # If not compiled
    if not hasattr(model, 'test_function'):
        if (isinstance(weights, type(None))):
            raise ValueError("Cannot compile without weights")
        if (isinstance(weights, string_types) and os.path.exists(weights)):
            model.load_weights(weights)
        else:
            model.set_weights(weights)
    return model


def assertType(x, t):
    '''Asserts that x is of type t and raises an error if not'''
    assert isinstance(x, t), "expected %r but got type %r" % (t, type(x))


def _checkAndAssert(data_dict, data_to_check):
    '''A helper function for simple_grab that checks and asserts the correct data types'''
    if ("model" in data_to_check):
        data_dict['model'] = assertModel(data_dict['model'],
                                         weights=data_dict.get('weights', None),
                                         loss=data_dict.get('loss', None),
                                         optimizer=data_dict.get('optimizer', None),
                                         custom_objects=data_dict.get('custom_objects', {})
                                         )
    if ("trial" in data_to_check): assertType(data_dict['trial'], KerasTrial)
    if ("X" in data_to_check): assertType(data_dict['X'], (np.ndarray, list, tuple))
    if ("Y" in data_to_check): assertType(data_dict['Y'], (np.ndarray, list, tuple))
    if ("predictions" in data_to_check): assertType(data_dict['predictions'], np.ndarray)
    if ("num_samples" in data_to_check): assertType(data_dict['num_samples'], int)

    return data_dict


def _call_iters(data_dict, to_return, sat_dict,verbose=0):
    '''A helper function for simple_grab that calls the DataIterators if necessary'''
    if (len(set.intersection(set(to_return), set(ITERATOR_REQS))) != 0):
        to_get = [req for req in to_return if req in ITERATOR_REQS and not req == sat_dict[req]]
        if (len(to_get) > 0):
            data_keys = []
            if ('X' in to_get): data_keys.append(data_dict.get('input_keys', 'X'))
            if ('Y' in to_get): data_keys.append(data_dict.get('label_keys', 'Y'))
            accumulate = data_dict.get('accumulate', None)  # if('accumulate' in to_get) else None
            if (sat_dict[to_get[0]][0] == 'trial'):
                dItr = TrialIterator(data_dict['trial'],
                                     data_keys=data_keys,
                                     input_keys=data_dict.get('input_keys','X'),
                                     label_keys=data_dict.get('label_keys','Y'),
                                     return_prediction='predictions' in to_get,
                                     accumulate=accumulate,
                                     custom_objects=data_dict.get('custom_objects',{}))
                out = dItr.as_list(verbose=verbose)
            else:
                dItr = DataIterator(data_dict.get('data', None),
                                    data_keys=data_keys,
                                    num_samples=data_dict.get('num_samples', None),
                                    input_keys=data_dict.get('input_keys', 'X'),
                                    label_keys=data_dict.get('label_keys', 'Y'),
                                    prediction_model=data_dict.get('model', None),
                                    accumulate=accumulate,
                                    custom_objects=data_dict.get('custom_objects', {}))
                out = dItr.as_list(verbose=verbose)
            out = assert_list(out)
            for i, key in enumerate(to_get):
                data_dict[key] = out[i]
    return data_dict


def simple_grab(to_return, verbose=0,**kargs):
    '''Returns the data requested in **to_return** given that the data can be found or derived from the given inputs.
        for example one can derive predictions from a model path, weights path, and X (input data).
        **See inputs below**. Outputs include **predictions**, **characteristics**, **X**, **Y**, **model**, **num_samples**.

        :param to_return: The set of data that you would like to get options: [predictions,X,Y,model,num_samples].
                For example if I wanted the labels and predictions I would set to_return = ['Y','predictions'].
        :type to_return: (list of str, or str)
        
        
        :param data: A directory path, or list of files that contain your dataset.
        :type data: (str,list)
        :param input_keys: Nested lists of strings or a single string representing the keys of the input collection(s)
                in your HDF5 store. This is essential for making predictions or returning X,Y from data.
        :type input_keys: (str, list)
        :param label_keys: Nested lists of strings or a single string representing the keys of the label collection(s)
                in your HDF5 store. This is essential for making predictions or returning X,Y from data.
        :type label_keys: (str, list)
        :param X: Your input data to your model
        :type X: (numpy.array or nested list of numpy.arrays)
        :param Y: Your data labels for your model
        :type Y: (numpy.array or nested list of numpy.arrays)
        :param model: The path to a model .json file, the json_string, or the compiled model
        :type model: (str,Model)
        :param  accumulate: an accumulator function for deriving **characteristics**
        :type  accumulate: function
        :param trial: A KerasTrial from which the model,weights, and dataset can be derived
        :type trial: CMS_Deep_Learning.storage.archiving.KerasTrial
        :param custom_objects: A dictionary of objects created by the user keyed by their name. For example custom layers.
        :param num_samples: The number of samples to grab for 'X' and 'Y'
        :type num_samples: int
        :type custom_objects: dict of classes
        :param loss: The loss to compile the model with
        :type loss: str
        :param optimizer: The optimizer to compile the model with
        :param predictions: Predictions of some model. However this is a useless input.
        :type predictions: numpy.array
        :param characteristic: A set of characteristics derived from **accumulate**
        :type characteristic: numpy.array
                
        :type optimizer: str
        
        :returns: the data requested in to_return'''
    flat_to_return = flatten(to_return)
    assert len(flat_to_return) > 0, 'to_return must be a nonempty list of return type_strings'
    data_dict = kargs
    data_to_check = set([])
    sat_dict = {}
    for req in flat_to_return:
        if not req in REQ_DICT:
            raise ValueError("Requirement %r not recognized" % req)
        satisfiers = REQ_DICT[req]
        ok = [not False in [x in data_dict for x in sat] \
              for sat in satisfiers]
        if (not req in data_dict and not True in ok):
            raise ValueError('To handle requirement %r need (%s) or %s' % \
                             (req, req, ' or '.join(['(' + ",".join(x) + ')' for x in satisfiers])))
        satisfier = req if req in data_dict else satisfiers[ok.index(True)]
        sat_dict[req] = satisfier
        for x in satisfier:
            data_to_check.add(x)

    data_dict = _checkAndAssert(data_dict, data_to_check)
    data_dict = _call_iters(data_dict, flat_to_return, sat_dict,verbose=verbose)
    return restructure([data_dict[r] for r in flat_to_return], to_return)