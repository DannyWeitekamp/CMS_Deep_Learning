import types
import sys, os
import numpy as np
import h5py
import glob
import itertools
from six import string_types
from CMS_Deep_Learning.storage.archiving import DataProcedure

if (sys.version_info[0] > 2):
    from inspect import signature

    getNumParams = lambda f: len(signature(f).parameters)
else:
    from inspect import getargspec

    getNumParams = lambda f: len(getargspec(f)[0])


def load_hdf5_dataset(data):
    """ based off - https://github.com/duanders/mpi_learn -- train/data.py
        Converts an HDF5 structure to nested lists of databases which can be
        copied to get numpy arrays or lists of numpy arrays."""
    if isinstance(data, h5py.Group):
        sorted_keys = sorted(data.keys())
        data = [data[key] for key in sorted_keys]
    return data


def retrieveData(data, data_keys, just_length=False, verbose=0):
    '''Grabs raw data from a DataProcedure or file
        
        :param data: the data to get the raw verion of. If not str or DataProcedure returns itself
        :type data: DataProcedure or str<path> or other
        :param data_keys: The names of the keys in the hdf5 store to get the data from
        :type data_keys: list of str
        :param just_length: If True just return the length of the data instead of the data itself
        :type just_length: bool
        :param verbose:
        :returns: The raw data as numpy.ndarray
        
        '''
    if (isinstance(data, DataProcedure)):
        return data.get_data(data_keys=data_keys, verbose=verbose)
    elif (isinstance(data, string_types)):
        h5_file = h5py.File(os.path.abspath(data), 'r')
        out = []
        for data_key in data_keys:
            data = h5_file[data_key]
            if (just_length):
                out.append(load_hdf5_dataset(data)[0].len())
            else:
                out.append(load_hdf5_dataset(data)[:])
        return tuple(out)
    else:
        return data


class DataIterator:
    '''A tool for retrieving inputs, labels,prediction values and functions of data
    
        :param data: A generator, list of DataProcedures and/or file paths, or a directory path in which to find the data
        :type data: lst or str
        :param num_samples: If using a generator, must specify now many samples to read
        :type num_samples: uint
        :param data_keys: Which keys to grab from the data_store, these will be the first outputs of the iterator
        :type data_keys: list of str
        :param input_key: The key in the source hdf5 store that corresponds to the input data
        :type input_key: str
        :param label_key: The key in the source hdf5 store that corresponds to the label data
        :type label_key: str
        :param accumilate: An accumilator function built from CMS_Deep_Learning.postprocessing.metrics.build_accumilator
            the output of the accumilate function will follow any specified data_key data in the output
        :type accumilate: function
        :param source_data_keys: Specifies the keys of the source hdf5 store... sometimes necessary
        :type source_data_keys: list of str
        :param prediction_model: A compiled model from which to get the predictions. If specified then predictions are returned
        :type prediction_model: Model
        :Output format: (data_keys outputs...,acummilate, predictions)
    '''

    def __init__(self, data, num_samples=None, data_keys=[], input_key="X", label_key="Y", accumilate=None,
                 source_data_keys=None, prediction_model=None):
        self.num_samples = num_samples
        self.accumilate = accumilate
        self.prediction_model = prediction_model
        self.data_keys = data_keys
        self.input_key = input_key
        self.label_key = label_key

        # Make sure the data is some kind of list 
        if (not isinstance(data, list)):
            if (os.path.exists(os.path.abspath(data)) and os.path.isdir(os.path.abspath(data))):
                data = sorted(glob.glob(os.path.abspath(data) + "/*.h5"))
            else:
                data = [data]

        # Resolve source_data_keys
        if (isinstance(data[0], DataProcedure)):
            if (source_data_keys == None): source_data_keys = data[0].data_keys
        if (source_data_keys == None): source_data_keys = [input_key, label_key]

        # Resolve the full set of data that needs to be read
        if (self.accumilate != None): self.num_params = getNumParams(self.accumilate)
        self.x_required = self.prediction_model != None or self.accumilate != None
        self.y_required = self.accumilate != None and self.num_params > 1

        self.input_index, self.label_index = -1, -1
        self.union_keys = self.data_keys[:]
        if (self.x_required):
            if (not self.input_key in self.union_keys):
                self.union_keys.append(self.input_key)
            self.input_index = self.union_keys.index(self.input_key)
        if (self.y_required):
            if (not self.label_key in self.union_keys):
                self.union_keys.append(self.label_key)
            self.label_index = self.union_keys.index(self.label_key)

        self.subset_ind = [source_data_keys.index(key)
                           for key in self.union_keys]

        # Peek at the first part of the data
        if (isinstance(data[0], DataProcedure) or isinstance(data[0], string_types)):
            first_data = retrieveData(data[0], self.union_keys)
        else:
            first_data = data[0]

        # If it is a generator undo the peeking with chain
        if (isinstance(first_data, types.GeneratorType)):
            peek = next(first_data)
            data = itertools.chain([peek], first_data)
            first_data = peek

        self.data = data
        if (len(self.union_keys) != len(first_data)):
            raise ValueError("source_data_keys %r do not match data size of %r" %
                             (source_data_keys, len(first_data)))

    def length(self, verbose=0):
        '''Finds the length of the iterator if taken as a single list'''
        if (self.num_samples == None):
            num_samples = 0
            for d in self.data:
                l = retrieveData(d, self.union_keys, just_length=True, verbose=verbose)[0]
                num_samples += l
            self.num_samples = num_samples
        return self.num_samples

    def _assert_raw(self, d, verbose=0):
        '''Makes sure that the data is raw and not a string or DataProcdedure'''
        if (isinstance(d, DataProcedure) or isinstance(d, string_types)):
            d = retrieveData(d, data_keys=self.union_keys)  # d.get_data(data_keys=self.union_keys,verbose=verbose)
        else:
            d = tuple([d[x] for x in self.subset_ind])
        return d

    def as_list(self, verbose=0):
        '''Return the data as a list of lists/numpy arrays'''
        samples_outs = [None] * len(self.union_keys)
        pred_out = None
        acc_out = None
        pos = 0

        # Just make sure that self.num_samples is resolved
        self.length()

        # Loop through the data, compute predictions and accum and put it in a list
        for d in self.data:
            if (pos >= self.num_samples):
                break
            out = self._assert_raw(d, verbose=verbose)

            for i, Z in enumerate(out):
                if (not isinstance(Z, list)): Z = [Z]

                if (self.input_index != -1 and i == self.input_index):
                    X = Z
                if (self.label_index != -1 and i == self.label_index):
                    Y = Z

                L = Z[0].shape[0]
                if (samples_outs[i] == None): samples_outs[i] = [[None] * self.length(verbose=verbose) for _ in
                                                                 range(len(Z))]
                for j, z in enumerate(Z):
                    Zj_out = samples_outs[i][j]
                    for k in range(L):
                        Zj_out[pos + k] = z[k]
            if (self.prediction_model != None):
                if (pred_out == None): pred_out = [None] * self.length(verbose=verbose)
                pred = self.prediction_model.predict_on_batch(X)
                for j in range(L):
                    pred_out[pos + j] = pred[j]

            if (self.accumilate != None):
                if (acc_out == None): acc_out = [None] * self.length(verbose=verbose)
                if (self.num_params == 1):
                    acc = self.accumilate(X)
                else:
                    acc = self.accumilate(X, Y)
                for j in range(L):
                    acc_out[pos + j] = acc[j]

            pos += L
        out = []
        for key in self.data_keys:
            Z_out = samples_outs[self.union_keys.index(key)]

            if (Z_out != None):
                for j, zo in enumerate(Z_out):
                    Z_out[j] = np.array(zo)
                out.append(Z_out)
        if (pred_out != None):
            out.append(np.array(pred_out))
        if (acc_out != None):
            out.append(np.array(acc_out))
        return out
    
    def next(self):
        raise NotImplementedError("Need to actually make this an iterator")
    '''
    def _listNext():
        for p in self.proc:
            X,Y = p.getData()
            pred = self.prediction_model.predict_on_batch(X) if self.prediction_model != None else None
            acc = self.accumilate(X) if self.accumilate != None else None
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
        :param input_key: The key in the source hdf5 store that corresponds to the input data
        :type input_key: str
        :param label_key: The key in the source hdf5 store that corresponds to the label data
        :type label_key: str
        :param accumilate: An accumilator function built from CMS_Deep_Learning.postprocessing.metrics.build_accumilator
            the output of the accumilate function will follow any specified data_key data in the output
        :type accumilate: function
        :param source_data_keys: Specifies the keys of the source hdf5 store... sometimes necessary
        :type source_data_keys: list of str
        :param return_prediction: Whether or not to return predictions
        :type return_prediction: bool
        :Output format: (data_keys outputs...,acummilate, predictions)
    '''

    def __init__(self, trial, data_type="val", data_keys=[], input_key="X", label_key="Y",accumilate=None,
                 source_data_keys=None, return_prediction=False,custom_objects={}):
        if (data_type == "val"):
            data = trial.get_val()
            num_samples = trial.nb_val_samples
        elif (data_type == "train"):
            data = trial.get_train()
            num_samples = trial.samples_per_epoch
        else:
            raise ValueError("data_type must be either val or train but got %r" % data_type)

        # print(data, trial.get_val(), trial.get_train)
        model = None
        if (return_prediction):
            model = trial.compile(loadweights=True, custom_objects=custom_objects)
        DataIterator.__init__(self, data, num_samples=num_samples, data_keys=data_keys,
                              input_key=input_key, label_key=label_key, source_data_keys=source_data_keys,
                              accumilate=accumilate, prediction_model=model)
