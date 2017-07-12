import os,sys, types
import numpy as np
import h5py
import glob
import itertools
from six import string_types,reraise
from CMS_Deep_Learning.storage.archiving import DataProcedure


def load_hdf5_dataset(data):
    """ based off - https://github.com/duanders/mpi_learn -- train/data.py
        Converts an HDF5 structure to nested lists of databases which can be
        copied to get numpy arrays or lists of numpy arrays."""
    if isinstance(data, h5py.Group):
        sorted_keys = sorted(data.keys())
        data = [data[key] for key in sorted_keys]
    return data


def retrieve_data(data, data_keys, just_length=False, assert_list=True, prep_func=None, verbose=0):
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
        :param prep_func: function
        :param verbose:
        :returns: The raw data as numpy.ndarray

        '''
    assert prep_func == None or isinstance(prep_func,types.FunctionType),\
        "prep_func must be function type but got %r" % type(prep_func)
    #Applies prep_func if it does exists
    f_ret = lambda x: prep_func(x) if prep_func != None else x
    
    ish5 = isinstance(data, h5py.File)
    if (isinstance(data, DataProcedure)):
        return f_ret(data.get_data(data_keys=data_keys, verbose=verbose))
    elif (isinstance(data, string_types) or ish5):
        f_path = os.path.abspath(data) if not ish5 else data.filename
        h5_file = h5py.File(f_path, 'r') if not ish5 else data
        out = []
        for data_key in data_keys:
            if isinstance(data_key, list):
                #Get Recursively keys are list
                ret = retrieve_data(h5_file, data_keys=data_key, assert_list=False, )
                out.append(ret)
            else:
                #Grab directly from the HDF5 store
                try:
                    data = h5_file[data_key]
                except KeyError:
                    raise KeyError("No such key %r in H5 store %r." % (data_key, f_path))
                if (just_length):
                    return len(load_hdf5_dataset(data)[0])
                else:
                    nxt = load_hdf5_dataset(data)[:]
                    if (assert_list):
                        out.append(nxt if isinstance(nxt, list) else [nxt])
                    else:
                        out.append(nxt)

        return f_ret(tuple(out))
    else:
        return f_ret(data)


# --------------------------SIZE UTILS-------------------------------
def nb_samples_from_h5(file_path):
    '''Get the number of samples contained in any .h5 file; numpy or pandas.

    :param file_path: The file_path
    :returns: the number of samples
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

def _size_set(x, s=None):
    '''A helper method that makes a set of the number of samples in a tree grabbed data
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
    '''Gets a generator that generates data of batch_size from a list of .h5 files or DataProcedures,
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
    from CMS_Deep_Learning.storage.iterators import retrieve_data
    if (isinstance(lst, string_types) and os.path.isdir(lst)):
        lst = glob.glob(os.path.abspath(lst) + "/*.h5")
    if (isinstance(lst, DataProcedure) or isinstance(lst, string_types)): lst = [lst]
    for d in lst:
        if (not (isinstance(d, DataProcedure) or (isinstance(d, string_types) and os.path.exists(d))) ):
            if(isinstance(d, string_types)):
                raise IOError("No such file %r" % d)
            else:
                raise TypeError("List elements should be existing file path or DataProcedure but got %r" % type(d))
            

    while True:
        for i, elmt in enumerate(lst):
            out = retrieve_data(elmt, data_keys=data_keys, prep_func=prep_func, verbose=verbose)
            tot_set = _size_set(out)  
            assert len(tot_set) == 1, "datasets (i.e Particle,Labels,HLF) to not have same number of elements"
            tot = list(tot_set)[0]
            for start in range(0, tot, batch_size):
                end = start + min(batch_size, tot - start)
                yield tuple([[x[start:end] for x in X] for X in out])