import numpy as np
import pandas as pd
import glob
import threading
from CMS_SURF_2016.utils.archiving import DataProcedure, KerasTrial
from CMS_SURF_2016.utils.meta import msgpack_assertMeta
from CMS_SURF_2016.layers.lorentz import Lorentz
from CMS_SURF_2016.layers.slice import Slice
import os
import re
import sys
import socket

class ObjectProfile():
    def __init__(self, name, max_size=100, sort_columns=None, sort_ascending=True, query=None, shuffle=False):
        ''' An object containing processing instructions for each observable object type
            #Arguements:
                name       -- The name of the data type (i.e. Electron, Photon, EFlowTrack, etc.)
                max_size   -- The maximum number of objects to use in training
                sort_columns -- What columns to sort on (See pandas.DataFrame.sort)
                sort_ascending -- Whether each column will be sorted ascending or decending (See pandas.DataFrame.sort)
                query        -- A selection query string to use before truncating the data (See pands.DataFrame.query)
                shuffle     -- Whether or not to shuffle the data
        '''
        if(max_size < -1):
            raise ValueError("max_size cannot be less than -1. Got %r" % max_size)
        self.name = name
        self.max_size = max_size
        self.sort_columns = sort_columns
        self.sort_ascending = sort_ascending
        self.query = query
        self.shuffle = shuffle
        self.class_name = self.__class__.__name__


    def __str__(self):
        main_clause = 'name:%r max_size=%r ' % (self.name, self.max_size)
        sort_clause = ''
        query_clause = ''
        if(self.sort_columns != None):
            sort_clause = 'sort_columns=%r sort_ascending=%r ' % (self.sort_columns, self.sort_ascending)
        if(self.query != None):
            query_clause = 'query=%r ' % (self.query)
        shuffle_clause = 'shuffle=%r' % self.shuffle

        return main_clause + sort_clause + query_clause + shuffle_clause
    
    __repr__ = __str__

def resolveProfileMaxes(object_profiles, label_dir_pairs, padding_multiplier = 1.0):
    '''Resolves the maximum number of objects for each ObjectProfile. Only runs if ObjectProfile.max_size
        is equal to -1 or None indicating that the value is unresolved. By resolving our max_size(s) we
        can make our preprocessing data sets as small as possible without truncating any data.
        #Arguments:
            object_profiles -- The list of ObjectProfile(s) to resolve
            label_dir_pairs -- A list of tuples of the form (label, data_directory) that contain
                                the directories to look through to find the global maximum for each
                                Object type.
            padding_ratio   -- A muliplier to either shrink or increase the size of the maxes
                                in case you are worried about previously unseen realworld data 
                                being larger than what is availiable at preprocessing.
        #Returns (void)
                '''
    unresolved = []
    maxes = {}
    for profile in object_profiles:
         if(profile.max_size == -1 or profile.max_size == None):
                unresolved.append(profile)
                maxes[profile.name] = 0
    if(len(unresolved) == 0): return
    
    for (label,data_dir) in label_dir_pairs:
        files, storeType = getFiles_StoreType(data_dir)
        # files = glob.glob(data_dir+"*.h5")
        files.sort()
        
         #Loop the files associated with the current label
        for f in files:
          
            # #Get the HDF Store for the file
            # store = pd.HDFStore(f)

            # #Get the NumValues frame which lists the number of values for each entry
            # try:
            #     num_val_frame = store.get('/NumValues')
            # except KeyError as e:
            #     raise KeyError(str(e) + " " + f)
            num_val_frame = getNumValFrame(f,storeType)

            for profile in unresolved:
                maxes[profile.name] = max(num_val_frame[profile.name].max(), maxes[profile.name])
    
    for profile in unresolved:
        profile.max_size = int(np.ceil(maxes[profile.name] * padding_multiplier))

def label_dir_pairs_args_decoder(*args, **kargs):
    '''Decodes the arguments to preprocessFromPandas_label_dir_pairs so that the ObjectProfile(s) are 
        properly reconstituted'''
    #print(args)
    out = []
    for a in args:
        if(isinstance(a, dict) and a.get('class_name', None) == "ObjectProfile"):
            profiles = a
            decoded = []
            for profile in profiles:
                # print(profile)
                decoded.append(ObjectProfile(profile['name'],
                                            profile.get('max_size', 100),
                                            profile.get('sort_columns', None),
                                            profile.get('sort_ascending', True),
                                            profile.get('query', None),
                                            profile.get('shuffle', False)))
            out.append(decoded)
        else:
            out.append(a)
    # args = list(args)
    # args[3] = out
    args = tuple(out)
    return (args, kargs)

def getFiles_StoreType(data_dir):
    if(not os.path.isdir(data_dir)):
            raise IOError("Directory %r does not exist." % data_dir)
    msgFiles = glob.glob(data_dir+"*.msg")
    hdfFiles = glob.glob(data_dir+"*.h5")
    if(len(msgFiles) == 0):
        files = hdfFiles
        storeType = "hdf5"
    elif(len(hdfFiles) == 0):
        files = msgFiles
        storeType = "msgpack"
    else:
        raise IOError("Directory %r contains both .msg files and .h5 files, please use only one \
                        filetype when generating pandas files, to avoid data repetition issues\
                        " % data_dir)

    #files = glob.glob(data_dir+"*.h5")
    if(len(files) < 1):
        raise IOError("Cannot read from empty directory %r" % data_dir)
    return (files, storeType)

def getNumValFrame(filename, storeType):
    if(storeType == "hdf5"):
        #Get the HDF Store for the file
        store = pd.HDFStore(filename)

        #Get the NumValues frame which lists the number of values for each entry
        try:
            num_val_frame = store.get('/NumValues')
        except KeyError as e:
            raise KeyError(str(e) + " " + filename)
        store.close()
        return num_val_frame
    elif(storeType == "msgpack"):
        meta_frames =  msgpack_assertMeta(filename)
        num_val_frame = meta_frames["NumValues"]
        # frames = pd.read_msgpack(f)
        # num_val_frame = frames["NumValues"]
    return num_val_frame

def padItem(x,max_size, vecsize, shuffle=False):
    '''A helper function that pads a numpy array up to MAX_SIZE or trucates it down to MAX_SIZE. If shuffle==True,
        shuffles the padded output before returning'''
    if(len(x) > max_size):
        out = x[:max_size]
    else:
        out = np.append(x ,np.array(np.zeros((max_size - len(x), vecsize))), axis=0)
    if(shuffle == True): np.random.shuffle(out)
    return out
   
    #arr[index] = np.array(padItem(x.values, max_size, shuffle=shuffle))
def preprocessFromPandas_label_dir_pairs(label_dir_pairs,start, samples_per_label, object_profiles, observ_types, verbose=1):
    '''Gets training data from folders of pandas tables
        #Arguements:
            label_dir_pairs -- a list of tuples of the form (label, directory) where the directory contains
                                tables containing data of all the same event types.
            start             --    Where to start reading (as if all of the files are part of one long list)
            samples_per_label -- The number of samples to read for each label
            object_profiles -- A list of ObjectProfile(s) corresponding to each type of observable object and
                                its preprocessing steps. 
            observ_types    -- The column headers for the data to be read from the panadas table
        #Returns:
            Training data with its correspoinding labels
            (X_train, Y_train)
    '''
    labels = [x[0] for x in label_dir_pairs]
    duplicates = list(set([x for x in labels if labels.count(x) > 1]))
    if(len(duplicates) != 0):
        raise ValueError("Cannot have duplicate lables %r" % duplicates)

    vecsize = len(observ_types)
    num_labels = len(label_dir_pairs)

    #Make sure that all the profile are proper objects and have resolved max_sizes
    for i,profile in enumerate(object_profiles):
        if(isinstance(profile, dict) and profile.get('class_name', None) == "ObjectProfile"):
            profile = ObjectProfile(profile['name'],
                                            profile.get('max_size', 100),
                                            profile.get('sort_columns', None),
                                            profile.get('sort_ascending', True),
                                            profile.get('query', None),
                                            profile.get('shuffle', False))
            object_profiles[i] = profile
        if(profile.max_size == -1 or profile.max_size == None):
            raise ValueError("ObjectProfile max_sizes must be resolved before preprocessing. \
                         Please first use: utils.preprocessing.resolveProfileMaxes(object_profiles, label_dir_pairs)")

    #Build vectors in the form [1,0,0], [0,1,0], [0, 0, 1] corresponding to each label
    label_vecs = {}
    for i, (label, data_dir) in enumerate(label_dir_pairs):
        arr = np.zeros((num_labels,))
        arr[i] = 1
        label_vecs[label] = arr
    
    X_train_indices = [None] * (len(object_profiles))
    X_train = [None] * (len(object_profiles))
    y_train = [None] * (samples_per_label * num_labels)

    #Prefill the arrays so that we don't waste time resizing lists
    for index, profile in enumerate(object_profiles):
        X_train[index] = [None] * (samples_per_label * num_labels)
        X_train_indices[index] = 0
    
    #Loop over label dir pairs and get the file list for each directory
    y_train_start = 0
    for (label,data_dir) in label_dir_pairs:

        files, storeType = getFiles_StoreType(data_dir)
        files.sort()
        samples_read = 0
        location = 0
         #Loop the files associated with the current label
        for f in files:
            
            # if(storeType == "hdf5"):
            #     #Get the HDF Store for the file
            #     store = pd.HDFStore(f)

            #     #Get the NumValues frame which lists the number of values for each entry
            #     try:
            #         num_val_frame = store.get('/NumValues')
            #     except KeyError as e:
            #         raise KeyError(str(e) + " " + f)
            # elif(storeType == "msgpack"):
            #     print("Bulk reading .msg. Be patient, reading in slices not supported.")
            #     sys.stdout.flush()
            #     frames = pd.read_msgpack(f)
            #     num_val_frame = frames["NumValues"]
            num_val_frame = getNumValFrame(f,storeType)

            file_total_entries = len(num_val_frame.index)

            assert file_total_entries > 0, "num_val_frame has zero values"
            
            if(location + file_total_entries <= start):
                location += file_total_entries
                continue


            if(storeType == "hdf5"):
                store = pd.HDFStore(f)
            elif(storeType == "msgpack"):
                print("Bulk reading .msg. Be patient, reading in slices not supported.")
                sys.stdout.flush()
                #Need to check for latin encodings due to weird pandas default
                try:
                    frames = pd.read_msgpack(f)
                except UnicodeDecodeError as e:
                    frames = pd.read_msgpack(f, encoding='latin-1')
            #Determine what row to start reading the num_val table which contains
            #information about how many rows there are for each entry
            file_start_read = start-location
            if(file_start_read < 0): file_start_read = 0
            #How many rows we will read from this table each corresponds to one entry
            samples_to_read = min(samples_per_label-samples_read, file_total_entries-file_start_read)
            assert samples_to_read >= 0
            
            #Get information about how many rows there are for each entry for the rows we want to skip and read
            skip_val_frame = num_val_frame[:file_start_read]
            num_val_frame = num_val_frame[file_start_read : file_start_read+samples_to_read]

            
            #Sample is another word for entry
            if(verbose >= 1): print("Reading %r samples from %r:" % (samples_to_read,f))
            
            #Loop over every profile and read the corresponding tables in the pandas_unjoined file
            for index, profile in enumerate(object_profiles):
                key = profile.name
                max_size = profile.max_size
                if(verbose >= 1): print("Mapping %r Values/Sample from %r" % (max_size, key))
                skip = skip_val_frame[key]
                
                #Where to start reading the table based on the sum of the selection start 
                select_start = skip.sum()
                nums = num_val_frame[key]
                select_stop = select_start + nums.sum()
                
                if(storeType == "hdf5"):
                    #If we are reading all the samples use get since it might be faster
                    #TODO: check if it is actually faster
                    if(samples_to_read == file_total_entries):
                        frame = store.get('/'+key)
                    else:
                        frame = store.select('/'+key, start=select_start, stop=select_stop)
                elif(storeType == "msgpack"):
                    frame = frames[key]
                    frame = frame[select_start:select_stop]
                
                arr_start = X_train_indices[index]
                arr = X_train[index]

                #Group by Entry
                groups = frame.groupby(["Entry"], group_keys=True)#[observ_types]
                group_itr = iter(groups)
                
                #Go through the all of the groups by entry and apply preprocessing based off of the object profile
                #TODO: is a strait loop slow? Should I use apply(lambda...etc) instead? Is that possible if I need to loop
                #      over index, x and not just x?
                for entry, x in group_itr:
                    if(profile.query != None):
                        x = x.query(profile.query)
                    if(profile.sort_columns != None):
                        x = x.sort(profile.sort_columns, ascending=profile.sort_ascending)
                    x = padItem(x[observ_types].values, max_size, vecsize, shuffle=profile.shuffle)
                    arr[arr_start + entry - file_start_read] = x
                
                #Go through the all of the entries that were empty for this datatype and make sure we pad them with zeros
                for i in range(arr_start, arr_start+samples_to_read):
                    if(arr[i] is None):
                        arr[i] = np.array(np.zeros((max_size, vecsize)))
                        
                #Iterate by samples to read so that we know how many are left when we read the next file
                X_train_indices[index] += samples_to_read

                #Free these (probably not necessary)
                frame = None
                groups = None

            #Free this (probably not necessary)
            num_val_frame = None
	    if(storeType == "hdf5"):
                store.close()
            location     += file_total_entries
            samples_read += samples_to_read
            if(verbose >= 1): print("*Read %r Samples of %r in range(%r, %r)" % (samples_read, samples_per_label, start, samples_per_label+start))
            if(samples_read >= samples_per_label):
                print('-' * 50)
                assert samples_read == samples_per_label
                break
        if(samples_read != samples_per_label):
            print(samples_read, samples_per_label)
            raise IOError("Not enough data in %r to read in range(%r, %r)" % (data_dir, start, samples_per_label+start))
        
        #Generate the target data as vectors like [1,0,0], [0,1,0], [0,0,1]
        for i in range(samples_per_label):
            y_train[y_train_start+i] = label_vecs[label]
        y_train_start += samples_per_label
    
    #Turn everything into numpy arrays and shuffle them just in case.
    #Although, we probably don't need to shuffle since keras shuffles by default.
    y_train = np.array(y_train)
    
    indices = np.arange(len(y_train))
    np.random.shuffle(indices)
    for index in range(len(X_train)):
        X_train[index] = np.array(X_train[index])[indices]

    y_train = y_train[indices]
    return X_train, y_train

def getGensDefaultFormat(archive_dir, splits, length, object_profiles, label_dir_pairs, observ_types, batch_size=100, megabytes=500):
    stride = strideFromTargetSize(object_profiles, label_dir_pairs, observ_types, megabytes=500)
    SNs = start_num_fromSplits(splits, length)
    all_dps = []
    all_datasets = []
    for s in SNs:
        dps = procsFrom_label_dir_pairs(s[0],
                                        s[1],
                                        stride,
                                        archive_dir,
                                        label_dir_pairs,
                                        object_profiles,
                                        observ_types)
        gen_DP = DataProcedure(archive_dir, False,genFromPPs,dps, batch_size, threading = False)
        num_samples = len(label_dir_pairs)*s[1]
        all_datasets += [(gen_DP, num_samples)]
        all_dps += dps
    return (all_dps,all_datasets)


def batchAssertArchived(dps, scripts_dir='/scratch/daint/dweiteka/scripts/', dp_out_dir='/scratch/daint/dweiteka/dp_out/'):
    unarchived = []
    dependencies = []
    for dp in dps:
        if(not dp.is_archived()):
            unarchived.append(dp)

    if("daint" in socket.gethostname()):
        if(not os.path.exists(scripts_dir + "tmp/")):
            os.makedirs(scripts_dir + "tmp/")
        runDPs_file = scripts_dir + "tmp/runDPs.sh"
        f = open(runDPs_file, 'w')
        os.chmod(runDPs_file, 0o777)
        f.write("#!/bin/bash\n")
        for u in unarchived:
            u.write()
            ofile = dp_out_dir + u.hash()[:5] + ".%j"
            print("OutFile: ",ofile)
            f.write('sbatch -t 01:00:00 -o %s -e %s %srunDP.sh %s %s\n' % (ofile,ofile,scripts_dir,archive_dir,u.hash()))
            
        f.close()
        
        out = os.popen(scripts_dir+"tmp/runDPs.sh").read()
        print("THIS IS THE OUTPUT:",out)
        dep_clause = ""
        matches = re.findall("Submitted batch job [0-9]+", out) 

        dependencies = [re.findall("[0-9]+", m)[0] for m in matches]
        if(len(dependencies) > 0):
            dep_clause = "--dependency=afterok:" + ":".join(dependencies)
    else:
        for u in unarchived:
            u.getData(archive=True)
    return dependencies

def batchExecuteAndTestTrials(archive_dir, tups, time_str="12:00:00", scripts_dir='/scratch/daint/dweiteka/scripts/', trial_out_dir='/scratch/daint/dweiteka/trial_out/'):
    print("PRINTINTINTITNT:",archive_dir, tups, time_str)
    if("daint" in socket.gethostname()):
        for trial, test, num_test, deps in tups:
            hashcode = trial.hash()
            test.write()
            test_hashcode = test.hash()
            dep_clause = "" if len(deps)==0 else "--dependency=afterok:" + ":".join(deps)
            ofile = trial_out_dir + hashcode[:5] + ".%j"
            sbatch = 'sbatch -t %s -o %s -e %s %s' % (time_str,ofile,ofile,dep_clause)
            sbatch += '%srunTrial.sh %s %s %s %s\n' % (scripts_dir,archive_dir,hashcode, test_hashcode, num_test)
            print(sbatch)
            out = os.popen(sbatch).read()
            print("THIS IS THE OUTPUT:",out)
    else:
        for trial, test, num_test, deps in tups:
            hashcode = trial.hash()
            test.write()
            test_hashcode = test.hash()
            trial = KerasTrial.find_by_hashcode(archive_dir, hashcode)
            trial.execute(custom_objects={"Lorentz":Lorentz,"Slice": Slice})

            test = DataProcedure.find_by_hashcode(archive_dir,test_hashcode)
            print("T@:", type(test))
            trial.test(test_proc=test,
                         test_samples=num_test,
                         custom_objects={"Lorentz":Lorentz,"Slice": Slice})

def strideFromTargetSize(object_profiles, num_labels, observ_types, megabytes=100):
    if(isinstance(num_labels, list)): num_labels = len(num_labels)
    megabytes_per_sample = sum(o.max_size for o in object_profiles) * len(observ_types) * 24.0 / (1000.0 * 1000.0)
    print(megabytes_per_sample)
    return int(megabytes/megabytes_per_sample)

def maxMutualLength(label_dir_pairs, object_profiles):
    '''Gets the mamximum number of samples that can mutually be read in the directories listed by
        label_dir_pairs. Must also input object_profiles so that it knows what keys to check '''
    label_totals = {}
    for (label,data_dir) in label_dir_pairs:

        files, storeType = getFiles_StoreType(data_dir)

        files.sort()
        
        keys = None
        if(object_profiles != None):
            keys = ["/" + o.name for o in object_profiles]
        
        label_totals[label] = 0
         #Loop the files associated with the current label
        
        for f in files:
            #Get the HDF Store for the file
            if(storeType == "hdf5"):
                #Get the HDF Store for the file
                store = pd.HDFStore(f)

                #Get the NumValues frame which lists the number of values for each entry

                if(keys != None and set(keys).issubset(set(store.keys())) == False):
                    raise KeyError('File: ' + f + ' may be corrupted:' + os.linesep + 
                                    'Requested keys: ' + str(keys) + os.linesep + 
                                    'But found keys: ' + str(store.keys()) )
                
                try:
                    num_val_frame = store.get('/NumValues')
                except KeyError as e:
                    raise KeyError(str(e) + " " + f)
            elif(storeType == "msgpack"):
                print("Bulk reading .msg. Be patient, reading in slices not supported.")
                sys.stdout.flush()

                #Need to check for latin encodings due to weird pandas default
                try:
                    frames = pd.read_msgpack(f)
                except UnicodeDecodeError as e:
                    frames = pd.read_msgpack(f, encoding='latin-1')
                num_val_frame = frames["NumValues"]
                    # store = pd.HDFStore(f)
                    # #print(keys)
                    # #print(store.keys())
                    # #print(set(keys).issubset(set(store.keys())))
                    # if(keys != None and set(keys).issubset(set(store.keys())) == False):
                    #     raise KeyError('File: ' + f + ' may be corrupted:' + os.linesep + 
                    #                     'Requested keys: ' + str(keys) + os.linesep + 
                    #                     'But found keys: ' + str(store.keys()) )
                    
                    # #Get file_total_entries
                    # try:
                    #     num_val_frame = store.get('/NumValues')
                    # except KeyError as e:
                    #     raise KeyError(str(e) + " " + f)
            file_total_entries = len(num_val_frame.index)
            label_totals[label] += file_total_entries
    #print(label_totals)
    return min(label_totals.items())[1]

def start_num_fromSplits(splits, length):
    '''Takes in a tuple of splits and a length and returns a list of tuples with the starts and number of
        samples for each split'''
    if(True in [x < 0.0 for x in splits]):
        raise ValueError("Splits cannot be negative %r" % str(splits)) 
    are_static_vals = [(True if int(x) > 0 else False) for x in splits]
    if(True in are_static_vals):
        ratios =  [s for s, a in zip(splits, are_static_vals) if(not a)]
        print(ratios)
        static_vals =  [s for s, a in zip(splits, are_static_vals) if(a)]
        s = sum(static_vals) 
        if(s > length):
            raise ValueError("Static values have sum %r exceeding given length %r" %(s,length)) 
        length -= s
    else:
        ratios = splits
    
    if(len(ratios) > 0 and np.isclose(sum(ratios),1.0) == False):
        raise ValueError("Sum of splits %r must equal 1.0" % sum(ratios))
    

    nums = [int(s) if(a) else int(s*length) for s, a in zip(splits, are_static_vals)]
    out = []
    start = 0
    for n in nums:
        out.append((start, n))
        start += n
    return out



def procsFrom_label_dir_pairs(start, samples_per_label, stride, archive_dir,label_dir_pairs, object_profiles, observ_types, verbose=1):
    '''Gets a list of DataProcedures that use preprocessFromPandas_label_dir_pairs to read from the unjoined pandas files
        #Arguments
            start -- Where to start reading in the filesystem (if we treat it as one long list for each directory)
            samples_per_label -- How many samples to read from the filesystem per event type
            stride -- How many samples_per_label to grab in each DataProcedure. This should be big enough to avoid 
                    excessive reads but small enough so that samples_per_label*labels total samples can fit reasonably
                    in memory.
            archive_dir -- the archive directory to store the preprocessed data.
            label_dir_pairs -- A list of tuples like (label_name, pandas_data_directory) telling us what to call the data
                                and where to find it.
            object_profiles -- A list of ObjectProfiles, used to determine what preprocessing steps need to be taken
            observ_types -- A list of the observable quantities in our pandas tables i.e ['E/c', "Px" ,,,etc.]
            verbose -- Whether or not to print
    '''
    procs = []
    end = start+samples_per_label
    if(verbose >= 1): print("Generating DataProcedure in range(%r,%r):" % (start, end))
    for proc_start in range(start, end, stride):
        proc_num = min(stride, end-proc_start)
        dp = DataProcedure(
                archive_dir,
                True,
                preprocessFromPandas_label_dir_pairs,
                label_dir_pairs,
                proc_start,
                proc_num,
                object_profiles,
                observ_types
            )
        procs.append(dp)
        #print(proc_start, samples_per_label, stride)
        if(verbose >= 1):
            num_lables = len(label_dir_pairs)
            print("   From %r labels in range(%r,%r) for %rx%r = %r Samples"
                     % (num_lables,proc_start, proc_start+proc_num, num_lables,proc_num,num_lables*proc_num))
    #print([p.hash() for p in procs])
    return procs

class dataFetchThread(threading.Thread):

    def __init__(self, proc, group=None, target=None, name=None,
                 args=(), kwargs=None, verbose=None):
        threading.Thread.__init__(self, group=group, target=target, name=name,
                                  verbose=verbose)
        self.proc = proc
        self.args = args
        self.kwargs = kwargs
        self.X = None
        self.Y = None
        return

    def run(self):
        self.X, self.Y = self.proc.getData()
        return

def genFromPPs(pps, batch_size, threading=False):
    '''Gets a generator that generates data of batch_size from a list of DataProcedures.
        Optionally uses threading to apply getData in parellel, although this may be obsolete
        with the proper fit_generator settings'''
    for pp in pps:
        if(isinstance(pp, DataProcedure) == False):
            raise TypeError("Only takes DataProcedure got" % type(pp))
            
    
    while True:
        if(threading == True):
            datafetch = dataFetchThread(pps[0])
            datafetch.start()
        for i in range(0,len(pps)):
            if(threading == True):
                #Wait for the data to come in
                while(datafetch.isAlive()):
                    pass
                X,Y = datafetch.X, datafetch.Y

                #Start the next dataFetch
                if(i != len(pps)-1):
                    datafetch = dataFetchThread(pps[i+1])
                else:
                    datafetch = dataFetchThread(pps[0])
                datafetch.start()
            else:
                X,Y = pps[i].getData()
                                   
            if(isinstance(X,list) == False): X = [X]
            if(isinstance(Y,list) == False): Y = [Y]
            tot = Y[0].shape[0]
            assert tot == X[0].shape[0]
            for start in range(0, tot, batch_size):
                end = start+min(batch_size, tot-start)
                yield [x[start:end] for x in X], [y[start:end] for y in Y]
                

def genFrom_label_dir_pairs(start, samples_per_label, stride, batch_size, archive_dir,label_dir_pairs, object_profiles, observ_types):
    '''Gets a data generator that use DataProcedures and preprocessFromPandas_label_dir_pairs to read from the unjoined pandas files
        and archive the results.
        #Arguments
            start -- Where to start reading in the filesystem (if we treat it as one long list for each directory)
            samples_per_label -- How many samples to read from the filesystem per event type
            stride -- How many samples_per_label to grab in each DataProcedure. This should be big enough to avoid 
                    excessive reads but small enough so that samples_per_label*labels total samples can fit reasonably
                    in memory.
            batch_size -- The batch size of the generator. How many samples it grabs in each batch.
            archive_dir -- the archive directory to store the preprocessed data.
            label_dir_pairs -- A list of tuples like (label_name, pandas_data_directory) telling us what to call the data
                                and where to find it.
            object_profiles -- A list of ObjectProfiles, used to determine what preprocessing steps need to be taken
            observ_types -- A list of the observable quantities in our pandas tables i.e ['E/c', "Px" ,,,etc.]
            verbose -- Whether or not to print
    '''
    pps = procsFrom_label_dir_pairs(start,
                                    samples_per_label,
                                    stride,
                                    archive_dir,
                                    label_dir_pairs,
                                    object_profiles,
                                    observ_types)
    gen = genFromPPs(pps, batch_size, threading = False)
    return gen

def XY_to_CSV(X,Y, csvdir):
    '''Writes a pair of data X and Y to a directory csvdir as .csv files'''
    if(csvdir[len(csvdir)-1] != "/"):
        csvdir = csvdir + "/"
    if(not os.path.isdir(csvdir)):
        os.makedirs(csvdir)
    X_path = csvdir+"X/"
    Y_path = csvdir+"Y/"
    if(not os.path.isdir(X_path)):
        os.makedirs(X_path)
    if(not os.path.isdir(Y_path)):
        os.makedirs(Y_path)
    if(not isinstance(X, list)): X = [X]
    if(not isinstance(Y, list)): Y = [Y]
    def writeit(obj, path, strbeginning):
        shape = obj.shape
        p = path+strbeginning + str(i) + ".csv"
        f = open(p, "wb")
        f.write("#Shape: "+str(shape)+"\n")
        reshaped = np.reshape(obj, (shape[0], np.prod(shape[1:])))
        np.savetxt(f, reshaped, delimiter=",")
        f.close()
    for i,x in enumerate(X):
        writeit(x, X_path, "X_")
        
    for i,y in enumerate(Y):
        writeit(y, Y_path, "Y_")


def XY_from_CSV(csvdir):
    '''Reads a pair of data X and Y from a directory csvdir that contains .csv files with the data'''
    if(csvdir[len(csvdir)-1] != "/"):
        csvdir = csvdir + "/"
    def readit(path):
        f = open(path, "rb")
        shape_str = f.readline()
        shape = tuple([int(re.sub("\D", "", s)) for s in shape_str.split(",")])
        arr = np.loadtxt(f,delimiter=',')
        return np.reshape(arr, shape)
    X_path = csvdir+"X/"
    Y_path = csvdir+"Y/"
    if(not os.path.isdir(X_path) or not os.path.isdir(Y_path)):
        raise IOError("csv directory does not contain X/, Y/")
   
    files = glob.glob(X_path+"*")
    files.sort()
    X = []
    for p in files:
        X.append(readit(p))
        
    files = glob.glob(Y_path+"*")
    files.sort()
    Y = []
    for p in files:
        Y.append(readit(p))
        
    return X,Y


def XY_to_pickle(X,Y, pickledir):
    '''Writes a pair of data X and Y to a directory pickledir as pickled files'''
    if(pickledir[len(pickledir)-1] != "/"):
        pickledir = pickledir + "/"
    if(not os.path.isdir(pickledir)):
        os.makedirs(pickledir)
    X_path = pickledir+"X/"
    Y_path = pickledir+"Y/"
    if(not os.path.isdir(X_path)):
        os.makedirs(X_path)
    if(not os.path.isdir(Y_path)):
        os.makedirs(Y_path)
    if(not isinstance(X, list)): X = [X]
    if(not isinstance(Y, list)): Y = [Y]
    def writeit(obj, path, strbeginning):
        shape = obj.shape
        p = path+strbeginning + str(i) 
        np.save(p,obj)
    for i,x in enumerate(X):
        writeit(x, X_path, "X_")
        
    for i,y in enumerate(Y):
        writeit(y, Y_path, "Y_")

def XY_from_pickle(pickledir):
    '''Reads a pair of data X and Y from a directory pickledir that contains pickle files with the data'''
    if(pickledir[len(pickledir)-1] != "/"):
        pickledir = pickledir + "/"
    def readit(path):
        arr = np.load(path)
        return arr
    X_path = pickledir+"X/"
    Y_path = pickledir+"Y/"
    if(not os.path.isdir(X_path) or not os.path.isdir(Y_path)):
        raise IOError("Pickle directory does not contain X/, Y/")
   
    files = glob.glob(X_path+"*")
    files.sort()
    X = []
    for p in files:
        X.append(readit(p))
        
    files = glob.glob(Y_path+"*")
    files.sort()
    Y = []
    for p in files:
        Y.append(readit(p))
        
    return X,Y

