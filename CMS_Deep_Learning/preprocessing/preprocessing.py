import glob
import os
import re
import sys
import threading
import time

import numpy as np
import pandas as pd
import h5py
from six import string_types,reraise

from CMS_Deep_Learning.storage.archiving import DataProcedure,read_json_obj,write_json_obj
from CMS_Deep_Learning.storage.meta import msgpack_assertMeta
from CMS_Deep_Learning.io import get_sizes_meta_dict, size_from_meta,gen_from_data



DEFAULT_PROFILE = {
                        "name" : " ",
                        "max_size" : 100,
                        "pre_sort_columns" : None,
                        "pre_sort_ascending" :True,
                        "sort_columns" : None,
                        "sort_ascending" : True,
                        "query" : None,
                        "shuffle" : False,
                        "addColumns" : None}
class ObjectProfile():
    


    def __init__(self, *args, **kargs):
        ''' An object containing processing instructions for each observable object type
            #unkeyed Arguements:
                name       -- The name of the data type (i.e. Electron, Photon, EFlowTrack, etc.)
                max_size   -- The maximum number of objects to use in training
            #keyed Arguements
                pre_sort_columns -- What columns to sort before cutting on max_size (See pandas.DataFrame.sort)
                pre_sort_ascending -- Whether each column will be sorted ascending or decending before cutting on max_size (See pandas.DataFrame.sort)
                sort_columns -- What columns to sort on after processing (See pandas.DataFrame.sort)
                sort_ascending -- Whether each column will be sorted ascending or decending after processing (See pandas.DataFrame.sort)
                query        -- A selection query string to use before truncating the data (See pands.DataFrame.query)
                shuffle     -- Whether or not to shuffle the data
                addColumns -- A dictionary with single constant floats or integers to fill an additional column in the table.
                             This column should be in observ_types if it is used with preprocessFromPandas_label_dir_pairs
        '''
        d = {}
        if(isinstance(args[0], dict)):
            d = args[0]
        elif(isinstance(args[0], str)):
            d["name"] = args[0]
            if(isinstance(args[1], int)):
                d["max_size"] = args[1]
        if(len(args) > 2):
            raise ValueError("Please explicitly name arguements with values %r" % args[2:])

        for key, value in DEFAULT_PROFILE.items():
            # print(kargs.get(key, "Nope"),d.get(key, "Nope"), value)
            setattr(self, key, kargs.get(key, d.get(key, value)))

        if (isinstance(self.pre_sort_columns, str)): self.pre_sort_columns = [self.pre_sort_columns]
        if (isinstance(self.sort_columns, str)): self.sort_columns = [self.sort_columns]

        if(self.max_size < -1):
            raise ValueError("max_size cannot be less than -1. Got %r" % self.max_size)
        if(self.addColumns != None and not isinstance(self.addColumns, dict)):
            raise ValueError("arguement addColumns must be a dictionary, but got %r" % type(self.addColumns))



        self.class_name = self.__class__.__name__



    def __str__(self):
        main_clause = 'name:%r max_size=%r ' % (self.name, self.max_size)
        sort_clause = ''
        query_clause = ''
        if(self.pre_sort_columns != None):
            sort_clause = 'pre_sort_columns=%r pre_sort_ascending=%r ' % (self.pre_sort_columns, self.pre_sort_ascending)
        if(self.sort_columns != None):
            sort_clause = 'sort_columns=%r sort_ascending=%r ' % (self.sort_columns, self.sort_ascending)
        if(self.query != None):
            query_clause = 'query=%r ' % (self.query)
        shuffle_clause = 'shuffle=%r' % self.shuffle

        return main_clause + sort_clause + query_clause + shuffle_clause
    
    __repr__ = __str__

MAX_NUM_JETS = 15
JET_PROFILE = ObjectProfile("Jet", MAX_NUM_JETS, pre_sort_columns=["PT"], pre_sort_ascending=False)
EVENT_CHARS_PROFILE = ObjectProfile("EventChars", 1)
JET_OBSERVS =  ['PT','Eta', 'Phi','Mass', 'Flavor', 'FlavorAlgo', 'FlavorPhys', 'BTag', 'BTagAlgo', 'BTagPhys','TauTag',
                'Charge', 'EhadOverEem', 'NCharged', 'NNeutrals', 'Beta', 'BetaStar', 'MeanSqDeltaR', 'PTD',
                'NSubJetsTrimmed', 'NSubJetsPruned', 'NSubJetsSoftDropped']


EVENT_CHARS = ['MET','HT','MuonMul','ElectronMul','JetMul','MaxJetPT', 'MaxLepPT']

JET_OUTPUT_OBSERVS = ['E/c', 'Px', 'Py', 'Pz'] + JET_OBSERVS

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
            if(isinstance(num_val_frame, type(None))):
                print("Skipping %r" % f)
                continue

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
    '''Gets a list of files from a directory in the filesystem and the type of data stored in it. Asserts that the directory is not empty.'''
    data_dir = os.path.expandvars(data_dir)
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
     
    
def getSizeMetaData(filename, storeType, sizesDict=None, verbose=0):
    '''Quickly resolves the number of entries in a file from metadata, making sure to update the metadata if necessary'''
    if(sizesDict == None):
        sizesDict = get_sizes_meta_dict(filename)
    modtime = os.path.getmtime(filename)
    # print(sizesDict[filename][1],modtime)
    if(not filename in sizesDict or sizesDict[filename][1] != modtime):
        num_val_frame = getNumValFrame(filename, storeType)
        if(isinstance(num_val_frame,type(None))): return None
        file_total_entries = len(num_val_frame.index)
        # print(file_total_entries)
        sizesDict[filename] = (file_total_entries,modtime)
        if (not os.path.isdir(filename)):
            split = os.path.split(filename)
            directory = "/".join(split[:-1])
        write_json_obj(sizesDict,directory,"sizesMetaData.json",verbose=verbose )
    return sizesDict[filename][0]
        
def getSizesDict(directory,verbose=0):
    '''Returns a dictionary of the number of sample points contained in each hdfStore/msgpack in a directory'''
    if(not os.path.isdir(directory)):
        split = os.path.split(directory)
        directory = "/".join(split[:-1])
    sizesDict = read_json_obj(directory, "sizesMetaData.json", verbose=verbose)
    return sizesDict
    
    
    

def getNumValFrame(filename, storeType):
    '''Finds the num_val_frame frame in a pandas file in either msg or h5 format'''
    if(storeType == "hdf5"):
        #Get the HDF Store for the file
        store = pd.HDFStore(filename)

        #Get the NumValues frame which lists the number of values for each entry
        try:
            num_val_frame = store.get('NumValues')
        except Exception as e:
            print(str(e) + " " + filename +"Please check to see if the files is corrupted. \
             Run 'll' in the folder where the file is, if it is much smaller than the others then it is corrupted. \
             If it is corrupted then delete it.")
            return None
        store.close()
        return num_val_frame
    elif(storeType == "msgpack"):
        meta_frames =  msgpack_assertMeta(filename)
        num_val_frame = meta_frames["NumValues"]
        # frames = pd.read_msgpack(f)
        # num_val_frame = frames["NumValues"]
    return num_val_frame

def _getStore(f, storeType):
    '''Helper Function - Gets the HDFStore or frames for the file and storeType'''
    frames = None
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
    return store,frames
def _getFrame(store, storeType, key, select_start, select_stop,
              samples_to_read, file_total_entries, frames):
    '''Helper Function - gets frame from its store/msgpack'''
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
    return frame


def entryGen(X,entryIndex):
    currEntry = -1
    start = 0
    for i, x in enumerate(X):
        # print(x,x[entryIndex])
        entry = x[entryIndex]
        # print(entry)
        if (entry != currEntry):
            if (entry < currEntry):
                raise ValueError("Dataframe 'Entry' column not sorted")
            if (start != i): yield int(currEntry), X[start: i]
            start = i
            currEntry = entry
    yield int(currEntry), X[start: len(X)]
    


def _groupsByEntry(f, storeType, samples_per_label, samples_to_read, file_total_entries, num_val_frame,file_start_read,object_profiles):
    '''Helper Function - produces dict keyed by object type and filled with groupBy objects w.r.t Entry'''
    store, frames = _getStore(f, storeType)

    #Get information about how many rows there are for each entry for the rows we want to skip and read
    skip_val_frame = num_val_frame[:file_start_read]
    num_val_frame = num_val_frame[file_start_read : file_start_read+samples_to_read]


    groupBys = {}
    colDict = {}
    #Loop over every profile and read the corresponding tables in the pandas file
    for profile in object_profiles:
        key = profile.name         
        #TODO: KLUDGE
        _key = key if(not key == "EventChars") else "MissingET" 
        #Where to start reading the table based on the sum of the selection start 
        select_start = int(skip_val_frame[_key].sum())
        select_stop = select_start + int(num_val_frame[_key].sum())

        frame = _getFrame(store, storeType, key, select_start, select_stop,
                          samples_to_read, file_total_entries,frames)
        columns = list(frame.columns)
        entryIndex = columns.index("Entry")
        
        #Convert to numpy array
        X = frame.values
        colDict[key] = columns
        
        #Group by Entry
        # groupBys[key] = frame.groupby(["Entry"], group_keys=True)
        groupBys[key] = {entry:x for entry,x  in entryGen(X,entryIndex)}#frame.groupby(["Entry"], group_keys=True)
        # print(key,groupBys[key].keys())
    return groupBys,colDict, store

def _applyParticleCuts(x, columns, profile, vecsize, observ_types):
    '''Helper Function - presorts, applies queries, adds columns, and makes cuts'''
    # if (profile.query != None):
    #     df = df.query(profile.query)
    # x = df.values
    # print(type(x))
    if(profile.pre_sort_columns != None):
        # Find sort_locs
        sort_locs = None
        assert not isinstance(profile.pre_sort_columns, str), "profile.pre_sort_columns improperly stored"
        if (True in [c in profile.pre_sort_columns for c in ["shuffle", "random"]]):
            np.random.shuffle(x)
        elif (not None in profile.pre_sort_columns):
            assert not False in [isinstance(s, str) or isinstance(s, unicode) for s in profile.pre_sort_columns], \
                "Type should be string got %s" % (",".join([str(type(s)) for s in profile.pre_sort_columns]))
            sort_locs = [columns.index(s) for s in profile.pre_sort_columns]
            # print(df.columns, sort_locs, )
        # Sort
        x = _sortByLocs(x, sort_locs, profile.pre_sort_ascending)#, observ_types)
        # x = _sortByColumns(x, profile.pre_sort_columns, profile.pre_sort_ascending, observ_types)
        # df = df.sort(profile.pre_sort_columns, ascending=profile.pre_sort_ascending)
    
    # x = df.values
    # Make cut, preserving only profile.max_size of top of table
    x = x[:profile.max_size]
    # Only use observable columns
    if(observ_types != None):
        x = np.take(x,[columns.index(o) for o in observ_types],axis=1)
    # print(x.shape)
    return x
    
def _addColumns(x,profile,observ_types):
    '''Helpher Function - adds columns of constants to the data'''
    for i, o in enumerate(observ_types):
        if (profile.addColumns != None and o in profile.addColumns):
            x = np.insert(x, i, profile.addColumns[o], axis=1)
    return x
    
def _sortByLocs(x,sort_locs,sort_ascending):#, observ_types,):
    # print(sort_locs)
    # print sort_locs, type(x)
    if (sort_locs != None):
        for loc in reversed(sort_locs):
            if (sort_ascending == True):
                x = x[x[:, loc].argsort()]
            else:
                x = x[x[:, loc].argsort()[::-1]]
    return x

def _padAndSort(x, profile,vecsize, observ_types):
    '''Helper Function - pads the data and sorts it'''
    if(isinstance(x, type(None))):
        #If a DataFrame does not exist for this entry then just inject zeros 
        x = np.array(np.zeros((profile.max_size, vecsize)))
    else:
        #Find sort_locs
        sort_locs = None
        assert not isinstance(profile.sort_columns, str), "profile.sort_columns improperly stored"
        if(profile.sort_columns != None):
            if(True in [c in profile.sort_columns for c in ["shuffle", "random"]]):
                np.random.shuffle(x)
            elif (not None in profile.sort_columns):
                assert not False in [isinstance(s, str) or isinstance(s, unicode) for s in profile.sort_columns], \
                    "Type should be string got %s" % (",".join([str(type(s)) for s in profile.sort_columns]))
                sort_locs = [observ_types.index(s) for s in profile.sort_columns]
        #Sort
        x = _sortByLocs(x, sort_locs, profile.sort_ascending)#, observ_types)
        #pad the array
        x = np.append(x ,np.array(np.zeros((profile.max_size - len(x), vecsize))), axis=0)
    return x    

def _gen_label_vecs(label_dir_pairs, num_labels):
    label_vecs = {}
    for i, (label, data_dir) in enumerate(label_dir_pairs):
        arr = np.zeros((num_labels,))
        arr[i] = 1
        label_vecs[label] = arr
    return label_vecs

def _initializeArrays(single_list, label_dir_pairs, num_object_profiles, samples_per_label, num_labels, maxJets=MAX_NUM_JETS):
    '''Helper Function - Generates the initial data structures for the X (data) and Y (target)'''
    
        
    if(single_list):
        X_train = [None] * (samples_per_label * num_labels)
        #global_profile = ObjectProfile("list", max_size="")
    else:
        X_train = [None] * (num_object_profiles)
        #Prefill the arrays so that we don't waste time resizing lists
        for index in range(num_object_profiles):
            X_train[index] = [None] * (samples_per_label * num_labels)
            
    y_train = [None] * (samples_per_label * num_labels)
    jets = [None] * (samples_per_label * num_labels)
    eventChars = [None] * (samples_per_label * num_labels)
    return X_train, y_train, jets, eventChars
   
def _check_Object_Profiles(object_profiles, observ_types):
    '''Helper Function - Makes sure that all ObjectProfiles are correctly formatted,
        makes formatting corrections if necessary'''
    for i,profile in enumerate(object_profiles):
        if(isinstance(profile, dict) and profile.get('class_name', None) == "ObjectProfile"):
            profile = ObjectProfile(profile)
            object_profiles[i] = profile
        if(profile.max_size == -1 or profile.max_size == None):
            raise ValueError("ObjectProfile max_sizes must be resolved before preprocessing. \
                         Please first use: utils.preprocessing.resolveProfileMaxes(object_profiles, label_dir_pairs)")
        if(profile.addColumns != None):
            for key, value in profile.addColumns.items():
                if(not key in observ_types):
                    raise ValueError("addColumn Key %r must be in observ_types" % key)
    return object_profiles

def _check_inputs(label_dir_pairs, observ_types):
    '''Helper Function - Makes sure that label_dir_pairs, and observ_types are correctly formatted'''
    labels = [x[0] for x in label_dir_pairs]
    duplicates = list(set([x for x in labels if labels.count(x) > 1]))
    if(len(duplicates) != 0):
        raise ValueError("Cannot have duplicate labels %r" % duplicates)
    if("Entry" in observ_types):
        raise ValueError("Using Entry in observ_types can result in skewed training results. Just don't.")
    

        
def preprocessFromPandas_label_dir_pairs(label_dir_pairs,start, samples_per_label, object_profiles, observ_types,
                                         single_list=False, sort_columns=None, sort_ascending=True,verbose=1):
    '''Gets training data from folders of pandas tables
    
        #Arguements:
            label_dir_pairs -- a list of tuples of the form (label, directory) where the directory contains
                                tables containing data of all the same event types.
            start             --    Where to start reading (as if all of the files are part of one long list)
            samples_per_label -- The number of samples to read for each label
            object_profiles -- A list of ObjectProfile(s) corresponding to each type of observable object and
                                its preprocessing steps. 
            observ_types    -- The column headers for the data to be read from the panadas table
            single_list -- If True all object types are joined into a single list.
            sort_columns -- If single_list the columns to sort by.
            sort_ascending -- If True sort in ascending order, false decending  
        #Returns:
            Training data with its correspoinding labels
            (X_train, Y_train)
    '''
    
    _check_inputs(label_dir_pairs, observ_types)
    #Make sure that all the profile are proper objects and have resolved max_sizes
    object_profiles = _check_Object_Profiles(object_profiles,observ_types)

    non_add_observtypes = [o for o in observ_types if not o in set.union(*[set(p.addColumns) for p in object_profiles])]
    
    vecsize = len(observ_types)
    num_labels = len(label_dir_pairs)
    
    #Build vectors in the form [1,0,0], [0,1,0], [0, 0, 1] corresponding to each label
    label_vecs = _gen_label_vecs(label_dir_pairs,num_labels)
    X_train, y_train, jets, eventChars = _initializeArrays(single_list, label_dir_pairs, len(object_profiles), samples_per_label, num_labels)
    X_train_index = 0
    
    #Loop over label dir pairs and get the file list for each directory
    y_train_start = 0
    for (label,data_dir) in label_dir_pairs:
        files, storeType = getFiles_StoreType(data_dir)
        files.sort()
        samples_read = 0
        location = 0
        
        sizesDict = get_sizes_meta_dict(data_dir)
         #Loop the files associated with the current label
        for f in files:
            file_total_entries = size_from_meta(f, storeType, sizesDict=sizesDict)#len(num_val_frame.index)
            if (file_total_entries == None):
                print("Skipping %r" % f)
                continue
            
            assert file_total_entries > 0, "num_val_frame has zero values"
            
            if(location + file_total_entries <= start):
                location += file_total_entries
                continue

            num_val_frame = getNumValFrame(f, storeType)
            assert file_total_entries == len(num_val_frame.index)
            #Determine what row to start reading the num_val table which contains
            #information about how many rows there are for each entry
            file_start_read = start-location if start > location else 0
            
            #How many rows we will read from this table each corresponds to one entry
            samples_to_read = min(samples_per_label-samples_read, file_total_entries-file_start_read)
            assert samples_to_read >= 0
            
            if(verbose >= 1): print("Reading %r samples from %r:" % (samples_to_read,f))
            
            #Grab data from tables and group by 'Entry #'
            groupBys,colDict,store = _groupsByEntry(f, storeType, samples_per_label, samples_to_read, file_total_entries,
                                                    num_val_frame, file_start_read, object_profiles + [JET_PROFILE, EVENT_CHARS_PROFILE])
                
            if(verbose >= 1): print("Values/Sample from: %r" % {p.name: p.max_size for p in object_profiles})
            
            cut_tables = [None] * (len(object_profiles)+2)
            last_time = time.clock()-1.0
            prev_entry = file_start_read
            for entry in range(file_start_read, file_start_read+samples_to_read):
                #Make a pretty progress bar in the terminal
                if(verbose >= 1):      
                    c = time.clock() 
                    if(c > last_time + .25):
                        percent = float(entry-file_start_read)/float(samples_to_read)
                        sys.stdout.write('\r')
                        sys.stdout.write("[%-20s] %r/%r  %r(Entry/sec)" % ('='*int(20*percent), entry, int(samples_to_read), 4 * (entry-prev_entry)))
                        sys.stdout.flush()
                        last_time = c
                        prev_entry = entry
                
                jet_indx = len(object_profiles)
                chars_indx = len(object_profiles)+1 
                #Apply Cuts for particles
                for index, profile in enumerate(object_profiles + [JET_PROFILE, EVENT_CHARS_PROFILE]) :
                        #print(groupBys.keys())
                        groups = groupBys[profile.name]
                        columns = colDict[profile.name]
                        obtypes = non_add_observtypes if index < jet_indx else [x for x in colDict[profile.name] if x != "Entry"]
                        # print(entry,entry in groups)
                        if(entry in groups):
                            x = _applyParticleCuts(groups[entry], columns, profile, vecsize, obtypes)
                            # x = _applyCuts(groups.get_group(entry), profile, vecsize, non_add_observtypes)
                            cut_tables[index] = _addColumns(x, profile, observ_types)
                        else:
                            cut_tables[index] = None
                
                if(single_list):
                    #joint the data, but ommit the jet and event char collections
                    x = np.concatenate([c for c in cut_tables[:-2] if c != None], axis=0)
                    list_profile = ObjectProfile("single_list",
                                                sum([profile.max_size for profile in object_profiles]),
                                                sort_columns=sort_columns,
                                                sort_ascending=sort_ascending)    
                                                    
                    x  = _padAndSort(x,list_profile,vecsize,observ_types)
                    X_train[X_train_index + entry - file_start_read] = x
                else:
                    for index, profile in enumerate(object_profiles):
                        arr = X_train[index]
                        c = cut_tables[index]
                        x  = _padAndSort(c,profile, vecsize,observ_types)
                        arr[X_train_index + entry - file_start_read] = x
                        
                jet_df = cut_tables[jet_indx]
                jet_obervs = [x for x in colDict['Jet'] if x != "Entry"]
                # print(_padAndSort(jet_df,JETS_PROFILE,len(JET_OUTPUT_OBSERVS),JET_OUTPUT_OBSERVS))
                # print("JET_DF",jet_df)
                jets[X_train_index + entry - file_start_read] = _padAndSort(jet_df, JET_PROFILE, len(jet_obervs), jet_obervs)

                chars_obervs = [x for x in colDict['EventChars'] if x != "Entry"]
                chars_df = cut_tables[chars_indx]
                eventChars[X_train_index + entry - file_start_read] = _padAndSort(chars_df, EVENT_CHARS_PROFILE,len(chars_obervs), chars_obervs)

                
                        
                
                
            
            X_train_index += samples_to_read
            
            #Free this (probably not necessary)
            num_val_frame = None
            if(storeType == "hdf5"):
                store.close()
            location     += file_total_entries
            samples_read += samples_to_read
            if(verbose >= 1): print("*Read %r Samples of %r in range(%r, %r)" % (samples_read, samples_per_label, start, samples_per_label+start))
            if(samples_read >= samples_per_label):
                if(verbose >= 1): print('-' * 50)
                assert samples_read == samples_per_label
                break
        if(samples_read != samples_per_label):
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
    if(single_list):
        X_train = np.array(X_train)[indices]
    else:
        for index in range(len(X_train)):
            X_train[index] = np.array(X_train[index])[indices]

    y_train = y_train[indices]
    return X_train, y_train, jets, eventChars
    

def getGensDefaultFormat(archive_dir, splits, length, object_profiles, label_dir_pairs, observ_types, single_list=False, sort_columns=None, sort_ascending=True, batch_size=100, megabytes=500, data_keys=["X","Y"],dp_data_keys=None, verbose=1):
    '''Creates a set of DataProcedures that return generators and their corresponding lengths. Each generator consists of a list DataProcedures that preprocess data
        from a set of label_dir_pairs in a given range. The size of the archived files for each DP is set by 'megabytes' so that each one is not too big. Each generator
        reads a number of samples per label type set by 'splits' and 'length', and feeds data in batches of 'batch_size' into training.
        #Arguments:
            archive_dir -- The archive directory that the DataProcedures of each generator will archive their information in.
            splits -- a list of either integers or floats between 0 and 1 (or both). Each entry in 'splits' designates a generator. If an Integer is given then a generator
                      is built with the number of samples per label designated by that integer (static). If a float is given then the number of samples per label is computed as a 
                      fraction of the argument 'length' minus the sum of the integer entries (ratio). Float (ratio) entries in splits must add up to 1.0.
            length -- The total number of samples per label to be split among the float (ratio) values of 'splits' plus the Integer (static) values. In other words the total number
                        of samples per value to be used by all of the generators built by this function. Does not matter if all splits are Integers (static).
            object_profiles -- A list of ObjectProfiles (see CMS_Deep_Learning.utils.preprocessing.ObjectProfile). These determine how the final preprocessed inputs will be
                            preprocessed. In the collections will appear in the order the object_profiles are given.
            label_dir_pairs -- A list of tuples where the first entry is a label and the second is the name of a directory containing pandas files (either msg or h5 format) corresponding 
                            to that label.
            observ_types -- A list of the types of observables to be used in the final preprocessed files.
            batch_size -- How many samples to feed into training at a time. 
            megabytes -- Determines how large in MB a DataProcedure archive should be. A smaller number means less data in memory at a time as each generator is used, but shorter more frequent
                        disk reads. 
            verbose -- Determines whether or not information is printed out as the generators are formed and as they are used. (TODO: the implementation of this might need some work, the specifics
                        of how this information is passed along the the DPs and their dependant functions might not be implemented correctly at the moment, leading to printouts even if verbose=0)
        #Returns (all_dps, all_datasets)
            all_dps -- A list of DataProcedures, this can be passed to CMS_Deep_Learning.utils.batch.batchAssertArchived to make sure that all the DPs are archived before proceeding to training
            all_datasets -- A list like [(generator1,num_samples1), (generator2, num_samples2), ... , max_q_size], where max_q_size designates how large the keras generator queue should be so that
                            each generator starts reading the next DP in the archive as it starts outputing data from the previous one.  
        '''
    if(isinstance(dp_data_keys, type(None))): dp_data_keys = data_keys
    assert isinstance(object_profiles, list)
    assert isinstance(label_dir_pairs, list)
    assert isinstance(observ_types, list)
    stride = strideFromTargetSize(object_profiles, label_dir_pairs, observ_types, megabytes=megabytes)
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
                                        observ_types,
                                        single_list=single_list,
                                        sort_columns=sort_columns,
                                        sort_ascending=sort_ascending,
                                        verbose=verbose,
                                        data_keys=dp_data_keys)
        gen_DP = DataProcedure(archive_dir, False, gen_from_data, [dps, batch_size], {'threading':False, 'verbose':verbose}, data_keys=data_keys)
        num_samples = len(label_dir_pairs)*s[1]
        all_datasets += [(gen_DP, num_samples)]
        all_dps += dps
    #Calculate a good max_q_size and add it to the all_datasets list
    all_datasets += [max(np.ceil(stride/float(batch_size)), 1)]
    return (all_dps,all_datasets)



       
            

def strideFromTargetSize(object_profiles, num_labels, observ_types, megabytes=100):
    '''Computes how large a stride is required to get DPs with archives of size megabytes'''
    if(isinstance(num_labels, list)): num_labels = len(num_labels)
    megabytes_per_sample = sum(o.max_size for o in object_profiles) * len(observ_types) * 24.0 / (1000.0 * 1000.0)
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
                    print('File: ' + f + ' may be corrupted:' + os.linesep + 
                                    'Requested keys: ' + str(keys) + os.linesep + 
                                    'But found keys: ' + str(store.keys()) )
                    print('Skipping %r' % f)
                    continue
                
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
    return min(label_totals.values())

def start_num_fromSplits(splits, length):
    '''Takes in a tuple of splits and a length and returns a list of tuples with the starts and number of
        samples for each split'''
    if(True in [x < 0.0 for x in splits]):
        raise ValueError("Splits cannot be negative %r" % str(splits)) 
    are_static_vals = [(True if int(x) > 0 else False) for x in splits]
    if(True in are_static_vals):
        ratios =  [s for s, a in zip(splits, are_static_vals) if(not a)]
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



def procsFrom_label_dir_pairs(start, samples_per_label, stride, archive_dir,label_dir_pairs, object_profiles, observ_types, single_list=False, sort_columns=None, sort_ascending=True, data_keys=["X", 'Y'], verbose=1):
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
                [label_dir_pairs,
                proc_start,
                proc_num,
                object_profiles,
                observ_types],
                {
                'single_list':single_list,
                'sort_columns':sort_columns,
                'sort_ascending':sort_ascending,
                'verbose':verbose
                },
                data_keys = data_keys
            )
        procs.append(dp)
        #print(proc_start, samples_per_label, stride)
        if(verbose >= 1):
            num_labels = len(label_dir_pairs)
            print("   From %r labels in range(%r,%r) for %rx%r = %r Samples"
                     % (num_labels,proc_start, proc_start+proc_num, num_labels,proc_num,num_labels*proc_num))
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
        self.X, self.Y = self.proc.get_data()
        return


def genFrom_label_dir_pairs(start, samples_per_label, stride, batch_size, archive_dir,label_dir_pairs, object_profiles, observ_types, verbose=1):
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
    dps = procsFrom_label_dir_pairs(start,
                                    samples_per_label,
                                    stride,
                                    archive_dir,
                                    label_dir_pairs,
                                    object_profiles,
                                    observ_types,
                                    verbose=verbose)
    gen = gen_from_data(dps, batch_size, threading = False, verbose=verbose)
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

