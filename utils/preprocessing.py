import numpy as np
import pandas as pd
import glob


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
        files = glob.glob(data_dir+"*.h5")
        files.sort()
        
         #Loop the files associated with the current label
        for f in files:
          
            #Get the HDF Store for the file
            store = pd.HDFStore(f)

            #Get the NumValues frame which lists the number of values for each entry
            try:
                num_val_frame = store.get('/NumValues')
            except KeyError as e:
                raise KeyError(str(e) + " " + f)

            for profile in unresolved:
                maxes[profile.name] = max(num_val_frame[profile.name].max(), maxes[profile.name])
    
    for profile in unresolved:
        profile.max_size = int(np.ceil(maxes[profile.name] * padding_ratio))

def label_dir_pairs_args_decoder(*args, **kargs):
    '''Decodes the arguments to preprocessFromPandas_label_dir_pairs so that the ObjectProfile(s) are 
        properly reconstituted'''
    #print(args)
    profiles = args[3]
    out = []
    for profile in profiles:
        # print(profile)
        out.append(ObjectProfile(profile['name'],
                                    profile.get('max_size', 100),
                                    profile.get('sort_columns', None),
                                    profile.get('sort_ascending', True),
                                    profile.get('query', None),
                                    profile.get('shuffle', False)))
    args = list(args)
    args[3] = out
    args = tuple(args)
    return (args, kargs)

def padItem(x,max_size, vecsize, shuffle=False):
    '''Pads a numpy array up to MAX_SIZE or trucates it down to MAX_SIZE. If shuffle==True,
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
                                its preprocessing steps
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

    #Make sure that all the profiles have resolved max_sizes
    for profile in object_profiles:
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
        files = glob.glob(data_dir+"*.h5")
        files.sort()
        samples_read = 0
        location = 0
        
         #Loop the files associated with the current label
        for f in files:
          
            #Get the HDF Store for the file
            store = pd.HDFStore(f)

            #Get the NumValues frame which lists the number of values for each entry
            try:
                num_val_frame = store.get('/NumValues')
            except KeyError as e:
                raise KeyError(str(e) + " " + f)

            file_total_entries = len(num_val_frame.index)
            
            #print(start)
            if(location + file_total_entries <= start):
                location += file_total_entries
                #print(location, file_total_entries)
                continue
            
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
                
                #If we are reading all the samples use get since it might be faster
                #TODO: check if it is actually faster
                if(samples_to_read == file_total_entries):
                    frame = store.get('/'+key)
                else:
                    frame = store.select('/'+key, start=select_start, stop=select_stop)
               
                
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
            store.close()
            location     += samples_to_read
            samples_read += samples_to_read
            if(verbose >= 1): print("*Read %r Samples of %r in range(%r, %r)" % (samples_read, samples_per_label, start, samples_per_label+start))
            if(samples_read >= samples_per_label):
                print('-' * 50)
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
    for index in range(len(X_train)):
        X_train[index] = np.array(X_train[index])[indices]

    y_train = y_train[indices]
    return X_train, y_train