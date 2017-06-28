import sys,os
if __package__ is None:
    #sys.path.append(os.path.realpath("../"))
    sys.path.append(os.path.realpath(__file__+"/../../../"))


import time, math,re,h5py
import argparse
from multiprocessing import Process
from time import sleep

PARTICLE_OBSERVS = ['Energy', 'Px', 'Py', 'Pz', 'Pt', 'Eta', 'Phi', 'Charge',
                    'ChPFIso', 'GammaPFIso', 'NeuPFIso',
                    'isChHad', 'isEle', 'isGamma', 'isMu', 'isNeuHad',
                    'vtxX', 'vtxY', 'vtxZ']
HLF_OBSERVS = ['HT', 'MET', 'MT', 'PhiMET', 'bJets', 'nJets']
DEFAULT_RPE = {"Particles": 801, "HLF": 1}
DEFAULT_OBSERVS = {"Particles": PARTICLE_OBSERVS, "HLF": HLF_OBSERVS }

import pandas as pd
import numpy as np


def get_from_pandas(f, rows_per_event, file_start_read, samples_to_read, file_total_events, observ_types):
    '''Helper Function - produces dict keyed by object type and filled with groupBy objects w.r.t EvtId'''
    store = pd.HDFStore(f)

    values = {}
    for key, rpe in rows_per_event.items():
        # Where to start reading the table based on the sum of the selection start 
        select_start = file_start_read * rpe
        select_stop = select_start + samples_to_read * rpe

        if (samples_to_read == file_total_events):
            frame = store.get('/' + key)
        else:
            frame = store.select('/' + key, start=select_start, stop=select_stop)

        columns = list(frame.columns)

        x = frame.values
        if (observ_types != None):
            x = np.take(x, [columns.index(o) for o in observ_types[key]], axis=-1)
        if (rpe > 1):
            n_rows, n_columns = x.shape
            x = x.reshape((n_rows / rpe, rpe, n_columns))

        values[key] = x
    return values, store


def _gen_label_vecs(data_dirs):
    num_labels = len(data_dirs)
    label_vecs = {}
    for i, data_dir in enumerate(data_dirs):
        arr = np.zeros((num_labels,))
        arr[i] = 1
        label_vecs[data_dir] = arr
    return label_vecs


def _initializeArrays(data_dirs, samples_per_class):
    '''Helper Function - Generates the initial data structures for the X (data) and Y (target)'''
    num_classes = len(data_dirs)
    X_train = [None] * (samples_per_class * num_classes)
    y_train = [None] * (samples_per_class * num_classes)
    HLF_train = [None] * (samples_per_class * num_classes)
    return X_train, y_train, HLF_train


def getSizesDict(directory, verbose=0):
    '''Returns a dictionary of the number of sample points contained in each hdfStore/msgpack in a directory'''
    from CMS_Deep_Learning.storage.archiving import read_json_obj
    if (not os.path.isdir(directory)):
        split = os.path.split(directory)
        directory = "/".join(split[:-1])
    sizesDict = read_json_obj(directory, "sizesMetaData.json", verbose=verbose)
    return sizesDict


def _readNumSamples(file_path):
    f = h5py.File(file_path, 'r')
    out = f["HLF"]['axis1'].len()
    f.close()
    return out


def _check_inputs(data_dirs, observ_types):
    '''Helper Function - Makes sure that data_dirs, and observ_types are correctly formatted'''
    if (len(set(data_dirs)) != len(data_dirs)):
        raise ValueError("Cannot have duplicate directories %r" % data_dirs)
    for x in observ_types.values():
        if ("EvtId" in x):
            raise ValueError("Using EvtId in observ_types can result in skewed training results. Just don't.")


def getSizeMetaData(filename, sizesDict=None, verbose=0):
    from CMS_Deep_Learning.storage.archiving import write_json_obj
    '''Quickly resolves the number of entries in a file from metadata, making sure to update the metadata if necessary'''
    if (sizesDict == None):
        sizesDict = getSizesDict(filename)
    modtime = os.path.getmtime(filename)
    if (not filename in sizesDict or sizesDict[filename][1] != modtime):
        file_total_events = _readNumSamples(filename)
        sizesDict[filename] = (file_total_events, modtime)
        if (not os.path.isdir(filename)):
            split = os.path.split(filename)
            directory = "/".join(split[:-1])
        write_json_obj(sizesDict, directory, "sizesMetaData.json", verbose=verbose)
    return sizesDict[filename][0]


def maxLepPtEtaPhi(X, locs):
    for x in X:
        if (x[locs['isEle']] or x[locs["isMu"]]):
            return x[locs['Pt']], x[locs['Eta']], x[locs['Phi']]


def MaxLepDeltaPhi(X, locs, mlpep=None):
    maxLepPt, maxLepEta, maxLepPhi = maxLepPtEtaPhi(X, locs) if isinstance(mlpep, type(None)) else mlpep
    out = maxLepPhi - X[:, locs["Phi"]]

    tooLarge = -2.0 * math.pi * (out > math.pi)
    tooSmall = 2.0 * math.pi * (out < -math.pi)
    out = out + tooLarge + tooSmall
    return out


def MaxLepDeltaEta(X, locs, mlpep=None):
    maxLepPt, maxLepEta, maxLepPhi = maxLepPtEtaPhi(X, locs) if isinstance(mlpep, type(None)) else mlpep
    return maxLepEta - X[:, locs["Eta"]]


def MaxLepDeltaR(X, locs, mlpep=None):
    mlpep = maxLepPtEtaPhi(X, locs) if isinstance(mlpep, type(None)) else mlpep
    # print(mlpep)
    return np.sqrt(MaxLepDeltaPhi(X, locs, mlpep) ** 2 + MaxLepDeltaEta(X, locs, mlpep) ** 2)


def MaxLepKt(X, locs):
    mlpep = maxLepPtEtaPhi(X, locs)
    maxLepPt, maxLepEta, maxLepPhi = mlpep
    return np.minimum(X[:, locs["Pt"]] ** 2, maxLepPt ** 2) * MaxLepDeltaR(X, locs, mlpep) ** 2


def MaxLepAntiKt(X, locs):
    mlpep = maxLepPtEtaPhi(X, locs)
    maxLepPt, maxLepEta, maxLepPhi = mlpep
    return np.minimum(X[:, locs["Pt"]] ** -2, maxLepPt ** -2) * MaxLepDeltaR(X, locs, mlpep) ** 2


SORT_METRICS = {f.__name__: f for f in
                [MaxLepDeltaPhi, MaxLepDeltaEta, MaxLepDeltaR, MaxLepKt, MaxLepAntiKt]}


def assertZerosBack(sort_slice, x, locs, sort_ascending):
    from numpy import inf
    sort_slice[np.all(x == 0.0, axis=1)] = inf if sort_ascending else -inf
    return sort_slice


def resolveMetric(s, locs, sort_ascending):
    if s in SORT_METRICS:
        return lambda x: assertZerosBack(SORT_METRICS[s](x, locs), x, locs, sort_ascending)
    else:
        raise ValueError("Unrecognized sorting metric %r" % s)


def _sortBy(x, sorts, sort_ascending):  # , observ_types,):
    if (sorts != None):
        for s in reversed(sorts):
            if (isinstance(s, int)):
                sort_slice = x[:, s]
            else:
                sort_slice = s(x)
            if (sort_ascending == True):
                x = x[sort_slice.argsort()]
            else:
                x = x[sort_slice.argsort()[::-1]]
    return x


def sort_numpy(x, sort_columns, sort_ascending, observ_types):
    '''Helper Function - pads the data and sorts it'''
    sort_locs = None
    assert not isinstance(sort_columns, str), "sort_columns improperly stored"
    if (sort_columns != None):
        if (True in [c in sort_columns for c in ["shuffle", "random"]]):
            np.random.shuffle(x)
        elif (not None in sort_columns):
            assert not False in [isinstance(s, str) or isinstance(s, unicode) for s in sort_columns], \
                "Type should be string got %s" % (",".join([str(type(s)) for s in sort_columns]))
            locs = {t: s for s, t in enumerate(observ_types)}
            sorts = [locs[s] if s in observ_types else resolveMetric(s, locs, sort_ascending)
                     for s in sort_columns]
            # KLUGE FIX
            x[x[:, locs["Energy"]] == 0] = 0.0
            # Sort
            x = _sortBy(x, sorts, sort_ascending)  # , observ_types)

    return x


# d, store = get_from_pandas("/bigdata/shared/Delphes/REDUCED_IsoLep/wjets_lepFilter_13TeV/wjets_lepFilter_1070.h5",
#                            DEFAULT_RPE
#                            , 0, 100, 101,
#                            DEFAULT_OBSERVS)

import glob


def pandas_to_numpy(data_dirs, start, samples_per_class,
                    observ_types=DEFAULT_OBSERVS, sort_columns=None, sort_ascending=True, verbose=1):
    '''Builds a trainable (particle level) sorted and (event level) shuffled numpy array from directories of pandas .h5 files.
        #Arguements:
            :param data_dirs: A list of pandas directories containing pandas .h5 files, tuples of ('label','dir'),
                                or dictionary with .values() equal to such a list. The order indicates which
                                files correspond to which output (i.e. the first directory corresponds to
                                [1,0,...,0] and the second to [0,1,...,0], etc.). For dictionaries the 
                                order defaults to the alphabetical order of the directory names.
            :param start:        Where to start reading (as if all of the files in a given directory are part of one long list)
            :param samples_per_class: The number of samples to read for each label. Every directory must have enough data starting
                                from 'start'.
            :param observ_types: The column headers for the data to be read from the panadas table. Also indicated the order of the columns.
            :param sort_columns: The columns to sort by, or special quantities including [MaxLepDeltaPhi,
                                MaxLepDeltaEta,MaxLepDeltaR,MaxLepKt,MaxLepAntiKt]
            :param sort_ascending: If True sort in ascending order, false decending  
        #Returns:
            particle training data with its correspoinding labels and High Level Features (HLF)
            (X_train, Y_train, HFL_train)
    '''
    if (isinstance(data_dirs, dict)): data_dirs = sorted(data_dirs.values(), key=lambda x: x.join(x.split("/")[::-1]))
    if (isinstance(data_dirs[0], tuple)): data_dirs = [x[1] for x in data_dirs]
    _check_inputs(data_dirs, observ_types)

    label_vecs = _gen_label_vecs(data_dirs)
    X_train, y_train, HLF_train = _initializeArrays(data_dirs, samples_per_class)
    X_train_index = 0

    y_train_start = 0
    for data_dir in data_dirs:
        files = glob.glob(os.path.abspath(data_dir) + "/*.h5")
        files.sort()
        samples_read, location = 0, 0

        sizesDict = getSizesDict(data_dir)

        last_time = time.clock() - 1.0
        #print("FILES",files)
        # Loop the files associated with the current label
        for f in files:
            file_total_events = getSizeMetaData(f, sizesDict=sizesDict)  # len(num_val_frame.index)
            if (file_total_events == None):
                print("Skipping %r" % f)
                continue

            assert file_total_events > 0, "num_val_frame has zero values"

            if (location + file_total_events <= start):
                location += file_total_events
                continue

            # Determine what row to start reading the num_val table which contains
            # information about how many rows there are for each entry
            file_start_read = start - location if start > location else 0

            # How many rows we will read from this table each corresponds to one entry
            samples_to_read = min(samples_per_class - samples_read, file_total_events - file_start_read)
            assert samples_to_read >= 0

            d, store = get_from_pandas(f, rows_per_event=DEFAULT_RPE,
                                       file_start_read=file_start_read,
                                       samples_to_read=samples_to_read,
                                       file_total_events=file_total_events,
                                       observ_types=observ_types)
            Particles, HLF = d["Particles"], d["HLF"]

            for s, (particles, hlf) in enumerate(zip(Particles, HLF)):
                # ----------pretty progress bar---------------
                if (verbose >= 1):
                    c = time.clock()
                    if (c > last_time + .25):
                        prog = X_train_index + s
                        percent = float(prog) / (samples_per_class * len(data_dirs))
                        sys.stdout.write('\r')
                        sys.stdout.write("[%-20s] %r/%r  %r(Event/sec)" % ('=' * int(20 * percent), prog,
                                                                           int(samples_per_class) * len(data_dirs),
                                                                           4 * prog))
                        sys.stdout.flush()
                        last_time = c
                # ------------------------------------------
                particles = sort_numpy(particles, sort_columns, sort_ascending, observ_types["Particles"])

                X_train[X_train_index + s] = particles
                HLF_train[X_train_index + s] = hlf

            X_train_index += samples_to_read

            store.close()
            location += file_total_events
            samples_read += samples_to_read
            if (samples_read >= samples_per_class):
                assert samples_read == samples_per_class
                break
        if (samples_read != samples_per_class):
            raise IOError(
                "Not enough data in %r to read in range(%r, %r)" % (data_dir, start, samples_per_class + start))

        # Generate the target data as vectors like [1,0,0], [0,1,0], [0,0,1]
        for i in range(samples_per_class):
            y_train[y_train_start + i] = label_vecs[data_dir]
        y_train_start += samples_per_class

    # Turn everything into numpy arrays and shuffle them just in case.
    # Although, we probably don't need to shuffle since keras shuffles by default.
    y_train = np.array(y_train)

    indices = np.arange(len(y_train))
    np.random.shuffle(indices)

    X_train = np.array(X_train)[indices]
    HLF_train = np.array(HLF_train)[indices]
    y_train = y_train[indices]

    return X_train, y_train, HLF_train

def store():
    pass

def splitsFromVal(v,n_samples):
    if(v == 0.0): return (n_samples,)
    if(v < 1.0):
        return (1.0-v,v)
    elif(isinstance(v, int)):
        return (n_samples-v, v)
    else:
        raise ValueError("Cannot make fractional validation samples %r" % v)
        
    

def set_range_from_splits(splits, length):
    '''Takes in a tuple of splits and a length and returns a list of tuples with the starts and number of
        samples for each split'''
    if (True in [x < 0.0 for x in splits]):
        raise ValueError("Splits cannot be negative %r" % str(splits))
    are_static_vals = [(True if int(x) > 0 else False) for x in splits]
    if (True in are_static_vals):
        ratios = [s for s, a in zip(splits, are_static_vals) if (not a)]
        static_vals = [s for s, a in zip(splits, are_static_vals) if (a)]
        s = sum(static_vals)
        if (s > length):
            raise ValueError("Static values have sum %r exceeding given length %r" % (s, length))
        length -= s
    else:
        ratios = splits
    print(ratios)
    if (len(ratios) > 0 and np.isclose(sum(ratios), 1.0) == False):
        raise ValueError("Sum of splits %r must equal 1.0" % sum(ratios))

    nums = [int(s) if (a) else int(s * length) for s, a in zip(splits, are_static_vals)]
    out = []
    start = 0
    for n in nums:
        out.append((start, n))
        start += n
    return out


def strideFromTargetSize(rows_per_event, observ_types, num_classes, megabytes=100):
    '''Computes how large a stride is required to build a file with size megabytes'''
    megabytes_per_sample = sum([rows_per_event[key]*observ_types for key, in rows_per_event]) * 24.0 / (1000.0 * 1000.0)
    return int(megabytes/megabytes_per_sample)


def _checkDir(dir):
    out_dir = os.path.abspath(dir)
    if (not(os.path.exists(out_dir) and os.path.isdir(out_dir))):
        raise IOError("no such directory %r" % out_dir)
    return out_dir

def runAndStore():
    pass    

def main(argv):
    parser = argparse.ArgumentParser(
        description='Convert ROOT data to numpy arrays stored as HDF5 for Machine Learning use.')
    parser.add_argument('sources', type=str, nargs='+')
    parser.add_argument('-o', '--output_dir', type=str, dest='output_dir', required=True)
    parser.add_argument('-n', '--num_samples', metavar='N', type=int, dest='num_samples', required=True)
    parser.add_argument('-p', '--num_processes', metavar='N', type=int, dest='num_processes', default=1,
                        help='How many processes to use concurrently.')
    parser.add_argument('-s', '--size', metavar='N', type=str, default='1000',
                        help='The number of samples per file to use. Can also indicate a target size in MB. An integer, or integer followed by MB (i.e 100MB)')
    parser.add_argument('-f', '--force', action='store_true',  default=False,
                        help='if true clean the output directory before starting, else throw an error')
    parser.add_argument('-v', '--validation_split', type=float, default=0.0, dest='v_split',
                        help='the proportion of samples that should be reserved for validation, or the number of samples_per_class that should be used')
    parser.add_argument('--sort_on', type=str, default=None, dest='sort_on',
                        help='The column or special value [MaxLepDeltaPhi,MaxLepDeltaEta,MaxLepDeltaR,MaxLepKt,MaxLepAntiKt] to sort on')
    parser.add_argument('--sort_ascending', action='store_true', default=False, dest='sort_ascending',
                        help='To sort ascending')
    parser.add_argument('--sort_descending', action='store_false', default=False, dest='sort_ascending',
                        help='To sort descending')
    
    # parser.add_argument('--clean_pandas', action='store_true', default=False)
    # parser.add_argument('--clean_archive', action='store_true', default=False)
    # parser.add_argument('--skip_parse', action='store_true', default=False)

    # parser.add_argument('--photon_max', metavar='N', type=int, default=DEFAULT_PHOTON_MAX)
    # parser.add_argument('--neutral_max', metavar='N', type=int, default=DEFAULT_NEUTRAL_MAX)
    # parser.add_argument('--charged_max', metavar='N', type=int, default=DEFAULT_CHARGED_MAX)
    # parser.print_usage()
    
    try:
        args = parser.parse_args(argv)
    except Exception:
        parser.print_usage()
    
    sources = [_checkDir(s) for s in args.sources]
    
	
    if ("MB" in args.size):
        megabytes = int(re.search(r'\d+', args.size).group())
        stride = strideFromTargetSize(rows_per_event=DEFAULT_RPE, observ_types=DEFAULT_OBSERVS, megabytes=megabytes)
    else:
        stride = int(re.search(r'\d+', args.size).group())
    print("STRIDE",stride)
    print(splitsFromVal(args.v_split,args.num_samples))
    SNs = set_range_from_splits(splitsFromVal(args.v_split, args.num_samples), args.num_samples)
    
    if(not os.path.exists(args.output_dir)):
        os.mkdir(args.output_dir)
    jobs = []
    for i,sn in enumerate(SNs):
        folder = os.path.abspath(args.output_dir) + ("/train" if(i==0) else '/val')
        if(len(glob.glob(folder+"/*.h5")) != 0):
            if(not args.force):
                raise IOError("directory %r is not empty use -f or --force to force overwrite" % folder)
        
        if(not os.path.exists(folder)):
            os.mkdir(folder)
        print(sn[1])
        order_of_mag = max(int(math.log(sn[1]/stride+1, 10)),3)
        end = sn[0] +sn[1] 
        for j,start in enumerate(range(sn[0], end, stride)):
            samples_per_class= min(stride, end - start)
            kargs = {'data_dirs':args.sources, 'start':start, 'samples_per_class':samples_per_class,
                    'observ_types':DEFAULT_OBSERVS, 'sort_columns':[args.sort_on], 'sort_ascending':args.sort_ascending, 'verbose':1}
            dest = os.path.abspath(folder + ("/%0" + str(order_of_mag) + "d.h5") % j)
            jobs.append((kargs,dest))
    
    def f(jobs):
        for kargs,dest in jobs:
            x = pandas_to_numpy(**kargs)
            #print(x.shape)
            h5f = h5py.File(dest, 'w')
            for D, key in zip(x, ["Particles","Labels","HLF"]):
                h5f.create_dataset(key, data=D)
            h5f.close()
            print("DONE")
    
    num_processes = args.num_processes
    processes = []
    splits = [jobs[i::num_processes] for i in range(num_processes)]
    print(splits)
    samples_per_process = np.ceil(args.num_samples / num_processes)
    # np.array_split(jobs, num_processes)
    for i, sublist in enumerate(splits[1:]):
        print("Thread %r Started" % i)
        p = Process(target=f, args=sublist)
        processes.append(p)
        p.start()
        sleep(.001)
    try:
        #print("SPLIT", splits[0])
        f(splits[0])
    except Exception as e:
        for p in processes:
            p.terminate()
        raise e
    for p in processes:
        p.join()
        
    #print(sources)
    #print(args.clean_archive)
    #print(args.clean_pandas)
    #print(args.output_dir)
    #print(args.num_samples)

    # if (not args.skip_parse):

    # applyPreprocessing(sources,
    #                    args.num_samples,
    #                    args.output_dir,
    #                    args.num_processes,
    #                    clean_pandas=args.clean_pandas,
    #                    clean_archive=args.clean_archive,
    #                    size=args.size)

if __name__ == "__main__":
   main(sys.argv[1:])
