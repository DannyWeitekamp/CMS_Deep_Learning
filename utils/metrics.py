import types
from keras.metrics import *
from .archiving import DataProcedure, KerasTrial
from .plot import plotBins

def accVsEventChar(model,
                   data,
                   char,
                   observ,
                   objects,
                   num_samples=None,
                   char2=None,
                   bins=20,
                   observable_ordering=['E/c', 'Px', 'Py', 'Pz', 'PT_ET','Eta', 'Phi', 'Charge', 'X', 'Y', 'Z', 
                     'Dxy', 'Ehad', 'Eem', 'MuIso', 'EleIso', 'ChHadIso','NeuHadIso','GammaIso'],
                   object_ordering=["Electron", "MuonTight", "Photon", "MissingET","EFlowPhoton", "EFlowNeutralHadron", "EFlowTrack"],
                   equalBins=False,
                   custom_objects=None,
                   plot=False):
    '''Computes event features and and returns binned data about the accuracy of a model against those features. Also computes the standard error for each bin.
        #Arguements:
            model -- The model being tested, or a KerasTrial containing a valid model.
            data  -- A generator, DataProcedure, or tuple pair (X,Y) containing the data to be run through the model. If a generator or DataProcedure
                     containing a generator is given the num_samples must be set. If model is a KerasTrial this can be set to None, and the validation
                     set will be found from the archive (or computed) and used in place of data.
            char  -- Any numpy function that reduces data along an axis, (i.e np.sum, np.avg, np.std). This is the 1st reduction of the characteristics
                     reducing the data within each object type of a sample.
            observ -- The observable to be reduced (i.e PT_ET, E/c, Phi).
            objects -- What objects should be included in the characteristic computation.
            num_samples -- The number of samples to be read from a generator dat input.
            char2 -- Defaults to the same as char. A numpy function that reduces data along an axis. In this case to reduce between objects.
            bins -- The number of bins to use in the analysis.
            observable_ordering -- A list of the observables in each sample ordered as they are in the sample. It is IMPORTANT that this matches the observables
                                    in the sample, otherwise the "observ" argument will not select the intended column in the data.
            object_ordering -- A list of the possible objects in the data ordered as they are in the sample. This corresponds to the ordering of the ObjectProfiles
                                when the data was created. If this argument does not match the data then the wrong objects will be selected for analysis.
            equalBins -- True/False, Defualt False. If True, will try to put an equal number of samples in each bin. This should probably be left False or else the bins
                            will be very unusual, varying significantly in their domain.
            custom_objects -- A dictionary keyed by names containing the classes of any model components not used in the standard Keras library.
            plot -- If True plot the bins automatically.
        #Returns:
            A list of dictionaries each containing information about a bin. The output of this can be plotted with CMS_SURF_2016
            '''
    if(isinstance(model, KerasTrial)):
        trial = model
        model = trial.compile(loadweights=True,custom_objects=custom_objects)
        if(data == None):
            val_proc = trial.val_procedure if isinstance(trial.val_procedure, str) else trial.val_procedure[0]
            if(num_samples == None): num_samples = trial.nb_val_samples
            p = DataProcedure.from_json(trial.archive_dir,val_proc)
            data = p.getData()
    predictions = None
    characteristics = None
    if(isinstance(data, DataProcedure)):
        data = data.getData()
    if(isinstance(data, tuple)):
        data = [data]
    elif(isinstance(data, types.GeneratorType)):
        if(num_samples == None): raise ValueError("Must provide argument num_samples if argument 'data' is a generator")
    else:
        raise TypeError("Arguement 'data' is not DataProcedure, Generator or (X,Y) instead got,%r" % type(data))
    if(num_samples == None):
        num_samples = 0
        for out in data:
            if(isinstance(out,DataProcedure)):
                X,Y = out.getData()
            else:
                X,Y = out
            #if(not isinstance(X, list)): Y = [Y]
            if(not isinstance(Y, list)): Y = [Y]
            num_samples += Y[0].shape[0]
    predictions = [None] * num_samples
    characteristics = [None] * num_samples
    y_vals = [None] * num_samples
    # print(possible_objects)
    # print(objects)
    if(not isinstance(objects, list)): objects = [objects]
    objects = [o if isinstance(o, int) else object_ordering.index(o) for o in objects]
    observ = observ if isinstance(observ, int) else observable_ordering.index(observ)
    if(char2 == None): char2 = char
    num_read = 0
    global_batch_size = None
    for i, out in enumerate(data):
        if(isinstance(out,DataProcedure)):
            X,Y = out.getData()
        else:
            X,Y = out
        batch_size = min(Y[0].shape[0], num_samples-num_read)

        if(global_batch_size == None): global_batch_size = batch_size
        if(not isinstance(X, list)): X = [X]
        if(not isinstance(Y, list)): Y = [Y]
        batch_predicts = model.predict_on_batch(X)

        obj_chars = np.array([char(X[o][:,:,observ], axis=1) for o in objects])
        assert obj_chars.shape[0] == len(obj_chars)
        assert obj_chars.shape[1] >= batch_size
        batch_chars = char2(obj_chars, axis=0)
        for j in range(batch_size):
            #print(type(c), type(p))
            #print(c, p.shape)
            #print(type(c),type(p),type(y))
            index = num_read+j
            # try:
            characteristics[index] = batch_chars[j]
            predictions[index] = batch_predicts[j]
            y_vals[index] = Y[0][j]
            # except Exception as e:
            #     print(index, i, j, batch_size, global_batch_size, "MOOP", len(characteristics),len(predictions), len(y_vals), len(batch_chars), len(batch_predicts))
        num_read += batch_size
        # if(num_read > 59000): print(num_read)
        if(num_read >= num_samples):
            # print(num_read,num_samples, len(characteristics), characteristics[:10])
            break
    characteristics = np.array(characteristics)
    predictions = np.array(predictions)
    y_vals = np.array(y_vals)
    
    sorted_indicies = np.argsort(characteristics)
    
    characteristics = characteristics[sorted_indicies]
    predictions = predictions[sorted_indicies]
    y_vals = y_vals[sorted_indicies]
    
    min_char = characteristics[0]
    max_char = characteristics[characteristics.shape[0]-1]
    if(not equalBins):
        stride = (max_char-min_char)/bins
        split_vals = [min_char+stride*(i+1) for i in range(bins-1)]
        split_at = characteristics.searchsorted(split_vals)
    else:
        stride = characteristics.shape[0]/float(bins)
        split_at = [int(stride*float(i+1)) for i in range(bins-1)]
       
    predict_bins = np.split(predictions, split_at)
    y_bins = np.split(y_vals, split_at)
    true_false_bins = [np.equal(np.argmax(p, axis=-1),np.argmax(y, axis=-1)).astype("float64") for (p,y) in zip(predict_bins, y_bins)]
    
    out_bins = []
    prevmax = min_char
    for i,tf in enumerate(true_false_bins):
        b = {}
        num = tf.shape[0]
        b["y"] = np.mean(tf)
        b["error"] = np.std(tf)/np.sqrt(num)
        b["num_samples"] = num
        b["min_bin_x"] = prevmax
        if(i == len(true_false_bins)-1):
            b["max_bin_x"] = max_char
        else:
            b["max_bin_x"] = prevmax = characteristics[split_at[i]]
        out_bins.append(b)

    if(plot): plotBins(out_bins)
    return out_bins
