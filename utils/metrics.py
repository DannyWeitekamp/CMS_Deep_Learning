import types
from keras.metrics import *
def accVsEventChar(model,
                   data,
                   char,
                   observ,
                   objects,
                   num_samples=None,
                   char2=None,
                   bins=20,
                   possible_observables=['E/c', 'Px', 'Py', 'Pz', 'Charge', "PT_ET", "Eta", "Phi", "Dxy_Ehad_Eem"],
                   possible_objects=["Electron", "MuonTight", "Photon", "MissingET","EFlowPhoton", "EFlowNeutralHadron", "EFlowTrack"],
                   equalBins=False):
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
    print(possible_objects)
    print(objects)
    if(not isinstance(objects, list)): objects = [objects]
    objects = [o if isinstance(o, int) else possible_objects.index(o) for o in objects]
    observ = observ if isinstance(observ, int) else possible_observables.index(observ)
    if(char2 == None): char2 = char
    num_read = 0
    global_batch_size = None
    for i, out in enumerate(data):
        if(isinstance(out,DataProcedure)):
            X,Y = out.getData()
        else:
            X,Y = out
        batch_size = Y[0].shape[0]
        if(global_batch_size == None): global_batch_size = batch_size
        if(not isinstance(X, list)): X = [X]
        if(not isinstance(Y, list)): Y = [Y]
        batch_predicts = model.predict_on_batch(X)

        obj_chars = np.array([char(X[o][:,:,observ], axis=1) for o in objects])
        assert obj_chars.shape[0] == len(obj_chars)
        assert obj_chars.shape[1] == batch_size
        batch_chars = char2(obj_chars, axis=0)
        for j in range(batch_size):
            #print(type(c), type(p))
            #print(c, p.shape)
            #print(type(c),type(p),type(y))
            index = i*global_batch_size+j
            #print(index)
            characteristics[index] = batch_chars[j]
            predictions[index] = batch_predicts[j]
            y_vals[index] = Y[0][j]
        num_read += batch_size
        if(num_read >= num_samples):
            print(num_read,num_samples, len(characteristics), characteristics[:10])
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
    return out_bins