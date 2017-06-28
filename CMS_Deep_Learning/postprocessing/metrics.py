from CMS_Deep_Learning.storage.input_handler import inputHandler

from CMS_Deep_Learning.storage.archiving import DataProcedure, KerasTrial
from CMS_Deep_Learning.storage.iterators import TrialIterator, DataIterator
import numpy as np


def build_accumilator(char,
                      observ,
                      objects,
                      char2=None,
                      observable_ordering=['E/c', 'Px', 'Py', 'Pz', 'PT_ET', 'Eta', 'Phi',
                                           "MaxLepDeltaEta", "MaxLepDeltaPhi", 'MaxLepDeltaR', 'MaxLepKt',
                                           'MaxLepAntiKt',
                                           "METDeltaEta", "METDeltaPhi", 'METDeltaR', 'METKt', 'METAntiKt',
                                           'Charge', 'X', 'Y', 'Z',
                                           'Dxy', 'Ehad', 'Eem', 'MuIso', 'EleIso', 'ChHadIso', 'NeuHadIso', 'GammaIso',
                                           "ObjFt1", "ObjFt2", "ObjFt3"],
                      object_ordering=["EFlowPhoton", "EFlowNeutralHadron", "EFlowTrack", "Electron", "MuonTight",
                                       "MissingET"]):
    ''' Builds an accumilator function, a functional of some list of numpy inputs, that can be used to compute
        data characteristics.

        :param char: Any numpy function that reduces data along an axis, (i.e np.sum, np.avg, np.std). This is the 1st reduction of the characteristics
                 reducing the data within each object type of a sample.
        :param observ: The observable to be reduced (i.e PT_ET, E/c, Phi).
        :param objects: What objects should be included in the characteristic computation.
        :param char2: Defaults to the same as char. A numpy function that reduces data along an axis. In this case to reduce between objects.
        :param observable_ordering: A list of the observables in each sample ordered as they are in the sample. It is IMPORTANT that this matches the observables
                                in the sample, otherwise the "observ" argument will not select the intended column in the data.
        :param object_ordering: A list of the possible objects in the data ordered as they are in the sample. This corresponds to the ordering of the ObjectProfiles
                            when the data was created. If this argument does not match the data then the wrong objects will be selected for analysis.
        :returns: the accumilator function
        '''
    if (not isinstance(objects, list)): objects = [objects]
    objects = [o if isinstance(o, int) or isinstance(o, dict) else object_ordering.index(o) for o in objects]
    observ = observ if isinstance(observ, int) else observable_ordering.index(observ)
    if (char2 == None): char2 = char

    def accum(X):
        assert X[0].shape[2] == len(observable_ordering), \
            "X and observable_ordering have different last dimension %r != %r" % (
            X[0].shape[2], len(observable_ordering))
        if (len(X) == 1):
            obj_chars = []
            for o in objects:
                if (isinstance(o, dict)):
                    indxs, vals = zip(*sorted([(observable_ordering.index(key), val) for key, val in o.items()],
                                              key=lambda x: x[0]))
                    vals = np.array(vals)
                    satisfied_mask = (X[0][:, :, indxs] == vals).all(axis=2).reshape((X[0].shape[0], X[0].shape[1], 1))
                    x_subs = (X[0] * satisfied_mask)[:, :, observ]

                    obj_chars.append(char(x_subs, axis=1))
                else:
                    raise ValueError("IDK? Try using dict vals instead.")
            obj_chars = np.array(obj_chars)
        else:
            obj_chars = np.array([char(X[o][:, :, observ], axis=1) for o in objects])
        assert obj_chars.shape[0] == len(obj_chars)
        batch_chars = char2(obj_chars, axis=0)
        return batch_chars

    return accum


def bin_metric_vs_char(args=[],
                       char_name=None,
                       char_collection=None,
                       accumilate=None,
                       num_samples=None,
                       bins=20,
                       equalBins=False,
                       custom_objects={},
                       plot=False,
                       **kargs):
    '''Computes event features and and returns binned data about the accuracy of a model against those features. Also computes the standard error for each bin.

        :param accumilate: an accumilator function build by build_accumilator
        :param num_samples: The number of samples to be read from a generator dat input.
        :param bins: The number of bins to use in the analysis.
        :param equalBins: True/False, Defualt False. If True, will try to put an equal number of samples in each bin. This should probably be left False or else the bins
                        will be very unusual, varying significantly in their domain.
        :param custom_objects: A dictionary keyed by names containing the classes of any model components not used in the standard Keras library.
        :param plot: If True plot the bins automatically.
        
        :Keyword Arguments:
        - **bins** (``list``) -- A list of dictionaries outputted by CMS_Deep_Learning.postprocessing.metrics.bin_metric_vs_char
        - **threshold** -- The threshold for the classifier for the 'true_class'
        - **true_class_index** (``int``) -- The index in the output vector corresponding to the 'true class' element        
        - **trial** (``KerasTrial``) -- a KerasTrial instance from which the model/predictions and validation set will be inferred
        - **Y** (``numpy.ndarray``) -- The groundtruth labels
        - **predictions** (``numpy.ndarray``) -- the model predictions
        - **model** (``Model``,``str``) -- a compiled model, uncompiled model or path to model json. For the latter options
                  weights=? must be given.
        - **weights** (``numpy.ndarry``,``str``) -- the model weights, or a path to the weights
        - **custom_objects** (``dict``) -- A dictionary of classes used inside a keras model that have been made by the user


        :returns: A list of dictionaries each containing information about a bin. The output of this can be plotted with CMS_SURF_2016
            '''
    inputs = args
    if (len(args) == 0):
        inputs = [kargs]
    else:
        raise NotImplementedError("Have not written to take multiple inputs")

    inp = inputs[0]

    if (accumilate != None):
        h = inputHandler(['Y', 'predictions', 'characteristics'])
        inp["accumilate"] = accumilate
        y_vals, predictions, characteristics = h(inp)
    else:
        raise NotImplementedError("Need to write code for getting characteristics strait from EventChars collection")

    if (len(y_vals) == 1):
        y_vals = y_vals[0]
    else:
        raise ValueError("Error multiple outputs is ambiguous, got %r outputs" % len(y_vals))
    
    true_class_index = inp.get('true_class_index',-1)
    if (len(y_vals.shape) == 1 or y_vals.shape[-1] == 1):
        true_class_index = 0
    elif (true_class_index == -1):
        raise ValueError("Must provide a true_class_index.")

    sorted_indicies = np.argsort(characteristics)

    characteristics = characteristics[sorted_indicies]
    predictions = predictions[sorted_indicies]
    y_vals = y_vals[sorted_indicies]

    min_char = characteristics[0]
    max_char = characteristics[characteristics.shape[0] - 1]
    if (not equalBins):
        stride = (max_char - min_char) / bins
        split_vals = [min_char + stride * (i + 1) for i in range(bins - 1)]
        split_at = characteristics.searchsorted(split_vals)
    else:
        stride = characteristics.shape[0] / float(bins)
        split_at = [int(stride * float(i + 1)) for i in range(bins - 1)]

    predict_bins = np.split(predictions, split_at)
    y_bins = np.split(y_vals, split_at)

    def non_gen(n):
        for i in range(n):
            yield

    n = len(predict_bins)
    tp_bins, fp_bins, tn_bins, fn_bins, cont_bins = non_gen(n), non_gen(n), non_gen(n), non_gen(n), non_gen(n)
    
    threshold = inp.get('threshold',-1)
    if (threshold == -1): threshold = 1.0 / max(y_vals.shape[-1], 2)

    PosClassPop_indicies = [[i for i, v in enumerate(np.argmax(y, axis=-1) == true_class_index) if v] \
                            for (p, y) in zip(predict_bins, y_bins)]
    NegClassPop_indicies = [[i for i, v in enumerate(np.argmax(y, axis=-1) != true_class_index) if v] \
                            for (p, y) in zip(predict_bins, y_bins)]

    tp_bins = [(p[:, true_class_index] >= threshold)[PosClassPop_indicies[i]].astype("int") \
               for i, p in enumerate(predict_bins)]
    fp_bins = [(p[:, true_class_index] >= threshold)[NegClassPop_indicies[i]].astype("int") \
               for i, p in enumerate(predict_bins)]
    tn_bins = [(p[:, true_class_index] < threshold)[NegClassPop_indicies[i]].astype("int") \
               for i, p in enumerate(predict_bins)]
    fn_bins = [(p[:, true_class_index] < threshold)[PosClassPop_indicies[i]].astype("int") \
               for i, p in enumerate(predict_bins)]

    cont_bins = [(np.argmax(y, axis=-1))[np.where(p[:, true_class_index] >= threshold)] for i, (p, y) in
                 enumerate(zip(predict_bins, y_bins))]
    labelArgMax_bins = [{x: np.sum(np.argmax(y, axis=-1) == x) for x in range(y_vals.shape[-1])} for y in y_bins]

    true_false_bins = [np.equal(np.argmax(p, axis=-1), np.argmax(y, axis=-1)).astype("float64") for (p, y) in
                       zip(predict_bins, y_bins)]
    out_bins = []
    prevmax = min_char
    for i, (tf, tp, fp, tn, fn, agm, cb) in enumerate(
            zip(true_false_bins, tp_bins, fp_bins, tn_bins, fn_bins, labelArgMax_bins, cont_bins)):
        b = {}
        num = tf.shape[0]
        _tp, _fp, _tn, _fn = np.sum(tp), np.sum(fp), np.sum(tn), np.sum(fn)
        pos_pop = max(_tp + _fn, 1)
        neg_pop = max(_tn + _fp, 1)
        b["tpr"] = float(_tp) / pos_pop
        b["fpr"] = float(_fp) / neg_pop
        b["ppv"] = float(_tp) / max(_fp + _tp, 1)
        b["acc"] = np.mean(tf)
        b["acc_std"] = np.std(tf)
        b["tpr_std"] = np.std(tp)
        b["fpr_std"] = np.std(fp)
        b["acc_error"] = b["acc_std"] / np.sqrt(num)
        b["tpr_error"] = b["tpr_std"] / np.sqrt(pos_pop)
        b["fpr_error"] = b["fpr_std"] / np.sqrt(neg_pop)
        unique, counts = np.unique(cb, return_counts=True)
        cont_classes_d = dict(zip(unique, counts))
        b["cont_classes"] = {key: float(val) / neg_pop for key, val in cont_classes_d.items() if
                             key != true_class_index}
        # print(b["cont_classes"])
        b["arg_max"] = agm
        b["num_samples"] = num
        b["min_bin_x"] = prevmax
        if (i == len(true_false_bins) - 1):
            b["max_bin_x"] = max_char
        else:
            b["max_bin_x"] = prevmax = characteristics[split_at[i]]
        out_bins.append(b)

    if (plot):
        from CMS_Deep_Learning.postprocessing.plot import plot_bins
        plot_bins(out_bins)
    return out_bins

def get_roc_points(args=[],tpr=[],fpr=[],thresh=[],**kargs):
    '''Finds the tpr,fpr, and threshold holding one of them constant.
    
        :param args: a list of kargs dictionaries for trials to evaluate
        :param tpr: a list of true positive rates to hold constant
        :param fpr: a list of false positive rates to hold constant
        :param thresh: a list of thesholds to hold constant
        :returns: a list of lists of tuples correspondind to the ROC points evaluated for the different trials
        '''
    if(len(args) == 0): args = [kargs]
    out = []
    for inp in args:
        roc_data = get_roc_data(**inp)
        #print(roc_dict)
        _fpr, _tpr, _thresh,auc = roc_data
        indicies = []
        for y in fpr:
            index, elmt = min(enumerate(_fpr), key=lambda x:abs(x[1]-y))
            indicies.append(index)
        for y in tpr:
            index, elmt = min(enumerate(_tpr), key=lambda x:abs(x[1]-y))
            indicies.append(index)
        for y in thresh:
            index, elmt = min(enumerate(_thresh), key=lambda x:abs(x[1]-y))
            indicies.append(index)
        fpr,tpr,thresh = [None]*len(indicies),[None]*len(indicies),[None]*len(indicies)
        for i,indx in enumerate(indicies):
            fpr[i],tpr[i],thresh[i] = _fpr[indx], _tpr[indx], _thresh[indx]
        out.append({"tpr": tpr, "fpr":fpr, "thresh": thresh})
    return out


def get_roc_data(**kargs):
    '''get ROC curve points,thresholds, and auc from labels and predictions

        :param ROC_data: a tuple (fpr, tpr,thres,roc_auc) containing the roc parametrization and the auc
        :param trial: a KerasTrial instance from which the model/predictions and validation set will be inferred
        :param Y: The data labels numpy.ndarray
        :param predictions: the predictions numpy.ndarray
        :param model: a compiled model, uncompiled model or path to model json. For the latter options
                      weights=? must be given.
        :param weights: the model weights, or a path to the weights
        :param custom_objects: A dictionary of classes used inside a keras model that have been made by the user
        '''
    inp = kargs
    if ("ROC_data" in inp):
        fpr, tpr, thres, roc_auc = inp["ROC_data"]
    else:
        from sklearn.metrics import roc_curve, auc
        h = inputHandler(['Y', 'predictions'])
        labels, predictions = h(inp)
        labels = labels[0]
        true_class_index = kargs.get("true_class_index", None)

        assert labels.shape == predictions.shape, "labels and predictions should have \
            the same shape, %r != %r" % (labels.shape, predictions.shape)
        n = labels.shape[0]
        if (len(labels.shape) > 1 and labels.shape[1] > 1):
            if (true_class_index != None):
                labels = labels[:, true_class_index].ravel()
                predictions = predictions[:, true_class_index].ravel()
            else:
                raise ValueError("must designate index of true class for data of shape %r" % list(labels.shape))

        fpr, tpr, thres = roc_curve(labels, predictions)
        roc_auc = auc(fpr, tpr)
    return fpr, tpr, thres, roc_auc





















# def getError(model, data=None, num_samples=None,custom_objects={}, ignoreAssert=False):
#     '''
#     Finds the standard error of the mean for the validation accuracy of a model on a dataset or a trial.
#     #Arguements:
#             model -- The model being evaluated, or a KerasTrial containing a valid model.
#             data  -- A generator, or DataProcedure containing the data to be run through the model. If a generator or DataProcedure
#                      containing a generator is given the num_samples must be set. If model is a KerasTrial this can be set to None, and the validation
#                      set will be found from the archive (or computed) and used in place of data.
#             num_samples -- The number of samples to evaluate the error on.
#             custom_objects -- A dictionary keyed by names containing the classes of any model components not used in the standard Keras library.
#             ignoreAssert -- If True ignore assertion errors. This code tests to see that the validation accuracy it computes is similar to the one computed by keras.
#                             If this is not the case then an error will be raised.
#     #Returns:
#         The standard error of the validation accuracy
#     '''
#     # trial = None
#     # if(isinstance(model, KerasTrial)):
#     #     trial = model
#     #     model = trial.compile(loadweights=True,custom_objects=custom_objects)
#     #     if(data == None):
#     #         val_proc = trial.val_procedure if isinstance(trial.val_procedure, str) else trial.val_procedure[0]
#     #         if(num_samples == None): num_samples = trial.nb_val_samples
#     #         p = DataProcedure.from_json(trial.archive_dir,val_proc)
#     #         data = p.getData()
#     # if(trial != None and trial.get_from_record("val_acc_error") == None):
#     #     #model = trial.compile(loadweights=True,custom_objects=custom_objects)
#     #     # val_proc = trial.val_procedure if isinstance(trial.val_procedure, str) else trial.val_procedure[0]
#     #     # if(num_samples == None): num_samples = trial.nb_val_samples
#     #     # p = DataProcedure.from_json(trial.archive_dir,val_proc)
#     #     # gen = p.getData()
#     #
#     #     num_read = 0
#     #     correct = 0
#     #     batch_metrics = None
#     #     num_batches = None
#     #     global_batch_size = None
#     #     i = 0
#     #
#     #     if(isinstance(data, DataProcedure)):
#     #         data = data.getData()
#     #     for X,Y in data:
#     #         batch_size = Y[0].shape[0] if isinstance(Y, list) else Y.shape[0]
#     #         if(batch_metrics == None):
#     #             global_batch_size = batch_size
#     #             num_batches =  np.ceil(num_samples/float(global_batch_size))
#     #             batch_metrics = [None] * num_batches
#     #         #if(batch_size != global_batch_size): continue
#     #         m = model.test_on_batch(X,Y)
#     #         if(i >= num_batches):
#     #             batch_metrics.append(m)
#     #         else:
#     #             #print(i)
#     #             batch_metrics[i] = m
#     #         num_read += batch_size
#     #         i += 1
#     #         if(num_read >= num_samples):
#     #             break
# 
#     isTrial = False
#     if(isinstance(model, KerasTrial)):
#         trial = model
#         model = trial.compile(loadweights=True,custom_objects=custom_objects)
#         isTrial = True
# 
#     def accum(X,Y):
#         return model.test_on_batch(X,Y)
# 
#     if (isTrial):
#         dItr = TrialIterator(trial, return_X=False, return_Y=False, return_prediction=False, accumilate=accum)
#     else:
#         dItr = DataIterator(data, return_X=False, return_Y=False,
#                             num_samples=num_samples, accumilate=accum)
# 
#     batch_metrics = dItr.as_list()
# 
#     batch_metrics = np.array(batch_metrics)
#     avg = np.mean(batch_metrics, axis=0, dtype='float64')
#     sem = np.std(batch_metrics, axis=0, dtype='float64')/np.sqrt(i)
#     if(not ignoreAssert and trial.get_from_record("val_acc") != None):
#         np.testing.assert_almost_equal(trial.get_from_record("val_acc"), avg[1], decimal=3)
#     else:
#         trial.to_record({"val_acc_" : avg[1]})
#     trial.to_record({"val_acc_error" : sem[1]})
#     return trial.get_from_record("val_acc_error")
