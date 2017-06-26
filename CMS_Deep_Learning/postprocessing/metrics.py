from CMS_Deep_Learning.postprocessing.plot import plot_bins
from CMS_Deep_Learning.storage.input_handler import inputHandler

from CMS_Deep_Learning.storage.archiving import DataProcedure, KerasTrial
from CMS_Deep_Learning.storage.iterators import TrialIterator, DataIterator
import numpy as np


def accVsEventChar(model,
                   data,
                   char,
                   observ,
                   objects,
                   num_samples=None,
                   char2=None,
                   bins=20,
                   observable_ordering=['E/c', 'Px', 'Py', 'Pz', 'PT_ET', 'Eta', 'Phi', 'Charge', 'X', 'Y', 'Z',
                                        'Dxy', 'Ehad', 'Eem', 'MuIso', 'EleIso', 'ChHadIso', 'NeuHadIso', 'GammaIso'],
                   object_ordering=["Electron", "MuonTight", "Photon", "MissingET", "EFlowPhoton", "EFlowNeutralHadron",
                                    "EFlowTrack"],
                   equalBins=False,
                   custom_objects={},
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
    if (not isinstance(objects, list)): objects = [objects]
    objects = [o if isinstance(o, int) else object_ordering.index(o) for o in objects]
    observ = observ if isinstance(observ, int) else observable_ordering.index(observ)
    if (char2 == None): char2 = char

    def accum(X):
        obj_chars = np.array([char(X[o][:, :, observ], axis=1) for o in objects])
        assert obj_chars.shape[0] == len(obj_chars)
        # assert obj_chars.shape[1] >= batch_size
        batch_chars = char2(obj_chars, axis=0)
        return batch_chars

    if (isinstance(model, KerasTrial)):
        dItr = TrialIterator(model, return_prediction=True, accumilate=accum)
    else:
        dItr = DataIterator(data, num_samples=num_samples, prediction_model=model, accumilate=accum)
    # tup =
    # print(len(tup), tup)
    y_vals, predictions, characteristics = dItr.as_list()
    if (len(y_vals) == 1):
        y_vals = y_vals[0]
    else:
        raise ValueError("Error multiple outputs is ambiguous, got %r outputs", len(y_vals))
    # characteristics = np.random.rand(400)
    # predictions = np.random.rand(400,2)
    # y_vals = np.random.rand(400,2)

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
    true_false_bins = [np.equal(np.argmax(p, axis=-1), np.argmax(y, axis=-1)).astype("float64") for (p, y) in
                       zip(predict_bins, y_bins)]

    out_bins = []
    prevmax = min_char
    for i, tf in enumerate(true_false_bins):
        b = {}
        num = tf.shape[0]
        b["y"] = np.mean(tf)
        b["error"] = np.std(tf) / np.sqrt(num)
        b["num_samples"] = num
        b["min_bin_x"] = prevmax
        if (i == len(true_false_bins) - 1):
            b["max_bin_x"] = max_char
        else:
            b["max_bin_x"] = prevmax = characteristics[split_at[i]]
        out_bins.append(b)

    if (plot): plot_bins(out_bins)
    return out_bins

def get_roc_points(args=[],tpr=[],fpr=[],thresh=[],**kargs):
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
