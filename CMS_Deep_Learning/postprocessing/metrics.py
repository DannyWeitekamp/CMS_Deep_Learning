from CMS_Deep_Learning.io import simple_grab, assert_list
from CMS_Deep_Learning.preprocessing.pandas_to_numpy import PARTICLE_OBSERVS
import numpy as np


def build_accumulator(char,
                      observ,
                      objects,
                      char2=None,
                      observable_ordering=PARTICLE_OBSERVS,
                      object_ordering=["EFlowPhoton", "EFlowNeutralHadron", "EFlowTrack", "Electron", "MuonTight",
                                       "MissingET"]):
    ''' Builds an accumulator function, a functional of some list of numpy inputs, that can be used to compute
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
        :returns: the accumulator function
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


def distribute_to_bins(bin_by, to_distribute=[], nb_bins=50, equalBins=False, ):
    '''Takes a numpy array of sample characteristics with shape (N,1) and 
        distributes the elements of some other (N,...) like arrays that correspond
        to the same samples into a list of numpy arrays of shape (d_i,...) where d_i
        corresponds the the size of the i_th bin in the output list.

        :param bin_by: numpy array of sample characteristics with shape (N,1)
        :type bin_by: numpy.array
        :param to_distribute: A list of numpy arrays with shape shape (N,...)
        :type to_distribute: list of numpy.array
        :param nb_bins: the number of bins to split into
        :type nb_bins: int
        :param equalBins: Whether force the binning to put the number of
                    samples in each bin.
        :type equalBins: bool

        :returns: tuple (split_vals, ...) 
            WHERE
            **split_vals** A list of the values at which the bins were split including 
                the minimum of the first bin, and the maximum of the last bin.
            **..** arrays given in **to_distribute** split into a list of numpy arrays
                each corresponding to a bin.
        '''
    if(len(bin_by.shape) > 1): bin_by = np.squeeze(bin_by)
    sorted_indicies = np.argsort(bin_by)
    sorted_chars = bin_by[sorted_indicies]

    min_char = sorted_chars[0]
    max_char = sorted_chars[sorted_chars.shape[0] - 1]
    if (not equalBins):
        stride = (max_char - min_char) / nb_bins
        split_vals = [min_char + stride * (i + 1) for i in range(nb_bins - 1)]
        split_at = np.searchsorted(sorted_chars,split_vals)
    else:
        stride = sorted_chars.shape[0] / float(nb_bins)
        split_at = [int(stride * float(i + 1)) for i in range(nb_bins - 1)]
        split_vals = [sorted_chars[0]] + [sorted_chars[i] for i in split_at] + [sorted_chars[-1]]

    to_distribute = [x[sorted_indicies] for x in to_distribute]
    return tuple([[min_char] + split_vals + [max_char]] + [np.split(x, split_at) for x in to_distribute])



def prediction_statistics(target, predictions, true_class_index, threshold=-1):
    '''Returns a dictionary containing a wide variety of statistics about a set of predictions and targets.'''
    from sklearn.metrics import confusion_matrix

    p, y = predictions, target

    if (threshold == -1): threshold = 1.0 / max(y.shape[-1], 2)

    b = {}
    argmax_p = np.argmax(p, axis=-1)
    argmax_y = np.argmax(y, axis=-1)

    # The indicies corresponding to the 'positive' and 'negative' classes
    PosClassPop_indicies = [j for j, v in enumerate(argmax_y == true_class_index) if v]
    NegClassPop_indicies = [j for j, v in enumerate(argmax_y != true_class_index) if v]

    # True-pos,False-pos,True-neg,False-neg
    tp_list = (p[:, true_class_index] >= threshold)[PosClassPop_indicies].astype("int")
    fp_list = (p[:, true_class_index] >= threshold)[NegClassPop_indicies].astype("int")
    tn_list = (p[:, true_class_index] < threshold)[NegClassPop_indicies].astype("int")
    fn_list = (p[:, true_class_index] < threshold)[PosClassPop_indicies].astype("int")

    # Contamination of 'positive' predictions with 'negative' classes
    cont_list = (argmax_y)[np.where(p[:, true_class_index] >= threshold)]

    # Counts of each class in the bin
    freq_dict = {x: np.sum(argmax_y == x) for x in range(y.shape[-1])}

    # True/False (i.e correct=1 incorrect=0)
    tf_list = np.equal(argmax_p, argmax_y).astype("float64")
    num = tf_list.shape[0]

    # Confusion matrix for this class C_ij = 
    # http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    b['confusion'] = confusion_matrix(argmax_y, argmax_p)
    b['norm_confusion'] = b['confusion'].astype('float') / b['confusion'].sum(axis=1)[:, np.newaxis]

    b["tp"] = np.sum(tp_list)
    b["fp"] = np.sum(fp_list)
    b["tn"] = np.sum(tn_list)
    b["fn"] = np.sum(fn_list)
    pos_pop = max(b["tp"] + b["fn"], 1)
    neg_pop = max(b["tn"] + b["fp"], 1)
    nb_pred_pos = max(b["fp"] + b["tp"], 1)
    b["tpr"] = float(b["tp"]) / pos_pop
    b["fpr"] = float(b["fp"]) / neg_pop
    b["ppv"] = float(b["tp"]) / nb_pred_pos
    b["acc"] = np.mean(tf_list)
    b["acc_error"] = np.sqrt(b['acc'] * (1.0 - b['acc']) / num)
    b["tpr_error"] = np.sqrt(b['tpr'] * (1.0 - b['tpr']) / pos_pop)
    b["fpr_error"] = np.sqrt(b['fpr'] * (1.0 - b['fpr']) / neg_pop)
    unique, counts = np.unique(cont_list, return_counts=True)
    cont_classes_d = dict(zip(unique, counts))
    cont_classes_d = {indx: cont_classes_d.get(indx, 0)
                      for indx in range(y.shape[-1]) if indx != true_class_index}
    b["freq"] = freq_dict
    b["cont_split"] = {key: float(val) for key, val in cont_classes_d.items() if
                       key != true_class_index}
    b["norm_cont_split"] = {key: float(val) / freq_dict[key] if freq_dict[key] != 0 else 0
                            for key, val in cont_classes_d.items() if
                            key != true_class_index}
    b["norm_cont_split_error"] = {key: np.sqrt(val * (1.0 - val) / freq_dict[key])
    if freq_dict[key] > 0.0 else 0.0
                                  for key, val in b["norm_cont_split"].items()}

    b["num_samples"] = num
    return b


def bin_metric_vs_char(args=[],
                       nb_bins=20,
                       equalBins=False,
                       plot=False,
                       **kargs):
    '''Computes event features and and returns binned data about the accuracy of a model against those features. Also computes the standard error for each bin.

        :param nb_bins: The number of bins to use in the analysis.
        :param equalBins: True/False, Defualt False. If True, will try to put an equal number of samples in each bin. This should probably be left False or else the bins
                        will be very unusual, varying significantly in their domain.
        :param plot: If True plot the bins automatically.
        :type plot: bool
        :param bins: A list of dictionaries outputted by CMS_Deep_Learning.postprocessing.metrics.bin_metric_vs_char
        :type bins: list of dict
        :param threshold: The threshold for the classifier for the True class. All other classes are collectively the False class.
        :type threshold: float
        :param true_class_index: The index in the output vector corresponding to the True class element.All other classes are collectively the False class.
        :type true_class_index: int
        :param *: Any argument available to :py:func:`CMS_Deep_Learning.io.simple_grab` to get **Y**, **predictions**, **characteristics**

        :returns: A list of dictionaries each containing information about a bin. The output of this can be plotted with CMS_Deep_Learning.postprocessing.plot.plot_bins
            '''

    inputs = args
    if (len(args) == 0):
        inputs = [kargs]
    else:
        raise NotImplementedError("Have not written to take multiple inputs")
    inp = inputs[0]

    if (not isinstance(inp.get('characteristics',None),type(None)) or not isinstance(inp.get('accumulate',None),type(None))):
        y_vals, predictions, characteristics = simple_grab(['Y', 'predictions', 'characteristics'], **inp)
    else:
        raise NotImplementedError("Need to write code for getting characteristics strait from EventChars collection")

    if (isinstance(y_vals, (list, tuple))):
        raise ValueError("Error multiple outputs is ambiguous, got %r outputs" % len(y_vals))

    true_class_index = inp.get('true_class_index', -1)
    threshold = inp.get('threshold', -1)
    if (len(y_vals.shape) == 1 or y_vals.shape[-1] == 1):
        true_class_index = 0
    elif (true_class_index == -1):
        raise ValueError("Must provide a true_class_index.")

    split_vals, y_bins, predict_bins = distribute_to_bins(characteristics, (y_vals, predictions))

    out_bins = []
    prevmax = split_vals[0]
    for i, (p, y) in enumerate(zip(predict_bins, y_bins)):
        b = prediction_statistics(target=y, predictions=p, true_class_index=true_class_index, threshold=threshold)
        b["min_bin_x"] = prevmax
        b["max_bin_x"] = prevmax = split_vals[i + 1] 
        out_bins.append(b)

    return out_bins


def get_class_fprs(y, p, threshs, true_class_index):
    argmax_p = np.argmax(p, axis=-1)
    argmax_y = np.argmax(y, axis=-1)
    n_samples = p.shape[0]
    n_threshs = threshs.shape[0]
    guess_true = p[:, true_class_index].reshape((n_samples, 1)) >= threshs.reshape((1, n_threshs))

    out = {}
    for i in range(p.shape[-1]):
        if (i == true_class_index): continue
        actually_is_class = (argmax_y == i).reshape(n_samples, 1);
        class_pop = np.sum(actually_is_class)
        contamination = np.sum(guess_true * actually_is_class, axis=0, dtype=np.float) / float(class_pop)
        assert (len(contamination) == n_threshs)
        out[i] = contamination

    return out


def get_roc_points(tpr=[], fpr=[], thresh=[], class_fprs={}, class_labels=None, suppress_warnings=False, verbose=0,
                   **kargs):
    '''Finds the tpr,fpr, and threshold holding one of them constant.

        :param tpr: a list of true positive rates to hold constant
        :param fpr: a list of false positive rates to hold constant
        :param class_fprs: a dictionary keyed by class index of false positive rates for each false class to keep constant
        :param thresh: a list of thesholds to hold constant
        :param *: Any argument available to :py:func:`CMS_Deep_Learning.postprocessing.metrics.get_roc_data` to get **ROC_data**,
                    and by extension any argument available to :py:func:`CMS_Deep_Learning.io.simple_grab` to get **Y**, **predictions**

        :returns: a list of lists of tuples correspondind to the ROC points evaluated for the different trials
    '''
    # --------------------- Grabbing Data -------------------------
    try:
        kargs["Y"], kargs["predictions"] = simple_grab(['Y', 'predictions'], **kargs)
    except Exception as e:
        if (verbose > 0): print(e)
    roc_data = get_roc_data(**kargs)
    _fpr, _tpr, _thresh, auc = roc_data

    # ------------------------------------------------------------


    # --------------------Decompose contamination by class---------------------
    if ("Y" in kargs and "predictions" in kargs and "true_class_index" in kargs):
        separated_conts = get_class_fprs(kargs["Y"], kargs["predictions"], _thresh, kargs["true_class_index"])
    elif (not suppress_warnings):
        import warnings
        warnings.warn("Cannot compute CLASS CONTAMINATIONS unless user inputs necessary data " + \
                      "for computing Y and predictions, in addition to true_class_index")

    # -------------------------------------------------------------------



    # ------------------------Find the closest points-------------------
    def indxClosest(target, lst):
        index, elmt = min(enumerate(lst), key=lambda x: abs(x[1] - target))
        return index

    indicies = []
    indicies += [indxClosest(y, _fpr) for y in fpr]
    indicies += [indxClosest(y, _tpr) for y in tpr]
    indicies += [indxClosest(y, _thresh) for y in thresh]
    for key, val in class_fprs.items():
        indicies += [indxClosest(y, separated_conts[key]) for y in val]

    fpr, tpr, thresh = _fpr[indicies], _tpr[indicies], _thresh[indicies]
    out = {"tpr": tpr, "fpr": fpr, "thresh": thresh}
    for j, val in separated_conts.items():
        label = class_labels[j] if class_labels != None else str(j)
        label = "fpr:" + label
        out[label] = val[indicies]

    # -----------------------------------------------------------------------
    return out


def get_roc_data(**kargs):
    '''Get ROC curve **tpr**, **fpr**, **thresholds**, and **auc** from labels and predictions.
        Takes all arguments availiable to :py:func:`CMS_Deep_Learning.io.simple_grab`
        
        :param ROC_data: a tuple (fpr, tpr,thres,roc_auc) containing the roc parametrization and the auc
        :param *: Any argument available to :py:func:`CMS_Deep_Learning.io.simple_grab` to get **Y**, **predictions**
        '''        
    inp = kargs
    if ("ROC_data" in inp):
        fpr, tpr, thres, roc_auc = inp["ROC_data"]
    else:
        from sklearn.metrics import roc_curve, auc
        labels, predictions = simple_grab(['Y', 'predictions'],**inp)
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
#         dItr = TrialIterator(trial, return_X=False, return_Y=False, return_prediction=False, accumulate=accum)
#     else:
#         dItr = DataIterator(data, return_X=False, return_Y=False,
#                             num_samples=num_samples, accumulate=accum)
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
