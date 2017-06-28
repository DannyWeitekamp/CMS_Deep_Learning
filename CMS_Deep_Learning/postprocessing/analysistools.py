import copy
import itertools


def findsubsets(S):
    '''Finds all subsets of a set S'''
    out = []
    for m in range(2, len(S)):
        out = out + [set(x) for x in itertools.combinations(S, m)]
    return out

def get_trial_dps(trial, data_type="train"):
    '''Gets all the DataProcedures from a trial'''
    from CMS_Deep_Learning.storage.archiving import DataProcedure
    if (data_type == "val"):
        proc = [DataProcedure.from_json(trial.archive_dir, t) for t in trial.val_procedure]
        # num_samples = trial.nb_val_samples
    elif (data_type == "train"):
        proc = [DataProcedure.from_json(trial.archive_dir, t) for t in trial.train_procedure]
        # num_samples = trial.samples_per_epoch
    return proc

def group_by_labels(trials):
    '''Takes in a set of trials and returns a dictionary keyed by trial record labels with lists of corresponding
    trials as values. '''
    # sets = findsubsets(labels)
    trials_by_set = {}
    # labels = set()
    for trial in trials:
        labels = trial.get_from_record('labels')
        # Accidentally mispelled labels so this is just to make sure the mispelling can be found
        if (labels == None): labels = trial.get_from_record('lables')
        key = tuple([str(x) for x in labels])
        lst = trials_by_set[key] = trials_by_set.get(key, [])
        lst.append(trial)

    return trials_by_set


def sortOnMetric(trials, sortMetric='val_acc'):
    '''Sort a list of trials on a record metric'''

    def getKey(trial):
        return trial.get_from_record(sortMetric)

    trials.sort(key=getKey, reverse=True)


def print_by_labels(trials, num_print=None, sortMetric='val_acc'):
    '''Prints trials ordered and grouped by their labeles

        :param trials: A list of KerasTrials.
        :param num_print: How many to print, Defaults is all of them.
        :param sortMetric: What metric to sort the trials on.
    '''
    trials_by_set = group_by_labels(trials)
    for classification in trials_by_set:
        lst = trials_by_set[classification]
        head_str = "\n\n Classification: %r %r" % (classification,
                                                   "Top %r trials" % num_print if num_print != None else "")
        print(head_str)
        sortOnMetric(lst, sortMetric=sortMetric)
        if (num_print == None): num_print = len(lst)
        for i in range(min(num_print, len(lst))):
            lst[i].summary(showTraining=False, showValidation=False, showFit=False, showCompilation=False)


def findWithMetrics(trials, metrics):
    '''Culls a list of trials, selecting only those trials that satisfy the given metrics

        :param trials: a list of KerasTrials 
        :param metrics: a dictionary of record values. Trials that satisfy these values will be kept, and the rest omitted 
                    (i.e. {'depth' : 1, ...})
        :returns: A culled list of KerasTrials'''
            

    if (trials == None or (not hasattr(trials, '__iter__'))):
        raise TypeError("trials must be iterable, but got %r", type(trials))
    if (not isinstance(metrics, dict)):
        raise TypeError("metrics expecting type dict, but got %", type(metrics))
    out = []
    for trial in trials:
        record = trial.read_record()
        ok = True
        for metric, value in metrics.iteritems():
            if (metric in record):
                # print(record[metric], value)
                if (metric == 'name'):
                    if (isinstance(value, list) == False): value = [value]
                    if (not record[metric] in value):
                        ok = False
                else:
                    record_value = record[metric]
                    if (isinstance(record_value, list)): record_value = tuple(record_value)
                    if (isinstance(value, list)): value = tuple(value)
                    # print(record_value, value)
                    if (record_value != value):
                        ok = False
            else:
                if (value != None):
                    ok = False
        if (ok): out.append(trial)
    return out


def getMetricValues(trials, metric):
    '''Gets a list of record values from a list of trials

        :param trials: a list of KerasTrials
        :param metric: a single record key or list of record keys.
        :returns: A list of record values or if metric is a list a list a tuples containing record values
    '''
            
    out = set()
    for trial in trials:
        m = trial.get_from_record(metric)
        if (m != None):
            if (isinstance(m, list)): m = tuple(m)
            out.add(m)
    return out


def assertOneToOne(trials, metricX, metricY=None, mode="max", ignoreIncomplete=True, verbose_errors=0):
    ''' Asserts that a set of trials have a one-to-one relationship on metricX. In other words that the trials in 'trials' can
        be uniquely identified by the value in their record keyed by 'metricX'. So if metricX='depth' this function asserts that
        no two trials in the input 'trials' list have the same depth value in their record. In the event of a conflict, the argument
        'mode' determines how to proceed.

        :param trials: a list of trials to be checked for a one-to-one relationship
        :param metricX: the metric (record key) whose value must uniquely idetify each trial
        :param metricY: In the event of a conflict when mode = "max" or "min", which metric to use to pick among the conflicting trials.
        :param mode: How to assert a one-to-one relationship between the trials. Either "max" or "min" which simply take the trial
                    with the maximum or minimum 'metricY' value among conflicting trials. Alternately "error" throws an error if a one-to-one
                    relationship cannot be resolved, showing the user the set of conflicting trials.
        :param ignoreIncomplete: Whether or not to ignore trials which did not finish training.
        :param verbose_errors: Whether or not to output long trial summaries if a conflict is found and mode = 'error'

        :returns: a list of trials
    '''
    trials = copy.copy(trials)
    if (trials == None or (not hasattr(trials, '__iter__'))):
        raise TypeError("trials must be iterable, but got %r" % type(trials))
    if (not mode in ["error"]):
        if (mode in ["max", "min"]):
            if (metricY == None):
                raise ValueError("metricY must be defined if mode = %r" % mode)
        else:
            raise ValueError("mode %r not recognized. Please choose 'error', 'max' or 'min'.")

    d = {}
    if (ignoreIncomplete): trials = [t for t in trials if t.is_complete()]
    for trial in trials:
        x = trial.get_from_record(metricX)
        if (isinstance(x, list)): x = tuple(x)
        lst = d.get(x, [])
        lst.append(trial)
        d[x] = lst
    for x, lst in d.iteritems():
        if (len(lst) > 1):
            if (mode == "error"):
                print(" \n\n ONE-TO-ONE ERROR! \n %r Trials with %r = %r" % (len(lst), metricX, x))
                for trial in lst:
                    if (not verbose_errors):
                        trial.summary(showTraining=False, showValidation=False, showFit=False, showCompilation=False)
                    else:
                        trial.summary(showTraining=True, showValidation=True, showFit=True, showCompilation=True)
                raise AssertionError(
                    "Supplied trials cannot have one-to-one relationship on metricX = %r. See the printout above for more information. " % metricX)
            else:
                if (mode == "max" or mode == "min"):
                    reverse = False
                    if (mode == "max"): reverse = True
                    lst.sort(key=lambda x: x.get_from_record(metricY), reverse=reverse)
                    # Remove the tail trials from the big trial list
                    for t in lst[1:]: trials.remove(t)
                else:
                    raise NotImplementedError("need to write this")
    return trials
