import itertools
from keras.callbacks import History
import matplotlib.pyplot as plt

        
import matplotlib.pyplot as plt
import numpy as np
import copy

def findsubsets(S):
    out = []
    for m in range(2, len(S)):
        out = out + [set(x) for x in itertools.combinations(S, m)]
    return out
def group_by_labels(trials):
    #sets = findsubsets(labels)
    #labels = set()
    trials_by_set = {}
    for trial in trials:
        labels = trial.get_from_record('lables')
        key = tuple([str(x) for x in labels])
        lst = trials_by_set[key] = trials_by_set.get(key, [])
        lst.append(trial)
                
    return trials_by_set

def sortOnMetric(trials,sortMetric='test_acc'):
    def getKey(trial):
        return trial.get_from_record(sortMetric)
    trials.sort(key=getKey, reverse=True)
def print_by_labels(trials, num_print=None, ):
    trials_by_set = group_by_labels(trials)
    for classification in trials_by_set:
        lst = trials_by_set[classification]
        head_str = "\n\n Classification: %r %r" % (classification,
                        "Top %r trials"%num_print if num_print != None else "" )
        print(head_str)
        sortOnMetric(lst)
        if(num_print == None): num_print = len(lst)
        for i in range(min(num_print, len(lst))):
            lst[i].summary(showTraining=False,showValidation=False,showFit=False, showCompilation=False)



def findWithMetrics(trials, metrics):
    if(trials == None or (not hasattr(trials, '__iter__'))):
        raise TypeError("trials must be iterable, but got %r", type(trials))
    if(not isinstance(metrics, dict)):
        raise TypeError("metrics expecting type dict, but got %", type(metrics))
    out = []
    for trial in trials:
        record = trial.read_record()
        ok = True
        for metric, value in metrics.iteritems():
            if(metric in record):
                #print(record[metric], value)
                if(metric == 'name'):
                    if(isinstance(value, list) == False): value = [value]
                    if(not record[metric] in value):
                        ok = False
                else:
                    record_value = record[metric]
                    if(isinstance(record_value, list)): record_value = tuple(record_value)
                    if(isinstance(value, list)): value = tuple(value)
                    #print(record_value, value)
                    if(record_value != value):
                        ok = False
            else:
                if(value != None):
                    ok = False
        if(ok): out.append(trial)
    return out
                
def getMetricValues(trials, metric):
    out = set()
    for trial in trials:
        m = trial.get_from_record(metric)
        if(m != None):
            if(isinstance(m, list)): m = tuple(m)
            out.add(m)
    return out
    
def assertOneToOne(trials, metricX, metricY=None, mode="max"):
    trials = copy.copy(trials)
    if(trials == None or (not hasattr(trials, '__iter__'))):
        raise TypeError("trials must be iterable, but got %r" % type(trials))
    if(not mode in ["error"]):
        if(mode in ["max", "min"]):
            if(metricY == None):
                raise ValueError("metricY must be defined if mode = %r" % mode)
        else:
            raise ValueError("mode %r not recognized. Please choose 'error', 'max' or 'min'.")
            
    d = {}
    for trial in trials:
        x = trial.get_from_record(metricX)
        lst = d.get(x, [])
        lst.append(trial)
        d[x] = lst
    for x, lst in d.iteritems():
        if(len(lst) > 1):
            if(mode == "error"):
                print("%r Trials with %r = %r" % (len(lst), metricX, len(metricX)))
                for trial in lst:
                    trial.summary(showTraining=False,showValidation=False,showFit=False, showCompilation=False)
                raise AssertionError("Supplied trials cannot have one-to-one relationship on metricX = %r" % metricX)
            else:
                if(mode == "max" or mode == "min"):
                    reverse = False
                    if(mode == "max"): reverse = True
                    lst.sort(key=lambda x:x.get_from_record(metricY), reverse=reverse)
                    trials = trials[:1]
                    # for t in lst[1:]: trials.remove(t) 
                else:
                    raise ImplementationError("need to write this")
    return trials