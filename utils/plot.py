'''
metrics.py
Contains custom utilities for ploting and displaying training data.
Author: Danny Weitekamp
e-mail: dannyweitekamp@gmail.com
''' 

import matplotlib.pyplot as plt
import numpy as np
from CMS_SURF_2016.utils.analysistools import *
from CMS_SURF_2016.utils.colors import *
from keras.callbacks import History

def plot_history( histories, plotLoss=True, plotAccuracy=True, plotBest=True):
    """ Plots an array of training histories against each other
        -input: [(String label, History hist), .... ]
        -Adopted from Jean-Roch Vlimant's Kreas tutorial"""

    colors=[tuple(np.random.random(3)) for i in range(len(histories))]
    if(plotLoss):
        plt.figure(figsize=(10,10))
        plt.xlabel('Epoch')
        plt.ylabel('loss')
        plt.title('Training Error by Epoch')
        for label,history in histories:
            if(isinstance(history, History)):
                history = history.history
            color = colors[i]
            l = label
            vl= label+" validation"
            if 'acc' in history:
                l+=' (best acc %2.4f)'% (max(history['acc']))
            if 'val_acc' in history:
                vl+=' (best acc %2.4f)'% (max(history['val_acc']))
            plt.plot(history['loss'], label=l, color=color)
            if 'val_loss' in history:
                plt.plot(history['val_loss'], lw=2, ls='dashed', label=vl, color=color)
                
        plt.legend()
        plt.yscale('log')
        plt.show()
    
    if(plotAccuracy):
        plt.figure(figsize=(10,10))
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        for i,(label,history) in enumerate(histories):
            if(isinstance(history, History)):
                history = history.history
            color = colors[i]
            if 'acc' in history:
                plt.plot(history['acc'], lw=2, label=label+" accuracy", color=color)
                if(plotBest):
                    best = max(history['acc'])
                    loc = history['acc'].index(best)
                    plt.scatter( loc, best, s=50, facecolors='none', edgecolors='k',
                                linewidth=2.0, label=label+"best accuracy = %0.4f" % best)
            if 'val_acc' in history:
                plt.plot(history['val_acc'], lw=2, ls='dashed', label=label+" validation accuracy", color=color)
                if(plotBest):
                    best = max(history['val_acc'])
                    loc = history['val_acc'].index(best)
                    plt.scatter( loc, best, s=50, facecolors='none', edgecolors='k',
                                marker='x',linewidth=2.0, label=label+"best validation accuracy = %0.4f" % best)
        plt.legend(loc='lower right')
        plt.show()


def print_accuracy( p, test_target):
    """ Prints the accuracy of a prediction array.
        -Taken from Jean-Roch Vlimant's Kreas tutorial"""
    p_cat = np.argmax(p,axis=1)
    print "Fraction of good prediction"
    print len(np.where( p_cat == test_target)[0])
    print len(np.where( p_cat == test_target )[0])/float(len(p_cat)),"%"
    
def print_accuracy_m( model, test_data, test_target):
    """ Prints the accuracy of a compiled model."""
    ##figure out the shape of the input expected
    #if hasattr('input_dim', model.layers[0]):
    p=model.predict(test_data)
    print_accuracy(p, test_target)




def plotBins(bins, min_samples=10, title='', xlabel='', ylabel='', color='g'):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for b in bins:
        
        if(b["num_samples"] >= min_samples):
            width = b["max_bin_x"]-b["min_bin_x"]
            x = b["min_bin_x"]
            ax.bar(x, b["y"], width=width, yerr=b["error"], color=color, ecolor='k', alpha=.8)
    ax.set_title(title, size=16)
    ax.set_xlabel(xlabel, size=14)
    ax.set_ylabel(ylabel, size=14)
   
    plt.show()

def plotMetricVsMetric(trials,metricX,metricY="test_acc",groupOn=None,constants={}, xlabel=None, ylabel=None, label="Trials", legend_label="", colors=None, alpha=.7, mode="max", verbose=0, verbose_errors=0):
    fig=plt.figure()
    ax1=fig.add_subplot(111)
    if(colors == None):
        colors = colors_contrasting
    trials_by_group = {}
    if(groupOn != None):
        possibleValues = getMetricValues(trials,groupOn)
        #print(possibleValues)
        for v in possibleValues:
            trials_by_group[v] = findWithMetrics(trials, {groupOn:v})
    if(verbose == 1): print("POINTS:")
    i = 0
    for group,group_trials in (sorted(trials_by_group.iteritems()) if len(trials_by_group) > 0 else [(label,trials)]):
        group_trials = findWithMetrics(group_trials, constants)
        group_trials = assertOneToOne(group_trials, metricX,metricY=metricY, mode=mode, verbose_errors=verbose_errors)
        Xs = [ trial.get_from_record(metricX) for trial in group_trials]
        Xs.sort()
        index = np.arange(len(Xs))
        Ys = [trial.get_from_record(metricY) for trial in group_trials]
        if(verbose == 1): print("%s: %r" % (group,zip(Xs, Ys)))
        c = colors[i % len(colors)] 
        j = (i * 3 +4) % len(colors)
        b = colors[j]
        i += 1
        rects1 = plt.scatter(index, Ys,
                         #color='b',
                         #color=tuple(np.random.random(3)),
                         alpha =alpha,
                         s=50,
                         edgecolors=b,
                         color=c,
                         label=group)
        plt.xticks(index, Xs)
    if(xlabel == None): xlabel = metricX
    if(ylabel == None): ylabel = metricY
    
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title('%s vs %s' %(metricY, metricX), fontsize=18)
    legend = ax1.legend(title=legend_label, fontsize=12,loc='center left', bbox_to_anchor=(1, 0.5))
    #plt.legend()
    plt.setp(legend.get_title(),fontsize=14)
    #plt.tight_layout()
    plt.show()