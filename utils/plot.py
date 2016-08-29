'''
metrics.py
Contains custom utilities for ploting and displaying training data.
Author: Danny Weitekamp
e-mail: dannyweitekamp@gmail.com
''' 

import matplotlib.pyplot as plt
import numpy as np
from .analysistools import *
from .colors import *
from keras.callbacks import History
from keras.utils.visualize_util import plot
from IPython.display import Image, display

def plot_history( histories, plotLoss=True, plotAccuracy=True, plotBest=True, title=None):
    """ Plots an array of training histories against each other
        -input: [(String label, History hist, (optional) color), .... ]
        -Adopted from Jean-Roch Vlimant's Keras tutorial"""

    colors=[tuple(np.random.random(3)) for i in range(len(histories))]
    if(plotLoss):
        plt.figure(figsize=(10,10))
        plt.xlabel('Epoch', fontsize=16)
        plt.ylabel('loss', fontsize=16)
        if(title == None):
            plt.title('Training Error by Epoch', fontsize=20)
        else:
            plt.title(title, fontsize=20)
        for i, h in enumerate(histories):
            if(len(h) == 2):
                label,history = h
                color = colors[i]
            elif(len(h) == 3):
                label,history,color = h
            if(isinstance(history, History)):
                history = history.history
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
        plt.xlabel('Epoch', fontsize=16)
        plt.ylabel('Accuracy', fontsize=16)
        if(title == None):
            plt.title('Validation Accuracy by Epoch', fontsize=20)
        else:
            plt.title(title,fontsize=20)
        for i, h in enumerate(histories):
            if(len(h) == 2):
                label,history = h
                color = colors[i]
            elif(len(h) == 3):
                label,history,color = h
            if(isinstance(history, History)):
                history = history.history
            if 'acc' in history:
                plt.plot(history['acc'], lw=2, label=label+" training accuracy", color=color)
                if(plotBest):
                    best = max(history['acc'])
                    loc = history['acc'].index(best)
                    plt.scatter( loc, best, s=50, facecolors='none', edgecolors=color,
                                linewidth=2.0, label=label+" best training accuracy = %0.4f" % best)
            if 'val_acc' in history:
                plt.plot(history['val_acc'], lw=2, ls='dashed', label=label+" validation accuracy", color=color)
                if(plotBest):
                    best = max(history['val_acc'])
                    loc = history['val_acc'].index(best)
                    plt.scatter( loc, best, s=50, facecolors='none', edgecolors=color,
                                marker='x',linewidth=2.0, label=label+" best validation accuracy = %0.4f" % best)
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
    ''' Plots the output of CMS_SURF_2016.utils.metrics.accVsEventChar
        #Arguments:
            bins -- A list of dictionaries with info about how the bins. (i.e the output of accVsEventChar)
            min_samples -- The minumum number of samples that must be in a bin for it to be plotted.
            title -- The title of the plot
            xlabel -- The xlabel of the plot
            ylabel -- the ylabel of the plot
            color -- the color of the plot (see how matplotlib handles colors)'''
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

def plotMetricVsMetric(trials,metricX,metricY="val_acc",groupOn=None,constants={}, xlabel=None, ylabel=None, label="Trials", legend_label="", colors=None, shapes=None, alpha=.7, mode="max", verbose=0, verbose_errors=0):
    '''Plots one metric that can be found in the records of a set of KerasTrials vs another (i.e. val_acc vs depth). 
        Asserts a one to one relationship incase of duplicate entries.
        #Arguments:
            trials -- A list of trials to be used for the plot. This list will be culled down by specifying constants={"record" : value, ...}.
            metricX -- The record entry to be used for the x-axis of the plot
            metricY -- The record entry to be used for the y-axis of the plot
            groupOn -- The record entry to group the data on to add a second explanitory variable
            constants -- A dictionary of record values (i.e {"depth" : 2, ...}) that should be kept constant in the plot. For example 
                        if you only wanted to plot trials with a dropout of 0.0 you would do constants = {"dropout" : 0.0}.
                        Ideally you want to keep record values that are not being compared constant, to maintain a one-to-one relationship.
                        To be certain of this one-to-one relationship set mode="error".
            xlabel -- The X label of the plot
            ylabel -- The Y label of the plot
            label -- How to label objects in the legend if groupOn=None
            legend_label -- The title for the lengend
            colors -- A list of colors to use to represent each group, defaults to CMS_SURF_2016.utils.colors.colors_contrasting
            shapes -- list of marker shapes to use in the graph, defualts to ['o','s','v', 'D', '^','*', '<', '>']
            alpha -- The alpha value (opacity) for each point in the plot
            mode -- How to assert a one-to-one relationship between the trials in each group. Either "max" or "min" which simply take the trial
                    with the maximum or minimum 'metricY' value among conflicting trials. Alternately "error" throws an error if a one-to-one
                    relationship cannot be resolved. The user can then edit the 'constants' argument to satisfy this relationship. 
                    See CMS_SURF_2016.analysistools.assertOneToOne for more information
            verbose -- Whether or not to output extra information about the internals of the function for debugging.
            verbose_errors -- Whether or not to print out longer summaries for conflicting trials if mode = "error".
            '''

    fig=plt.figure()
    ax1=fig.add_subplot(111)
    if(colors == None):
        colors = colors_contrasting
    if(shapes == None):
        shapes = ['o','s','v', 'D', '^','*', '<', '>']
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
        Ys = [trial.get_from_record(metricY) for trial in group_trials]
        #Sort lists together
        Xs, Ys = [list(x) for x in zip(*sorted(zip(Xs, Ys), key=lambda p: p[0]))]
        if(verbose == 1): print("%s: %r" % (group,zip(Xs, Ys)))
        c = colors[i % len(colors)] 
        j = (i +1) % len(colors)
        b = colors[j]
        s = shapes[i % len(shapes)]
        i += 1
        rects1 = plt.scatter(Xs, Ys,
                         #color='b',
                         #color=tuple(np.random.random(3)),
                         marker=s,
                         alpha =alpha,
                         s=50,
                         edgecolors=b,
                         color=c,
                         label=group)
        plt.xticks(Xs)
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

def plotEverything(trials, custom_objects={}):
    '''Plots all the information about a list of trials. For each trial: the model summary, a picture of the model and a plot of accuracy vs Epoch
        #Arguemnts
            trials -- A list of trials to plot
            custom_objects -- in case your model includes layers that are not keras defaults, a dictionary of the layer classes keyed by their names
    ''' 
    if(not isinstance(trials, list)): trials = [trials]
    for b in trials:
        b.summary(showTraining=False,showValidation=False,showFit=True, showCompilation=False)
        labels = b.get_from_record("labels")
        if(labels == None): labels = b.get_from_record("lables")
        name = str(tuple(labels)) if(labels != None) else "Cannot Find Labels"
        model = b.get_model(custom_objects=custom_objects)
        history = b.get_history()
        plot_history([(name, history)], plotLoss = False)
        dot = plot(model, to_file="model.png", show_shapes=True, show_layer_names=False)
        image = Image("model.png")
        display(image)

def showColors(colors):
    '''Plots a list of colors with outlines taken from the same list'''
    fig, ax = plt.subplots(1)
    fig.set_size_inches((10,10))

    # Show the whole color range
    for i in range(len(colors)):
        x = np.random.normal(loc=(i%4)*3, size=100)
        y = np.random.normal(loc=(i//4)*3, size=100)
        c = colors[i]
        j = (i * 3 +4) % len(colors)
        b = colors[j]
        
        ax.scatter(x, y, label=str(i), alpha=.7, edgecolor=b,s=60, facecolor=c, linewidth=1.0)