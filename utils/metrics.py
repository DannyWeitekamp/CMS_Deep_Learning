'''
metrics.py
Contains custom metric utilities for ploting and displaying metrics.
Author: Danny Weitekamp
e-mail: dannyweitekamp@gmail.com
''' 

import matplotlib.pyplot as plt
import numpy as np
def plot_history( histories ):
     """ Plots an array of training histories against each other
        -input: [(String label, History hist), .... ]
        -Adopted from Jean-Roch Vlimant's Kreas tutorial"""
    plt.figure(figsize=(10,10))
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.title('Training Error by Epoch')
    colors=[]
    do_acc=False
    for label,histobj in histories:
        color = tuple(np.random.random(3))
        colors.append(color)
        l = label
        vl= label+" validation"
        if 'acc' in histobj.history:
            l+=' (acc %2.4f)'% (histobj.history['acc'][-1])
            do_acc = True
        if 'val_acc' in histobj.history:
            vl+=' (acc %2.4f)'% (histobj.history['val_acc'][-1])
            do_acc = True
        plt.plot(histobj.history['loss'], label=l, color=color)
        if 'val_loss' in histobj.history:
            plt.plot(histobj.history['val_loss'], lw=2, ls='dashed', label=vl, color=color)


    plt.legend()
    plt.yscale('log')
    plt.show()
    if not do_acc: return
    plt.figure(figsize=(10,10))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    for i,(label,histobj) in enumerate(histories):
        color = colors[i]
        if 'acc' in histobj.history:
            plt.plot(histobj.history['acc'], lw=2, label=label+" accuracy", color=color)
        if 'val_acc' in histobj.history:
            plt.plot(histobj.history['val_acc'], lw=2, ls='dashed', label=label+" validation accuracy", color=color)
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
    if hasattr('input_dim', model.layers[0]):
        p=model.predict(test_data)

    accuracy(p)