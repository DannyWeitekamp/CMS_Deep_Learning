'''
metrics.py
Contains custom metric utilities for ploting and displaying metrics.
Author: Danny Weitekamp
e-mail: dannyweitekamp@gmail.com
''' 

import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import History
def plot_history( histories, plotLoss=True, plotAccuracy=True):
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
                l+=' (acc %2.4f)'% (history['acc'][-1])
                do_acc = True
            if 'val_acc' in history:
                vl+=' (acc %2.4f)'% (history['val_acc'][-1])
                do_acc = True
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
            if 'val_acc' in history:
                plt.plot(history['val_acc'], lw=2, ls='dashed', label=label+" validation accuracy", color=color)
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