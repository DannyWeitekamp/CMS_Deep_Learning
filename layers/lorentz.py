''' 
lorentz.py
Contains the LorentzLayer
Author: Danny Weitekamp
e-mail: dannyweitekamp@gmail.com
''' 

from keras import backend as K
from keras.constraints import maxnorm
from keras.engine.topology import Layer
import theano
import numpy as np
from theano.compile.nanguardmode import NanGuardMode

#Build K matricies
np_K = np.zeros((3,4,4))
for i in range(0,3):
    np_K[i,0,i+1] = 1
    np_K[i,i+1,0] = 1
    
_K = K.variable(np_K)


def _lorentz(x, boosts,weights=None, sphereCoords=False):
    ''' Outputs a backend variable that Calculates the vectorial sum of 4-vectors
        boosted individually into different reference frames
    '''
    
    #Initialize Helpful variables
    x_shape = K.shape(x)
    batch_size = x_shape[0]
    vector_cluster_size = x_shape[1]
    _bI = K.repeat_elements(K.reshape(K.eye(4), (1,4,4)), vector_cluster_size, axis=0)
    _b1 = K.repeat_elements(K.eye(1),vector_cluster_size, axis=0)
    
    #Get _mag and _n from boost which can be formatted in either
    # Cartesian or Spherical coordinates
    if(sphereCoords):
        #split up the boost by components. dtype='int64' is to cast to a theano.tensor.lvector
        _splits =K.variable([1,1,1], dtype='int64') #theano.tensor.lvector([1,1,1])
        _theta, _phi,_mag = theano.tensor.split(boosts,_splits, 3, axis=1)
        _theta = _theta * np.pi
        _phi = _phi * (2 * np.pi)
        _nx = K.sin(_theta) * K.cos(_phi) 
        _ny = K.sin(_theta) * K.sin(_phi)
        _nz = K.cos(_theta)
        _n = K.concatenate([_nx, _ny, _nz], axis=1)
    else:
        _mag = K.sqrt(K.sum(K.square(boosts), axis=1,keepdims=True))
        _inv_mag = 1/_mag
        _n = boosts *  _inv_mag
    
    #Calculate the Lorentz factor of the boost
    _sqrMag = K.square(_mag)
    _g = 1/K.sqrt((1.-_sqrMag))
    
    #Repeat the K tensor b=vector_cluster_size times 
    _bK = K.reshape(_K, (1,3,4,4))
    _bK = K.repeat_elements(_bK, vector_cluster_size, axis=0)
    #Dot K with n for each cluster vector to get _nk = Bxnx + Byny + Bznz
    _nK = K.batch_dot(_n, _bK, axes=[[1],[1]])
    #Reshape _nk so that we can batch dot it along the [1] axis
    _nK = K.reshape(_nK, (vector_cluster_size,1,4,4))
    #Square _nK and reshape it correctly for each cluster vector
    _nK2 = K.reshape(K.batch_dot(_nK,_nK), (vector_cluster_size,1,4,4))
    #Calculate the boost matrix
    _B = _bI - K.batch_dot(_g*_mag, _nK, axes=[[1],[1]]) +K.batch_dot(_g-_b1,_nK2,axes=[[1],[1]])
    
    #Apply trained weights to each Boost in the cluster
    if(weights != None):
        _B = K.reshape(_B, (vector_cluster_size,1,4,4))
        weights = K.reshape(weights, (vector_cluster_size,1,1,1))
        _B = K.batch_dot(weights, _B, axes=[[1],[1]])
    
    
    #Reshape x and _B so we can dot them along the cluster axis
    x = K.reshape(x, (batch_size, vector_cluster_size, 1, 4))
    _B = K.reshape(_B, (1,vector_cluster_size,4,4))
    _mB = K.repeat_elements(_B,batch_size, axis=0)
    
    #Dot x and _B along the cluster axis to give the summed result of the boosted vectors
    out = K.reshape(K.batch_dot(_mB, x, axes=[[1,3],[1,3]]), (batch_size, 1,4))
    
    return out

class LorentzLayer(Layer):
    ''' A layer that uses the lorentz transformation to analyze input 4-vectors
        in different relativistic frames.
        Trains on a set of weights:
            Bo (boost: which boosts each of any number of input 4-vectors)
            W  (weight: applies a mulitplier to the boosted 4-vectors)
            Bi (bias: boosts the vectorial sum of the input 4-vectors)
    '''
    def __init__(self, cluster_size, sphereCoords=False, **kwargs):
        self.output_dim = 4
        self.sphereCoords = sphereCoords
        kwargs['input_shape'] = (cluster_size, 4)
        super(LorentzLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        #The cluster size
        input_dim = input_shape[1]
        
        #Boosts for each vector in the cluster
        initial_boosts_value = np.random.random((input_dim,3))
        #Bias Boost for the vector sum
        initial_bias_value = np.random.random((1,3))
        #Weight values for each vector in the cluster
        initial_weights_value = np.random.random((input_dim,1))
        
        #If in Cartesian Coordinates scale so maxNorm = 1
        if(~self.sphereCoords):
            initial_boosts_value *= .33
            initial_bias_value *= .33
        
        #store weights
        self.Bo = K.variable(initial_boosts_value)
        self.Bi = K.variable(initial_bias_value)
        self.W = K.variable(initial_weights_value)
        
        #If in Cartesian Coordinates apply maxnorm constraint so that we can
        #only boost our vectors into real reference frames
        if(~self.sphereCoords):
            self.constraints[self.Bo] = maxnorm(axis=1)
            self.constraints[self.Bi] = maxnorm(axis=1)
        
        #Let keras know about our weights
        self.trainable_weights = [self.W, self.Bi, self.Bo]

    def call(self, T, mask=None):
        #T dimensions are (batch_size, cluster_size, 4)
        #lorentzboost of the vectorial sum of each lorentzboosted 4 vector in the cluster
        summed_boosted = _lorentz( T, self.Bo, weights=self.W, sphereCoords=self.sphereCoords)
        out = _lorentz(summed_boosted, self.Bi, sphereCoords=self.sphereCoords)
        return out

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)