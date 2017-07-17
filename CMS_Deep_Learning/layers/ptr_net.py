from keras.engine.topology import Layer,initializations
from keras import backend as K
import theano

def softmax(x, axis=-1):
    """Softmax activation function.
    # Arguments
        x : Tensor.
        axis: Integer, axis along which the softmax normalization is applied.
    # Returns
        Tensor, output of softmax transformation.
    # Raises
        ValueError: In case `dim(x) == 1`.
    """
    ndim = K.ndim(x)
    if ndim == 2:
        return K.softmax(x)
    elif ndim > 2:
        e = K.exp(x - K.max(x, axis=axis, keepdims=True))
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s
    else:
        raise ValueError('Cannot apply softmax to a tensor that is 1D')


class Ptr_Layer(Layer):
    def __init__(self, attention_width,**kwargs):
        self.supports_masking = True
        self.init = initializations.get('glorot_uniform')
        self.attention_width = attention_width
        super(Ptr_Layer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2

        self.W1 = self.add_weight((self.attention_width ,input_shape[1][-1]),
                                 initializer=self.init,
                                 name='{}_W1'.format(self.name))
        self.W2 = self.add_weight((self.attention_width ,input_shape[2][-1] if len(input_shape) > 2 else input_shape[1][-1]),
                                  initializer=self.init,
                                  name='{}_W2'.format(self.name))

        self.v = self.add_weight((self.attention_width,),
                                  initializer=self.init,
                                  name='{}_v'.format(self.name))

        # if self.bias:
        #     self.b = self.add_weight((self.attention_width,1),
        #                              initializer='zero',
        #                              name='{}_b'.format(self.name))
        # else:
        #     self.b = None

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, X, mask=None):
        assert isinstance(X,list) and len(X) >= 2, "Bad input expecting list of input,encoder,decoder"
        
        if(len(X) == 3):
            x,e,d = X
        elif(len(X) == 2):
            x,e = X
            d = e
        
        n = e.shape[1]
        # n = theano.printing.Print('n')(n)
        # 
        # x = theano.printing.Print('x', attrs=['shape'])(x)
        # e = theano.printing.Print('e', attrs=['shape'])(e)
        # d = theano.printing.Print('d', attrs=['shape'])(d)
        
        # Stack v.T into an (n,d) matrix
        # v_stacked = K.repeat_elements(self.v.T, n, axis=0)
        # E = 

        # E = theano.printing.Print('E', attrs=['shape'])(E)
        # E = 
        # E = theano.printing.Print('E', attrs=['shape'])(E)

        E = K.repeat_elements(K.expand_dims(K.dot(e, self.W1.T),dim=1), n, axis=1)

        # E = theano.printing.Print('E', attrs=['shape'])(E)
        
        D = K.repeat_elements(K.expand_dims(K.dot(d,self.W2.T), dim=1), n, axis=1)
        
        # D = theano.printing.Print('D', attrs=['shape'])(D)

        D_T = K.permute_dimensions(D,(0,2,1,3))

        # D_T = theano.printing.Print('D_T', attrs=['shape'])(D_T)
        # self.v = theano.printing.Print('v', attrs=['shape'])(self.v)
        
        # moop = K.tanh(E + D_T)
        # moop = theano.printing.Print('moop', attrs=['shape'])(moop)
        u = K.dot(K.tanh(E + D_T),self.v)
        u = K.permute_dimensions(softmax(u,axis=1), (0,2,1))

        # u = theano.printing.Print('U', attrs=['shape'])(u)
        # u = theano.printing.Print('U')(u)
        

        # s = K.sum(K.sum(K.sum(d,axis=-1))) +  K.sum(K.sum(e,axis=-1))
        # s = K.sum(K.sum(K.sum(K.sum(u,axis=-1))))#K.sum(K.sum(K.sum(K.sum(D,axis=-1)))) +  K.sum(K.sum(K.sum(E,axis=-1)))
        
        
        x = K.permute_dimensions(K.batch_dot(x,u, axes=[1,2]),(0,2,1))
        

        # x = theano.printing.Print('X', attrs=['shape'])(x)
        # x = theano.printing.Print('X')(x)
        
        # x = x 
        
        return x 

    def get_output_shape_for(self, input_shape):
        return tuple(input_shape[0])


