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

# def softmax(x_):
#     maxes = K.max(x_, axis=0, keepdims=True)
#     e = K.exp(x_ - maxes)
#     dist = e / K.sum(e, axis=0)
#     return dist

class Ptr_Layer(Layer):
    def __init__(self, attention_width,**kwargs):
        self.supports_masking = True
        self.init = initializations.get('glorot_uniform')
        self.attention_width = attention_width
        super(Ptr_Layer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        # self.attention_width = input_shape[1][-1]
        self.W1 = self.add_weight((self.attention_width ,input_shape[1][-1]),
                                 initializer=self.init,
                                 name='{}_W1'.format(self.name))
        self.W2 = self.add_weight((self.attention_width ,input_shape[2][-1] if len(input_shape) > 2 else input_shape[1][-1]),
                                  initializer=self.init,
                                  name='{}_W2'.format(self.name))

        self.v = self.add_weight((self.attention_width,),
                                  initializer=self.init,
                                  name='{}_v'.format(self.name))
        self.trainable_weights = [self.W1,self.W2]
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
        # e = x
        n = e.shape[1]
        # Shape key:
        # x:  #(batch_size ,sequence_len, feature_dim)
        # e:  #(batch_size ,sequence_len, recurrent_dim)
        # d:  #(batch_size ,sequence_len, recurrent_dim)


        # 
        # def _ptr_probs(e,d):
        #     # xemb = p[x_, tensor.arange(n_samples), :]  # n_samples * dim_proj
        #     # h, c = _lstm(xm_, xemb, h_, c_, 'lstm_de')
        #     e = theano.printing.Print('e', attrs=['shape'])(e)
        #     d = theano.printing.Print('d', attrs=['shape'])(d)
        #     n = e.shape[0]
        #     u = K.repeat_elements(K.dot(e, self.W1),n) + K.repeat_elements(K.dot(d, self.W2),n).T  # n_steps * n_samples * dim
        #     u = K.tanh(u)  # n_sizes * n_samples * dim_proj
        #     u = K.dot(u, self.v)  # n_sizes * n_samples
        #     # prob = tensor.nnet.softmax(u.T).T  # n_sizes * n_samples
        #     u = theano.printing.Print('u', attrs=['shape'])(u)
        #     prob = softmax(u)
        #     prob = theano.printing.Print('prob', attrs=['shape'])(prob)
        #     
        #     return prob
        # 
        # # x = K.permute_dimensions(K.batch_dot(x,u, axes=[1,2]),(0,2,1))
        # # d = theano.printing.Print('d', attrs=['shape'])(d)
        # 
        # u,_ = theano.scan(_ptr_probs,sequences=[e,d])
        # 
        # u = theano.printing.Print('U', attrs=['shape'])(u)
        # 
        # 
        # x = x + K.sum(K.sum(K.sum(u, axis=-1))) #+ K.sum(K.sum(e, axis=-1))
        # 

        # if False:
        # n = theano.printing.Print('n')(n)
        # 
        
        
        # x = theano.printing.Print('x', attrs=['shape'])(x) #(batch_size ,sequence_len, feature_dim)
        # e = theano.printing.Print('e', attrs=['shape'])(e) #(batch_size ,sequence_len, recurrent_dim)
        # d = theano.printing.Print('d', attrs=['shape'])(d) #(batch_size ,sequence_len, recurrent_dim)
        
        # Stack v.T into an (n,d) matrix
        # v_stacked = K.repeat_elements(self.v.T, n, axis=0)
        # E = 

        # E = theano.printing.Print('E', attrs=['shape'])(E)
        # E = 
        # E = theano.printing.Print('E', attrs=['shape'])(E)

        # e = theano.printing.Print('e')(e)
        if(False):
            dot_e, dot_d = K.dot(e, self.W1.T), K.dot(d, self.W2.T) # (batch_size ,sequence_len, recurrent_dim)
            # dot_e = theano.printing.Print('dot_e')(dot_e)
            # dot_d = theano.printing.Print('dot_d')(dot_d)
            # # e = theano.printing.Print('e', attrs=['shape'])(e)  
            E = K.repeat_elements(K.expand_dims(dot_e,dim=1), n, axis=1) # (batch_size ,sequence_len, sequence_len, recurrent_dim)
            D = K.repeat_elements(K.expand_dims(dot_d, dim=1), n, axis=1) # (batch_size ,sequence_len, sequence_len, recurrent_dim)
            
            # D = theano.printing.Print('D', attrs=['shape'])(D)
    
            D_T = K.permute_dimensions(D,(0,2,1,3)) # (batch_size ,sequence_len, sequence_len, recurrent_dim)
    
            # D_T = theano.printing.Print('D_T',attrs=['shape'])(D_T) # (batch_size ,sequence_len, sequence_len, recurrent_dim) transposed
            # self.v = theano.printing.Print('v', attrs=['shape'])(self.v)
            
            # moop = K.tanh(E + D_T)
            # moop = theano.printing.Print('moop', attrs=['shape'])(moop)
            u = K.dot(K.tanh(E + D_T),self.v) # (batch_size ,sequence_len, sequence_len)
            # u = theano.printing.Print('U', attrs=['shape'])(u) 
    
            u = K.permute_dimensions(softmax(u,axis=1), (0,2,1))#softmax(u,axis=1)# 
        u = K.tanh(K.dot(e, self.W1.T) + K.permute_dimensions(K.dot(d, self.W2.T),(0,2,1)))
        u = K.permute_dimensions(softmax(u,axis=1), (0,2,1))
        
        # indicies = K.argmax(x,axis=-1)
        # x = K.gather(x, indicies)
        # u = theano.printing.Print('U', attrs=['shape'])(u)
        # x = theano.printing.Print('X', attrs=['shape'])(x)
        # u = theano.printing.Print('U')(u)
        

        # s = K.sum(K.sum(K.sum(d,axis=-1))) +  K.sum(K.sum(e,axis=-1))
        # s = K.sum(K.sum(K.sum(K.sum(u,axis=-1))))#K.sum(K.sum(K.sum(K.sum(D,axis=-1)))) +  K.sum(K.sum(K.sum(E,axis=-1)))
        
        
        # x = x + K.sum(K.sum(K.sum(u, axis=-1))) #+ K.sum(K.sum(e, axis=-1))#K.batch_dot(u,x, axes=[1,2])#K.permute_dimensions(K.batch_dot(x,u, axes=[1,2]),(0,2,1))
        
    
            # x = theano.printing.Print('X', attrs=['shape'])(x)
            # x = theano.printing.Print('X')(x)
            
            # x = x

        # x = theano.printing.Print('X_BEFORE')(x)
        soft_sorted_x = K.batch_dot(x, u, axes=[1, 2])
        # soft_sorted_x = theano.printing.Print('soft_sorted_x', attrs=['shape'])(soft_sorted_x)
        
        x = K.permute_dimensions(soft_sorted_x, (0, 2, 1))
        # x = theano.printing.Print('X_AFTER')(x)
        return x 

    def get_output_shape_for(self, input_shape):
        return tuple(input_shape[0])


