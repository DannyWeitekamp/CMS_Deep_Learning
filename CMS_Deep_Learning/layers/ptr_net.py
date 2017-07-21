from keras.engine.topology import Layer, initializations
from keras import backend as K  
 

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
    # if ndim == 2:
    #    return K.softmax(x)
    if ndim >= 2:
        e = K.exp(x - K.max(x, axis=axis, keepdims=True))
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s
    else:
        raise ValueError('Cannot apply softmax to a tensor that is 1D')


def giniSparsity(softmax_matrix, sparsity_coeff=.025):
    return sparsity_coeff * K.sum(K.sum(K.sum(softmax_matrix * (1.0 - softmax_matrix)))) / K.prod(
        K.cast((K.shape(softmax_matrix)), 'float32'))


# K.eval(giniSparsity(.002*np.random.random((100,100,100))) ) 

class Ptr_Layer(Layer):
    def __init__(self, attention_width, implementation='custom', seq_len=None, sparsity_coeff=1000.0, **kwargs):

        self.supports_masking = True
        self.init = initializations.get('glorot_uniform')
        self.attention_width = attention_width
        self.sparsity_coeff = sparsity_coeff
        self.implementation = implementation
        self.seq_len = seq_len
        super(Ptr_Layer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2

        if (self.implementation == 'custom'):
            assert self.attention_width == input_shape[1][-2], "attention width %r != seq size %r" % (
            self.attention_width, input_shape[1][-2])

        # self.attention_width = input_shape[1][-1]
        self.W1 = self.add_weight((self.attention_width, input_shape[1][-1]),  # (att_dim, recurrent_dim)
                                  initializer=self.init,
                                  name='{}_W1'.format(self.name))
        self.W2 = self.add_weight(
            (self.attention_width, input_shape[2][-1] if len(input_shape) > 2 else input_shape[1][-1]),
            # (att_dim, recurrent_dim)
            initializer=self.init,
            name='{}_W2'.format(self.name))
        if (self.implementation != 'custom'):
            self.v = self.add_weight((self.attention_width, 1),
                                     initializer=self.init,
                                     name='{}_v'.format(self.name))

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, X, mask=None):
        assert isinstance(X, list) and len(X) >= 2, "Bad input expecting list of input,encoder,decoder"

        if (len(X) == 3):
            x_T, e_T, d_T = X
        elif (len(X) == 2):
            x_T, e_T = X
            d_T = e_T
        # (batch_size ,sequence_len, feature_dim) -> (batch_size ,feature_dim,sequence_len)
        x = K.permute_dimensions(x_T, (0, 2, 1))
        # print("SHAPE!!!!", K.eval(x.shape))
        # x_T = theano.printing.Print('x_T',attrs=['shape'])(x_T)
        if (K.backend() == "tensorflow"):
            assert self.seq_len != None, 'Must set Ptr_Layer(seq_len=?) if using Tensorflow'
            seq_len = self.seq_len
        else:
            seq_len = K.shape(e_T)[1]
        # Shape key:
        # x_T:  #(batch_size ,sequence_len, feature_dim)
        # e_T:  #(batch_size ,sequence_len, recurrent_dim)
        # d_T:  #(batch_size ,sequence_len, recurrent_dim)

        # (batch_size ,sequence_len, recurrent_dim) * (recurrent_dim,att_dim) -> #(batch_size ,sequence_len,att_dim)
        _e_T, _d_T = K.dot(e_T, K.transpose(self.W1)), K.dot(d_T, K.transpose(
            self.W2))  # (batch_size ,sequence_len, att_dim)
        _e, _d = K.permute_dimensions(_e_T, (0, 2, 1)), K.permute_dimensions(_d_T, (
        0, 2, 1))  # (batch_size ,att_dim, sequence_len)

        # _e = theano.printing.Print('_e', attrs=['shape'])(_e)
        # _d = theano.printing.Print('_d', attrs=['shape'])(_d)

        def Tmap(fn, arrays, dtype='float32'):
            # assumes all arrays have same leading dim
            indices = K.range(K.shape(arrays[0])[0])
            out = K.map_fn(lambda ii: fn(*[array[ii] for array in arrays]), indices, dtype=dtype)
            return out

        if (self.implementation == 'ptr_net'):
            print("PTR_NET")

            E_T = K.repeat_elements(K.expand_dims(_e_T, dim=1), seq_len,
                                    axis=1)  # (batch_size ,sequence_len, sequence_len, att_dim)
            D_T = K.repeat_elements(K.expand_dims(_d_T, dim=1), seq_len,
                                    axis=1)  # (batch_size ,sequence_len, sequence_len, att_dim)

            D = K.permute_dimensions(D_T, (0, 2, 1, 3))  # (batch_size ,sequence_len, sequence_len, att_dim)

            u = K.squeeze(K.dot(K.tanh(E_T + D), self.v), axis=-1)  # (batch_size ,sequence_len, sequence_len)
            u = K.permute_dimensions(u, (0, 2, 1))
            # axis=2 is row axis therefore u*x has columns that are linear combos of x
            u = softmax(u, axis=2)  # (batch_size ,sequence_len, sequence_len) 
        elif (self.implementation == 'ptr_net_scan'):
            def _ptr_net_u(_e_T, _d_T):
                __E_T = K.repeat_elements(K.expand_dims(_e_T, dim=0), seq_len,
                                          axis=0)  # (sequence_len, sequence_len, att_dim)
                __D_T = K.repeat_elements(K.expand_dims(_d_T, dim=0), seq_len,
                                          axis=0)  # (sequence_len, sequence_len, att_dim)

                __D = K.permute_dimensions(__D_T, (1, 0, 2))  # (sequence_len, sequence_len, att_dim)

                u = K.dot(K.tanh(__E_T + __D), self.v)  # (sequence_len, sequence_len)
                u = K.squeeze(u, axis=-1)
                u = K.permute_dimensions(u, (1, 0))
                u = softmax(u, axis=1)  # (sequence_len, sequence_len) 

                return u

            assert K.backend() == 'tensorflow', 'ptr_net_scan only works with tensorflow backend'
            import tensorflow as tf
            u = tf.map_fn(lambda x: _ptr_net_u(x[0], x[1]), (_e_T, _d_T), dtype=tf.float32)

        elif (self.implementation == 'custom'):

            # only onto if att_dim == sequence_len
            u = _e + _d_T  ## (batch_size ,att_dim, att_dim)
            u = softmax(u, axis=2)  ## (batch_size ,att_dim, att_dim)  
        else:
            raise ValueError("implementation not recognized: %r" % self.implementation)

        self.add_loss(giniSparsity(u, self.sparsity_coeff))

        soft_sorted_x = K.batch_dot(u, x, axes=[1, 2])

        # x_T = K.permute_dimensions(soft_sorted_x, (0, 2, 1))
        return soft_sorted_x  # +K.sum(K.sum(K.sum(u)))#+ K.sum(K.sum(K.sum(_e))) + K.sum(K.sum(K.sum(_d))) 

    def get_output_shape_for(self, input_shape):
        return tuple(input_shape[0])






