import theano.tensor as T

from keras import backend as K  
import numpy as np

from keras.engine.topology import Layer  
import theano


class Aggregate(Layer):
    '''Strings together several layers to make them seem like one '''

    def __init__(self, layers, apply_mask=True, mask_value=0.0, scalefactor=1.0, **kwargs):
        # self.seq_len = seq_len
        # self.init = initializations.get('glorot_uniform')
        self.apply_mask = apply_mask
        self.mask_value = mask_value
        self.supports_masking = True
        self.scalefactor = scalefactor
        self.layers = layers
        super(Aggregate, self).__init__(**kwargs)

    def build(self, input_shape):
        self.trainable_weights = []
        for l in self.layers:
            l.build(input_shape)
            input_shape = l.get_output_shape_for(input_shape)
            self.trainable_weights += l.trainable_weights
        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, inp, mask=None):
        # print("Aggregate", K.int_shape(inp))
        # if(mask != None): print("Agg_mask", K.int_shape(mask))

        x = inp
        for l in self.layers:
            x = l.call(x)

        x = x * self.scalefactor

        if (mask != None and self.apply_mask):
            # print("XSHAPE", K.int_shape(x))
            mask = K.cast(mask, dtype='float32')
            start = i = K.ndim(mask)
            shp = K.shape(x) if K.backend() == "theano" else K.int_shape(x)
            for s in shp[start:]:
                mask = K.expand_dims(mask, dim=i)
                mask = K.repeat_elements(mask, s, axis=i)
                i += 1
            # x = K.squeeze(x,axis=-2)
            x = mask * x + (1.0 - mask) * self.mask_value
            # x = K.expand_dims(x,dim=-2)

        # print("AGG_OUTSHAPE", K.int_shape(x))

        return x

    def get_output_shape_for(self, input_shape):
        for l in self.layers:
            input_shape = l.get_output_shape_for(input_shape)
        return input_shape


def null_cost(y_true, y_pred):
    # output itself is cost and must be minimized
    return K.sum(y_pred) * 0 + K.sum(y_true) * 0


def dist_cost(y_true, y_pred):
    # output itself is cost and must be minimized
    return K.sum(y_pred) * 0 + K.sum(y_true) * 0


def indx_to_transform(indicies):
    if len(indicies.shape) >= 2:
        indicies = indicies.tolist()
        was_list = True
    else:
        indicies = [indicies]
        was_list = False
    out = [np.eye(len(ind))[ind] for ind in indicies]
    return np.array(out) if was_list else out[0]


def applyTransform(inp):
    u, x_T = inp
    x = K.permute_dimensions(x_T, (0, 2, 1))
    u_T = K.permute_dimensions(u, (0, 2, 1))
    x_T = K.batch_dot(u_T, x, axes=[1, 2])
    return x_T



def gradSort(met, X):
    # -----------Start Batch Loop------------
    m = T.fvector()
    x = T.fmatrix()

    # Sort the input on the metric
    z = T.argsort(m, axis=0)
    out = x[z] + 0 * T.sum(m)

    sort = theano.function([m, x], [out])

    # Fix the gradient of the sort operation to be the sum of 
    #   the gradients with respect to the input features
    def grad_edit(inps, grads):
        m, x = inps
        g, = grads

        z = T.argsort(m, axis=0)
        s = T.sum(g, axis=-1)

        am = T.max(abs(s), axis=-1)

        s = 10 * (s - T.clip(s, -.90 * am, .90 * am))

        out = s
        return out, g

    op = theano.OpFromGraph([m, x], [out])
    op.grad = grad_edit

    results, updates = theano.map(fn=op, sequences=[met, X], name='batch_sort')
    # ---------END Batch Loop-----------------

    r = results
    return r

class HardSort(Layer):
    def __init__(self, seq_len, **kwargs):
        self.seq_len = seq_len
        super(HardSort, self).__init__(**kwargs)

    def build(self, input_shape):
        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, inp, mask=None):
        met, X = inp
        return gradSort(met, X)

    def get_output_shape_for(self, input_shape):
        return input_shape[1]