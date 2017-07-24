import sys, os

if __package__ is None:
    import sys, os

    sys.path.append(os.path.realpath("../"))
import unittest
from keras.layers import Input, GRU,RepeatVector
from keras.layers.recurrent import Recurrent
from keras.engine import Model
import keras.backend as K 
from CMS_Deep_Learning.layers.ptr_net import Ptr_Layer
from CMS_Deep_Learning.layers.slice import Slice
import numpy as np
import theano
from keras.engine.topology import Layer,initializations,Merge
from keras import backend as K
from keras import optimizers
from keras.regularizers import Regularizer

# from keras import losses


class Identity(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(Identity, self).__init__(**kwargs)

    def compute_mask(self, x, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        x = x[:,-1,:]
        # x = theano.printing.Print("x", attrs=['shape'])(x)
        return x

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[-1])
    

def test_works():
    x = Input(shape=(30, 1), name="input")
    e = GRU(128,return_sequences=True)(x)
    s = Slice("[-1,:]")(e)
    # s = Slice('[-1,:]')(e)
    # s = theano.printing.Print("s")(s)
    r = RepeatVector(30)(s)
    m = Merge(mode='concat',concat_axis=2)([r,x])
    d = GRU(128,return_sequences=True)(m)
    p  = Ptr_Layer(30)([x,e,d])
    
    model = Model(input=x, output=p, name='test')
    
    # print(Sort(nb_out=5).get_output_shape_for((1,2,3)))

    inp = np.random.randint(size=(10000, 30, 1),low=0,high=100)
    indicies = np.argsort(inp[:, :, 0])
    # print(indicies)
    target = np.array([np.take(inp[i], indicies[i], axis=-2) for i in range(inp.shape[0])])
    # print("Input")
    # print(inp)
    # print("Target")
    # print(target)
    model.compile(optimizer=optimizers.Adam(),loss='mse')
    model.fit(inp, target, nb_epoch=500,batch_size=100)
    # print(model.evaluate(inp, target, batch_size=50))
    # print(target)


# class TestLorentz(unittest.TestCase):



if __name__ == '__main__':
    # import objgraph
    from CMS_Deep_Learning.layers.slice import  Slice
    
    
    sl = Slice("[-1,:]")
    print(sl.process_split_str)
    print(sl._decodeSlice("-1"))
    print(sl._decodeSlice("-1:"))
    print(sl._decodeSlice(":-1"))
    print(sl.get_output_shape_for((None,100,150)))
    # objgraph.show_most_common_types()
    test_works()
    # objgraph.show_most_common_types()
    # unittest.main()

