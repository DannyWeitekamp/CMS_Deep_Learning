import sys, os

if __package__ is None:
    import sys, os

    sys.path.append(os.path.realpath("../"))
import unittest
from keras.layers import Input, GRU,RepeatVector
from keras.engine import Model
from CMS_Deep_Learning.layers.ptr_net import Ptr_Layer
from CMS_Deep_Learning.layers.slice import Slice
import numpy as np
import theano

def test_works():
    x = Input(shape=(None, 4), name="input")
    e = GRU(100,return_sequences=True)(x)
    
    # s = Slice('[-1,:]')(e)
    # s = theano.printing.Print("s")(s)
    # r = RepeatVector(3)(s)
    # d = GRU(100,return_sequences=True)(e)
    p  = Ptr_Layer(10)([x,e])
    
    model = Model(input=x, output=p, name='test')
    
    # print(Sort(nb_out=5).get_output_shape_for((1,2,3)))

    inp = np.random.random((10000, 30, 4))
    indicies = np.argsort(inp[:, :, 0])
    # print(indicies)
    target = np.array([np.take(inp[i], indicies[i], axis=-2) for i in range(inp.shape[0])])
    # print("Input")
    # print(inp)
    # print("Target")
    # print(target)
    model.compile(optimizer='adam',loss='mse')
    model.fit(inp, target, nb_epoch=5,batch_size=100)
    print(model.evaluate(inp, target, batch_size=50))
    # print(target)


# class TestLorentz(unittest.TestCase):



if __name__ == '__main__':
    import objgraph

    # objgraph.show_most_common_types()
    test_works()
    # objgraph.show_most_common_types()
    # unittest.main()

