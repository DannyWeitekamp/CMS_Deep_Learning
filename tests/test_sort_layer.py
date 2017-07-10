import sys, os

if __package__ is None:
    import sys, os

    sys.path.append(os.path.realpath("../"))
import unittest
from keras.layers import Input
from keras.engine import Model
from CMS_Deep_Learning.layers.sort_layer import Sort,Perturbed_Sort,Finite_Differences
import numpy as np


def test_works():
    a = Input(shape=(None, 4), name="input")
    ls = Sort(initial_beta=np.array([1, 0, 0, 0]), name='sort')
    s = ls(a)
    p_ls = Perturbed_Sort(ls)
    p_s = p_ls(a)
    model = Model(input=[a], output=[s, p_s], name='test')

    # print(Sort(nb_out=5).get_output_shape_for((1,2,3)))

    inp = np.random.random((1000, 300, 4))
    indicies = np.argsort(inp[:, :, 0])
    # print(indicies)
    target = np.array([np.take(inp[i], indicies[i], axis=-2) for i in range(inp.shape[0])])
    # print("Input")
    # print(inp)
    # print("Target")
    # print(target)
    model.compile(optimizer=Finite_Differences(model, ls, p_ls), loss={'sort': 'mse',
                                                                       'pert_sort': 'mse'})
    # model.fit(inp, [target, target], nb_epoch=200,batch_size=1000)
    print(model.evaluate(inp,[target,target],batch_size=1000))
    # print(target)
    
    

# class TestLorentz(unittest.TestCase):
    
        

if __name__ == '__main__':
    import objgraph
    objgraph.show_most_common_types()
    test_works()
    objgraph.show_most_common_types()
    # unittest.main()

