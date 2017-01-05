import sys, os
if __package__ is None:
	import sys, os
	sys.path.append(os.path.realpath("../../"))
import unittest
import numpy as np
import keras.backend as K
from CMS_SURF_2016.layers.lorentz import _lorentz



class TestLorentz(unittest.TestCase):
	def test_boost(self):
		beta = np.sqrt(.75)
		inp = K.variable(np.array([	[[0, 1, 0, 0],
	                        		 [0, 0, 1, 0],
	                           		 [0, 0, 0, 1]]]))
		b = K.variable(np.array([ 	 [beta, 0, 0],
	                            		 [0, beta, 0],
	                            		 [0, 0, beta]]))
		o = K.eval(_lorentz(inp, b))
		should_be = [[[-1.7320508,  2.       ,  0.       ,  0.       ],
			        [-1.7320508,  0.       ,  2.       ,  0.       ],
			        [-1.7320508,  0.       ,  0.       ,  2.       ]]]
		np.testing.assert_almost_equal(o,should_be, decimal=6)

		inp = K.variable(np.array(o))
		o = K.eval(_lorentz(inp, np.array([[0,0,0]]), sum_input=True))
		should_be = [[[-5.19615221,	2,	2,	2,]]]
		np.testing.assert_almost_equal(o,should_be, decimal=6)

		beta = np.sqrt(3./8.)
		inp = K.variable(np.array([[	[0, 1, 0, 0],
	                    		        [0, 0, 1, 0],
	                            		[0, 0, 1, 1]]]))
		b = K.variable(np.array([	[0, beta, beta],
	                         		[0, beta, beta],
	                         		[0, beta, beta]]))
		
		o = K.eval(_lorentz(inp, b))
		should_be = [[[ 0.       ,  1.       ,  0.       ,  0.       ],
			        [-1.2247452,  0.       ,  1.5000002,  0.5000002],
			        [-2.4494903,  0.       ,  2.0000005,  2.0000005]]]
		np.testing.assert_almost_equal(o,should_be, decimal=6)

		inp = K.variable(np.array(o))
		o = K.eval(_lorentz(inp, np.array([[0,0,0]]), sum_input=True))
		should_be = [[[-3.67423534,	1,	3.50000072,	2.50000048]]]
		np.testing.assert_almost_equal(o,should_be, decimal=6)

	def test_weight(self):
		inp = K.variable(np.array([[	[0, 1, 0, 0],
	                    		        [0, 0, 1, 0],
	                            		[0, 0, 1, 1]]]))
		b = K.variable(np.array([	[0, 0, 0],
	                         		[0, 0, 0],
	                         		[0, 0, 0]]))
		w = K.variable(np.array([[[[2],[2],[2]]]]))
		o = K.eval(_lorentz(inp, b, weights=w))
		should_be = [[	[0., 2.,  0.,  0.],
			        	[0,  0.,  2.,  0.],
			        	[0,  0.,  2.,  2.]]]
		np.testing.assert_almost_equal(o,should_be, decimal=6)

		inp = K.variable(np.array(o))
		w = K.variable(np.array([[[[2]]]]))
		o = K.eval(_lorentz(inp, np.array([[0,0,0]]), weights=w, sum_input=True))
		should_be = [[[0,4,8,4]]]
		np.testing.assert_almost_equal(o,should_be, decimal=6)


if __name__ == '__main__':
	unittest.main()

