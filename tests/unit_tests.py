import sys, os
if __package__ is None:
	import sys, os
	sys.path.append(os.path.realpath("../../"))
import unittest
import numpy as np
import keras.backend as K
from CMS_SURF_2016.layers.lorentz import _lorentz



class TestStringMethods(unittest.TestCase):
	def test_lorentz(self):
		beta = np.sqrt(.75)
		inp = K.variable(np.array([	[[0, 1, 0, 0],
	                        		 [0, 0, 1, 0],
	                           		 [0, 0, 0, 1]]]))
		b = K.variable(np.array([ 	 [beta, 0, 0],
	                            		 [0, beta, 0],
	                            		 [0, 0, beta]]))
		o = K.eval(_lorentz(inp, b))
		should_be = [[[-5.19615221,	2,	2,	2,]]]
		np.testing.assert_almost_equal(o,should_be)

		beta = np.sqrt(3./8.)
		inp = K.variable(np.array([[	[0, 1, 0, 0],
	                    		        [0, 0, 1, 0],
	                            		[0, 0, 1, 1]]]))
		b = K.variable(np.array([	[0, beta, beta],
	                         		[0, beta, beta],
	                         		[0, beta, beta]]))
		
		o = K.eval(_lorentz(inp, b))
		should_be = [[[-3.67423534,	1,	3.50000072,	2.50000048]]]
		np.testing.assert_almost_equal(o,should_be)


if __name__ == '__main__':
	unittest.main()

