import unittest
from LorentzLayer import _lorentz

class TestStringMethods(unittest.TestCase):

    def test_lorentz(self):
	beta np.sqrt(.75)
	inp = K.variable(np.array([	[[0, 1, 0, 0],
                        		 [0, 0, 1, 0],
                           		 [0, 0, 0, 1]]]))
	b = K.variable(np.array([ 	 [beta, 0, 0],
                            		 [0, beta, 0],
                            		 [0, 0, beta]]))
	print(K.eval(K.shape(inp)))
	print(K.eval(K.shape(b)))
	o = _lorentz(inp, b)
	#print(K.eval(_g))
	print(K.eval(o))

	beta = np.sqrt(3./8.)
	inp = K.variable(np.array([[	[0, 1, 0, 0],
                    		        [0, 0, 1, 0],
                            		[0, 0, 1, 1]]]))
	b = K.variable(np.array([	[0, beta, beta],
                         		[0, beta, beta],
                         		[0, beta, beta]]))

if __name__ == '__main__':
    unittest.main()
