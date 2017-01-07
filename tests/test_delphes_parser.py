import os
import sys
import unittest

if __package__ is None:
    sys.path.append(os.path.realpath("../"))
    sys.path.append(os.path.realpath("../../"))
import CMS_Deep_Learning
from CMS_Deep_Learning.preprocessing.delphes_parser import delphes_to_pandas

class TestDelphesParser(unittest.TestCase):
    def test_sanity(self):
        p = os.path.dirname(os.path.abspath(CMS_Deep_Learning.__file__))
        loc = p + "/../data/qcd_lepFilter_13TeV_2.root"
        loc = os.path.abspath(loc)
        print(loc)
        self.assertTrue(os.path.exists(loc))
        df = delphes_to_pandas(loc, fixedNum=10)
        print(df)

if __name__ == '__main__':
    unittest.main()

print(df)
