import os
import sys
import unittest
import CMS_Deep_Learning

if __package__ is None:
    sys.path.append(os.path.realpath("../"))
    sys.path.append(os.path.realpath("../../"))
from CMS_Deep_Learning.preprocessing.delphes_parse import delphes_to_pandas
print(os.path.realpath("/data/shared/Delphes/wjets_lepFilter_13TeV/wjets_lepFilter_13TeV_183.root"))
df = delphes_to_pandas(os.path.realpath("/data/shared/Delphes/wjets_lepFilter_13TeV/wjets_lepFilter_13TeV_183.root"),fixedNum=100)



class TestDelphesParser(unittest.TestCase):
    def sanityCheck(self):
        p = os.path.dirname(os.path.abspath(CMS_Deep_Learning.__file__))
        loc = p + "../data/qcd_lepFilter_13TeV_2.root"
        loc = os.path.abspath(loc)
        self.assertTrue(os.path.exists(loc))
        df = delphes_to_pandas(loc, fixedNum=100)



print(df)
