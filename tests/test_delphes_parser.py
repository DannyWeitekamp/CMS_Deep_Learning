import os
import sys
import unittest
import pandas as pd

if __package__ is None:
    sys.path.append(os.path.realpath("../"))
    sys.path.append(os.path.realpath("../../"))
import CMS_Deep_Learning
from CMS_Deep_Learning.preprocessing.delphes_parser import delphes_to_pandas, ISO_TYPES

def checkOmission(t,particles, tracks):
    for entry, part_df in particles:
        track_df = tracks.get_group(entry)
        for row in part_df.iterrows():
            t.assertFalse((track_df == row).all(1).any())

def checkIsoNonNegative(t,df):
    t.assertFalse((df[[tup[0] for tup in ISO_TYPES]] < 0.0).any(1).any(0))


def checkFillTrackInfo(t,df):
    t.assertTrue((df[['X', 'Y', 'Z', 'Dxy']] != 0.0).any(1).all(0))



class TestDelphesParser(unittest.TestCase):
    def test_sanity(self):
        p = os.path.dirname(os.path.abspath(CMS_Deep_Learning.__file__))
        loc = p + "/../data/qcd_lepFilter_13TeV_2.root"
        loc = os.path.abspath(loc)
        print(loc)
        self.assertTrue(os.path.exists(loc))
        frames = delphes_to_pandas(loc, fixedNum=10)
        self.assertTrue(isinstance(frames, dict))
        self.assertFalse( False in [isinstance(val, pd.DataFrame) for key, val in frames.items()])
        should_be = set(['Electron', 'MuonTight', 'MissingET', 'EFlowPhoton', 'EFlowNeutralHadron', 'EFlowTrack'])
        keys = set([key for key, val in frames.items()])
        self.assertTrue(should_be.issubset(keys), "%r should be subset of output %r" %(should_be, keys))

        electrons = frames["Electron"].groupby(["Entry"], group_keys=True)
        muons = frames["MuonTight"].groupby(["Entry"], group_keys=True)
        tracks = frames["EFlowTrack"].groupby(["Entry"], group_keys=True)
        checkOmission(self, electrons, tracks)
        checkOmission(self, muons, tracks)
        checkFillTrackInfo(self, electrons)
        checkFillTrackInfo(self, muons)
        for df in frames.values():
            checkIsoNonNegative(self, df)

        print(df)

if __name__ == '__main__':
    unittest.main()

