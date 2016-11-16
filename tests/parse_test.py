import sys, os
if __package__ is None:
    sys.path.append(os.path.realpath("../"))
from utils.delphes_parse import delphes_to_pandas

df = delphes_to_pandas("../delphi_analysis/ttbar_13TeV_106.h5")
print(df)