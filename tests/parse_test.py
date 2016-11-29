import sys, os
if __package__ is None:
    sys.path.append(os.path.realpath("../"))
    sys.path.append(os.path.realpath("../../"))
from utils.delphes_parse import delphes_to_pandas
print(os.path.realpath("/data/shared/Delphes/wjets_lepFilter_13TeV/wjets_lepFilter_13TeV_183.root"))
df = delphes_to_pandas(os.path.realpath("/data/shared/Delphes/wjets_lepFilter_13TeV/wjets_lepFilter_13TeV_183.root"))
print(df)
