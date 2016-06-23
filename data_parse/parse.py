import sys, os
if __package__ is None:
	import sys, os
	sys.path.append(os.path.realpath("../../"))
from CMS_SURF_2016.utils.data_parse import extract_ROOT_data_to_hdf


extract_ROOT_data_to_hdf("../data/ttbar_13TeV_80.root",
						 "ttbar_13TeV_80.h5",
						  ["Jet.T","Jet.Eta", "Jet.Phi", "Jet.PT", "Jet.Flavor", "Jet.Charge"],
						  columns=["T", "Eta", "Phi", "PT", "Flavor", "Charge"],
						  verbosity=1)
