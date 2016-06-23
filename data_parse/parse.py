import sys, os
if __package__ is None:
	import sys, os
	sys.path.append(os.path.realpath("../../"))
from CMS_SURF_2016.utils.data_parse import extract_ROOT_data_to_hdf
from CMS_SURF_2016.utils.data_parse import generate_obj_leaves

columns=["T", "Eta", "Phi", "PT", "Flavor", "Charge"]
leaves = generate_obj_leaves("Jet", columns)

extract_ROOT_data_to_hdf("../data/ttbar_13TeV_80.root",
						 "ttbar_13TeV_80.h5",
						  leaves,
						  columns=columns,
						  verbosity=1)
