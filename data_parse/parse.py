import sys, os
if __package__ is None:
	import sys, os
	sys.path.append(os.path.realpath("../../"))
from CMS_SURF_2016.utils.data_parse import ROOT_to_pandas
from CMS_SURF_2016.utils.data_parse import leaves_from_obj
from CMS_SURF_2016.utils.data_parse import DataProcessingProcedure
import numpy as np

C = np.float64(2.99792458e8);  
# print(C)
def four_vec_func(inputs):
		E = inputs[0]
		Eta = inputs[1]
		Phi = inputs[2]
		E_over_c = E/C
		# print("E?C", E, E_over_c)
		px = E_over_c * np.sin(Phi) * np.cos(Eta) 
		py = E_over_c * np.sin(Phi) * np.sin(Eta)
		pz = E_over_c * np.cos(Phi)
		return [E_over_c, px, py, pz]
four_vec_leaves, _ = leaves_from_obj("Photon", ["E", "Eta", "Phi"])
four_vec_proc = DataProcessingProcedure(four_vec_func, four_vec_leaves, ["E/c", "Px","Py","Pz"])

def PID_func(inputs):
	return [22]
PID_proc = DataProcessingProcedure(PID_func, [], ["PID"])


columns=["T", 'EhadOverEem', four_vec_proc, "E", PID_proc]
leaves, columns = leaves_from_obj("Photon", columns)
# columns = ['T', 'Monkey', 'E/c', 'Px', 'Py', 'Pz', 'E']

# print(leaves)
# print(columns)
# print()
# frame = ROOT_to_pandas("../data/ttbar_13TeV_80.root",
# 					  leaves,
# 					  columns=columns,
# 					  verbosity=1)
# frame.to_hdf("ttbar_13TeV_80.h5", 'data')


E_over_c_proc = DataProcessingProcedure(lambda x:[x[0]/C], ["Particle.E"], ["E/c"])
columns= [E_over_c_proc, "Px", "Py", "Pz", "PID", "Charge"]
leaves, columns = leaves_from_obj("Particle", columns)
particle_frame = ROOT_to_pandas("../data/ttbar_13TeV_80.root",
                             leaves,
                              columns=columns,
                              verbosity=1)