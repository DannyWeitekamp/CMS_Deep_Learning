from mpi4py import MPI
import sys,os
software = "/data/shared/Software/"
if(not software in sys.path):
    sys.path.append(software)
from CMS_SURF_2016.utils.archiving import KerasTrial
from CMS_SURF_2016.utils.MPIArchiving import MPI_KerasTrial

print("Mooop")
if(len(sys.argv) != 3):
    raise ValueError("MPIKerasTrail_execute.py -- Incorrect number of arguments.")
archive_dir = sys.argv[1]
hashcode = sys.argv[2]
trial = MPI_KerasTrial.find_by_hashcode(archive_dir, hashcode)
if(trial == None):
    raise ValueError("hashcode does not exist")
if(not isinstance(trial, MPI_KerasTrial)):
    raise TypeError("Trial is not MPI_KerasTrial, got type %r" % type(trial))
trial.execute(isMPI_Instance=True)
print(sys.argv[0])
print(sys.argv[1])
