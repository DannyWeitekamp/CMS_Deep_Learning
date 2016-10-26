from mpi4py import MPI
import sys,os
software = "/data/shared/Software/"
if(not software in sys.path):
    sys.path.append(software)
p = "/home/dweitekamp/mpi_learn/"
if(not p in sys.path):
    sys.path.append(p)
from mpi_learn.mpi.manager import get_device


masters = 1
max_gpus = 2
# print("Mooop")
if(len(sys.argv) != 3):
    raise ValueError("MPIKerasTrail_execute.py -- Incorrect number of arguments.")

archive_dir = sys.argv[1]
hashcode = sys.argv[2]
# numProcesses = sys.argv[3]

print(archive_dir, hashcode)

comm = MPI.COMM_WORLD.Dup()
# We have to assign GPUs to processes before importing Theano.
device = get_device( comm, masters, gpu_limit=max_gpus )
print "Process",comm.Get_rank(),"using device",device
os.environ['THEANO_FLAGS'] = "device=%s,floatX=float32" % (device)
import theano
from CMS_SURF_2016.utils.MPIArchiving import MPI_KerasTrial

trial = MPI_KerasTrial.find_by_hashcode(archive_dir, hashcode)
if(trial == None):
    raise ValueError("hashcode does not exist")
if(not isinstance(trial, MPI_KerasTrial)):
    raise TypeError("Trial is not MPI_KerasTrial, got type %r" % type(trial))
trial._execute_MPI(comm=comm)
print(sys.argv[0])
print(sys.argv[1])
