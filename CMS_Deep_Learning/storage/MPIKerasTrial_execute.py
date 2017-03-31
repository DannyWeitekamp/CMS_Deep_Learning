import os
import sys

from mpi4py import MPI

software = "/data/shared/Software/"
if(not software in sys.path):
    sys.path.append(software)
p = "/home/dweitekamp/mpi_learn/"
if(not p in sys.path):
    sys.path.append(p)
from mpi_learn.mpi.manager import get_device

import argparse


parser = argparse.ArgumentParser()
# parser.add_argument('--foo', help='foo help')
parser.add_argument('archive_dir',help='archive directory of the trial')
parser.add_argument('hashcode',help='hashcode for the trial')
parser.add_argument('--masters',help='number of masters', default=1, type=int)
parser.add_argument('--max-gpus',help='maximum number of gpus to use', dest='max_gpus', default=-1, type=int)

args = parser.parse_args()

archive_dir = args.archive_dir
hashcode = args.hashcode
masters = args.masters
max_gpus = args.max_gpus
# print("Mooop")
# if(len(sys.argv) != 3):
#     raise ValueError("MPIKerasTrail_execute.py -- Incorrect number of arguments.")

# if(len(sys.argv)) > 
# numProcesses = sys.argv[3]

print(archive_dir, hashcode, masters, max_gpus)

comm = MPI.COMM_WORLD.Dup()
# We have to assign GPUs to processes before importing Theano.
device = get_device( comm, masters, gpu_limit=max_gpus,  gpu_for_master=True)
print("Process",comm.Get_rank(),"using device",device)
os.environ['THEANO_FLAGS'] = "device=%s,floatX=float32" % (device)
from CMS_Deep_Learning.storage.MPIArchiving import MPI_KerasTrial

trial = MPI_KerasTrial.find(archive_dir, hashcode)
if(trial == None):
    raise ValueError("hashcode does not exist")
if(not isinstance(trial, MPI_KerasTrial)):
    raise TypeError("Trial is not MPI_KerasTrial, got type %r" % type(trial))
trial._execute_MPI(comm=comm)
# print(sys.argv[0])
# print(sys.argv[1])
