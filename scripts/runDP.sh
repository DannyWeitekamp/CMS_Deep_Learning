#!/bin/bash -l

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-core=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint=gpu

export CRAY_CUDA_MPS=1

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
module load daint-gpu
module load h5py/2.6.0-CrayGNU-2016.11-Python-2.7.12-serial
module load pycuda/2016.1.2-CrayGNU-2016.11-Python-2.7.12-cuda-8.0.5

source /scratch/snx3000/dweiteka/p2.7/bin/activate

python $1scripts/runDP.py $1 $2 $3 
