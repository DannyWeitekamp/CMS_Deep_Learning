#!/bin/bash
#DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
#echo $DIR
module load Python/2.7.11-CrayGNU-2016.03
module load pycuda/2015.1.3-CrayGNU-2016.03-Python-2.7.11

module unload gcc/4.9.3
module load gcc/4.8.2
source /scratch/daint/vlimant/p2.7/bin/activate

python /scratch/daint/dweiteka/scripts/runDP.py $1 $2
