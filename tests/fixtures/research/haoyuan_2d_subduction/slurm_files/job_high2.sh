#!/bin/bash -l
#SBATCH -p high2
#SBATCH -N 1
#SBATCH -n 64
#SBATCH --threads-per-core=2
#SBATCH --tasks-per-node=64
#SBATCH -o task-%j.stdout
#SBATCH -e task-%j.stderr
#SBATCH -t 72:00:00

module unload openmpi/4.0.1
            module load openmpi/4.1.0-mpi-io
module list

srun  /home/lochy/software/aspect/build_master_TwoD/aspect case.prm
