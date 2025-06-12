#!/bin/bash -l
#SBATCH -J task
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --threads-per-core=2
#SBATCH --tasks-per-node=64
#SBATCH -o task-%j.stdout
#SBATCH -e task-%j.stderr
#SBATCH -t 96:00:00
#SBATCH --partition=rome-256-512
#SBATCH --switches=1

module unload openmpi/4.0.1
module load openmpi/4.1.0-mpi-io
>&2 echo "list of modules:"
>&2 module list

srun  /home/lochy/software/aspect/build_master_TwoD/aspect case_f.prm
