#!/bin/bash -l
#SBATCH -N 1
#SBATCH -n 64
#SBATCH --threads-per-core=2
#SBATCH --tasks-per-node=64
#SBATCH -o task-%j.stdout
#SBATCH -e task-%j.stderr
#SBATCH -t 96:00:00
#SBATCH --partition=high2
#SBATCH --switches=1

module unload openmpi/4.0.1
module unload slurm/20.11.8
module unload deal.II
module load openmpi/4.1.0-mpi-io
source /group/billengrp/software/deal.ii/deal.ii-9.3.3-Native-32bit-candi-gcc-11.1.0-openmpi4.1.0-mpi-io-rome-256-512/configuration/enable.sh
>&2 echo "list of modules:"
>&2 module list
>&2 echo "aspect source: ${ASPECT_SOURCE_DIR}"

srun  ${ASPECT_SOURCE_DIR}/build_master_TwoD_high2/aspect case.prm
