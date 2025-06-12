#!/bin/bash -l
#SBATCH -N 1
#SBATCH -n 64
#SBATCH --threads-per-core=2
#SBATCH --tasks-per-node=64
#SBATCH -o task-%j.stdout
#SBATCH -e task-%j.stderr
#SBATCH -t 300:00:00
#SBATCH -A billen
#SBATCH --partition=p-billen
#SBATCH --switches=1

module unload openmpi/4.0.1
module load openmpi/4.1.0-mpi-io
module unload deal.II
module load /group/billengrp/software/deal.ii/deal.ii-9.4.0-NoNative-32bit-candi-gcc-11.1.0-openmpi4.1.0-mpi-io-rome-256-512/configuration/modulefiles/default
>&2 echo "list of modules:"
>&2 module list
>&2 echo "aspect source: ${ASPECT_SOURCE_DIR}"

srun  ${ASPECT_SOURCE_DIR}/build_master_TwoD_dealii-9.4.0/aspect case.prm
