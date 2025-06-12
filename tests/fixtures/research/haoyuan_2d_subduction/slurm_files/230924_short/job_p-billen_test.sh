#!/bin/bash -l
#SBATCH -N 1
#SBATCH -n 128
#SBATCH --threads-per-core=1
#SBATCH --tasks-per-node=128
#SBATCH -o task-%j.stdout
#SBATCH -e task-%j.stderr
#SBATCH -t 300:00:00
#SBATCH -A billen
#SBATCH --partition=p-billen
#SBATCH --switches=1

source /group/billengrp/software/deal.ii/deal.ii-9.4.0-FullNative-32bit-candi-gcc-11.1.0-openmpi4.1.4-mpi-io-rome-256-512/configuration/enable.sh

>&2 echo "list of modules:"
>&2 module list
>&2 echo "aspect source: ${ASPECT_SOURCE_DIR}"

ulimit -l unlimited

srun  ${ASPECT_SOURCE_DIR}/build_master_TwoD_p-billen/aspect case.prm
