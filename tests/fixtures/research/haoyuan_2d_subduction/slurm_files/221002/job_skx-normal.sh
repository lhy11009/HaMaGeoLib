#!/bin/bash -l
#SBATCH -N 1
#SBATCH -n 48
#SBATCH --threads-per-core=1
#SBATCH --tasks-per-node=48
#SBATCH -o task-%j.stdout
#SBATCH -e task-%j.stderr
#SBATCH -t 48:00:00
#SBATCH --partition=skx-normal
#SBATCH --switches=1

>&2 echo "list of modules:"
>&2 module list
>&2 echo "aspect source: ${ASPECT_SOURCE_DIR}"

ibrun  ${ASPECT_SOURCE_DIR}/build_master_TwoD/aspect case.prm