#!/bin/bash -l
#SBATCH -N 1
#SBATCH -n 56
#SBATCH --threads-per-core=1
#SBATCH --tasks-per-node=56
#SBATCH -o task-%j.stdout
#SBATCH -e task-%j.stderr
#SBATCH -t 5:00:00
#SBATCH --partition=normal
#SBATCH --switches=1
#SBATCH --mail-user=hylli@ucdavis.edu
#SBATCH -A EAR23021

>&2 echo "list of modules:"
>&2 module list
>&2 echo "aspect source: ${ASPECT_SOURCE_DIR}"

ibrun  ${ASPECT_SOURCE_DIR}/build_master_TwoD/aspect case.prm
