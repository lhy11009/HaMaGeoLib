#!/bin/bash -l
#SBATCH -N 1
#SBATCH -n 68
#SBATCH --threads-per-core=1
#SBATCH --tasks-per-node=68
#SBATCH -o task-%j.stdout
#SBATCH -e task-%j.stderr
#SBATCH -t 48:00:00
#SBATCH --partition=normal
#SBATCH --switches=1
#SBATCH -A TG-EES220051

>&2 echo "list of modules:"
>&2 module list
>&2 echo "aspect source: ${ASPECT_SOURCE_DIR}"

ibrun  ${ASPECT_SOURCE_DIR}/build_master_TwoD_knl/aspect case.prm