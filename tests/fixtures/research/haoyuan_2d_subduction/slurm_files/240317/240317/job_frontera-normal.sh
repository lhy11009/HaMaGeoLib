#!/bin/bash -l
#SBATCH -N 8
#SBATCH -n 448
#SBATCH --threads-per-core=1
#SBATCH --tasks-per-node=56
#SBATCH -o task-%j.stdout
#SBATCH -e task-%j.stderr
#SBATCH -t 48:00:00
#SBATCH --partition=normal
#SBATCH --switches=1
#SBATCH --mail-user=hylli@ucdavis.edu
#SBATCH -A EAR23027
#SBATCH --job-name=eba3d_width51_c22_AR4

module load gcc/12.2.0
source /work2/06806/hyli/frontera/Softwares/dealii/dealii-9.4.0-Native-32bit-candi-gcc-12.2.0-impi-21.9.0-normal/configuration/enable.sh


>&2 echo "list of modules:"
>&2 module list
>&2 echo "aspect source: ${ASPECT_SOURCE_DIR}"

ibrun ${ASPECT_SOURCE_DIR}/build_master_TwoD_9.4.0/aspect case.prm
