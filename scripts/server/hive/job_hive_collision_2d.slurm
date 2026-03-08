#!/bin/bash -l
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --threads-per-core=1
#SBATCH --tasks-per-node=32
#SBATCH -o job_%j.stdout
#SBATCH -e job_%j.stderr
#SBATCH -t 48:00:00
#SBATCH -A billengrp
#SBATCH --partition=high
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hylli@ucdavis.edu
#SBATCH --job-name=collision_2d_job
#SBATCH --constraint='zen|zen2|zen3' # Do not run or compile on zen4 node

# module load gcc/13.2.0
srun ${ASPECT_SOURCE_DIR}/build_main_collision/aspect-release ./case.prm
# srun ${ASPECT_SOURCE_DIR}/build_main_collision/aspect-debug ./case.prm
