#!/bin/bash -l
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --threads-per-core=1
#SBATCH --tasks-per-node=8
#SBATCH -o job_%j.stdout
#SBATCH -e job_%j.stderr
#SBATCH -t 48:00:00
#SBATCH -A billengrp
#SBATCH --partition=high
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hylli@ucdavis.edu
#SBATCH --job-name=mow_2d_job
#SBATCH --constraint='zen2|zen3' # Do not run or compile on zen4 node

# set up environment
echo "Loading required modules and variables..."

# run in debug and release
srun ${ASPECT_SOURCE_DIR}/build_mater_TwoD_rebase/aspect-release ./case.prm
