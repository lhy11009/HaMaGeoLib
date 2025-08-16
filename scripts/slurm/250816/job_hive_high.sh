#!/bin/bash -l
#SBATCH -N 1
#SBATCH -n 128
#SBATCH --threads-per-core=1
#SBATCH --tasks-per-node=128
#SBATCH -o job_%j.stdout
#SBATCH -e job_%j.stderr
#SBATCH -t 48:00:00
#SBATCH -A billengrp
#SBATCH --partition=high
#SBATCH --mail-type=ALL              # Notify user by email on all events
#SBATCH --mail-user=hylli@ucdavis.edu # Email address for notifications

srun  ${ASPECT_SOURCE_DIR}/build_master_TwoD_p-billen/aspect case.prm
