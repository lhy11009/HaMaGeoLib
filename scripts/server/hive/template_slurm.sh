#!/bin/bash
#SBATCH --account=billengrp          # SLURM account to use
#SBATCH --partition=high         # Partition to submit the job
#SBATCH --time=48:00:00              # Max execution time (24 hours)
#SBATCH -N 1                         # Number of nodes
#SBATCH -n 8                        # Number of CPUs
#SBATCH --job-name=case_2d   # Job name

#SBATCH --mail-type=ALL              # Notify user by email on all events
#SBATCH --mail-user=hylli@ucdavis.edu # Email address for notifications

#SBATCH --output=job_%j.log # Redirect standard output to file
#SBATCH --error=job_%j.log   # Redirect standard error to file

set -x  # Enable debug mode (prints commands before executing)

srun /quobyte/billengrp/lochy/Software/aspect/build_master_TwoD_rebase/aspect-release ./case.prm
