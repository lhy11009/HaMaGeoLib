#!/bin/bash
#SBATCH --account=billengrp          # SLURM account to use
#SBATCH --partition=high         # Partition to submit the job
#SBATCH --time=48:00:00              # Max execution time (24 hours)
#SBATCH -N 1                         # Number of nodes
#SBATCH -n 8                        # Number of CPUs
#SBATCH --job-name=case_2d   # Job name

#SBATCH --mail-type=ALL              # Notify user by email on all events
#SBATCH --mail-user=hylli@ucdavis.edu # Email address for notifications

#SBATCH --output=job_%j.stdout # Redirect standard output to file
#SBATCH --error=job_%j.stderr  # Redirect standard error to file


# for compile aspect
## Enable debug mode (prints commands before executing)
# set -x  
## exclusive use
#SBATCH --exclusive                  # Exclusive use of node (no sharing with other jobs)
## deal.ii 9.6.1
# /quobyte/billengrp/Software/deal.ii/deal.ii-9.6.1-toolchain-gcc-13.2.0-openmpi5.0.5
## ASPECT
# /quobyte/billengrp/lochy/Software/aspect
## WorldBuilder
# /quobyte/billengrp/lochy/Software/WorldBuilder

# for running aspect
srun /quobyte/billengrp/lochy/Software/aspect/build_master_TwoD_rebase/aspect-release ./case.prm