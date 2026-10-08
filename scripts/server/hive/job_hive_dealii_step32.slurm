#!/bin/bash
# This script submits a job to compile deal.ii and then ASPECT
# For deal.ii, a repository of candi is given
# For ASPECT, a repository of ASPECT is given
# Before running, make sure all modules are loaded
# Also, adjust the ASPECT branch and mode (debugrelease, release, debug)you want to compile
# There are options to only compile ASPECT, in this case, an existing deal.ii directory should be given.
# Before the compilation of ASPECT, the script will source the configuration/enable.sh file from deal.ii
#SBATCH --account=billengrp          # SLURM account to use
#SBATCH --partition=high         # Partition to submit the job
#SBATCH --time=24:00:00              # Max execution time (24 hours)
#SBATCH -N 1                         # Number of nodes
#SBATCH -n 16                        # Number of CPUs
#SBATCH --exclusive                  # Exclusive use of node (no sharing with other jobs)
#SBATCH --job-name=dealii-step-32   # Job name

#SBATCH --mail-type=ALL              # Notify user by email on all events
#SBATCH --mail-user=hylli@ucdavis.edu # Email address for notifications

#SBATCH --output=candi_output_%j.out # Redirect standard output to file
#SBATCH --error=candi_error_%j.err   # Redirect standard error to file
#SBATCH --constraint='zen2|zen3' # Do not run or compile on zen4 node

# Dealii and ASPECT DIR
INSTALL_PREFIX=/quobyte/billengrp/Software/deal.ii/deal.ii-9.8.0-toolchain-gcc-13.2.0-openmpi5.0.5-Oct7

# environment
echo "Loading required modules and variables..."
module load cmake/3.28.1
module load openmpi/5.0.5
module load gcc/13.2.0
echo "Sourcing up DEAL.II configuration..."
source "$INSTALL_PREFIX/configuration/enable.sh"


# for running deal.ii step 32
# run it in the local folder
cd ${INSTALL_PREFIX}/deal.II-v9.8.0/examples/step-32
cmake .
make -j 4
srun step-32
