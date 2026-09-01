#!/bin/bash
#SBATCH --partition=skx-dev         # Partition to submit the job
#SBATCH -N 1                         # Number of nodes
#SBATCH --ntasks-per-node=4
#SBATCH --time=2:00:00              # Max execution time (24 hours)
#SBATCH --mem=0
#SBATCH --switches=1

#SBATCH -A TG-EES260002
#SBATCH --job-name=mow_3d   # Job name

#SBATCH --mail-type=ALL              # Notify user by email on all events
#SBATCH --mail-user=hylli@ucdavis.edu # Email address for notifications

#SBATCH --output=candi_%j.out # Redirect standard output to file
#SBATCH --error=candi_%j.err   # Redirect standard error to file

# Dealii and ASPECT DIR
INSTALL_PREFIX=/work2/06806/hyli/stampede3/Softwares/dealii/deal.ii-9.8.0-Native-32bits-candi-intel-26.0-impi-21.18-20260831

# environment
echo "Loading required modules and variables..."
source ~/env/aspect_env_intel26.sh
echo "Sourcing up DEAL.II configuration..."
source "$INSTALL_PREFIX/configuration/enable.sh"


# for running deal.ii step 32
# run it in the local folder
cd ${INSTALL_PREFIX}/deal.II-v9.8.0/examples/step-32
cmake .
make -j 4
ibrun step-32
