#!/bin/bash -l
# This script submits a job to compile deal.ii and then ASPECT
# For deal.ii, a repository of candi is given
# For ASPECT, a repository of ASPECT is given
# Before running, make sure all modules are loaded
# Also, adjust the ASPECT branch and mode (debugrelease, release, debug)you want to compile
# There are options to only compile ASPECT, in this case, an existing deal.ii directory should be given.
# Before the compilation of ASPECT, the script will source the configuration/enable.sh file from deal.ii
#SBATCH --partition=skx-dev         # Partition to submit the job
#SBATCH -N 1                         # Number of nodes
#SBATCH -n 16                        # Number of CPUs
#SBATCH --time=2:00:00              # Max execution time (24 hours)
#SBATCH --mem=0
#SBATCH --switches=1

#SBATCH -A TG-EES260002
#SBATCH --job-name=candi-toolchain   # Job name

#SBATCH --mail-type=ALL              # Notify user by email on all events
#SBATCH --mail-user=hylli@ucdavis.edu # Email address for notifications

#SBATCH --output=candi_%j.out # Redirect standard output to file
#SBATCH --error=candi_%j.err   # Redirect standard error to file

# set -x  # Enable debug mode (prints commands before executing)

# Build options
# Note:
# set up packages and options in the candi.cfg file beforehand
CANDI_DIR=/work2/06806/hyli/stampede3/Softwares/candi # Path to candi (a script for installing deal.ii and dependencies)
INSTALL_PREFIX=/work2/06806/hyli/stampede3/Softwares/dealii/deal.ii-9.7.0-Native-32bits-candi-intel-24.0-impi-21.11

# Loading modules
echo "Loading required modules and variables..."
source ~/env/aspect_env.sh
source ~/env/utilities.sh

if [[ -z "$CC" || -z "$CXX" || -z "$FC" ]]; then
	echo "Error: Intel MPI compiler wrappers not found."
	exit 1
fi

# Path to download, unpack and build. candi will use these path to build deal.ii
export DOWNLOAD_PATH=/tmp/src1
export UNPACK_PATH=/tmp/unpack1
export BUILD_PATH=/tmp/build1


# Clean directories
remove_if_exists "$DOWNLOAD_PATH"
remove_if_exists "$UNPACK_PATH"
remove_if_exists "$BUILD_PATH"

# Checking environment
echo "Listing loaded modules:"
module list
echo "Checking compiler paths..."
echo "C compiler (mpicc) located at: ${CC}"
echo "C++ compiler (mpic++) located at: ${CXX}"
echo "Fortran compiler (mpif90) located at: ${FC}"
echo "CMAKE located at: $(which cmake)"

# Build deal.II

if ls "$INSTALL_PREFIX"/deal.II* >/dev/null 2>&1; then
    DEAL_II_DIR=$(ls -d "$INSTALL_PREFIX"/deal.II* | head -n 1)
    if [[ -d "$DEAL_II_DIR/include" && -d "$DEAL_II_DIR/lib" ]]; then
        echo "deal.II* exists in $INSTALL_PREFIX and contains 'include' and 'lib' directories."
        echo "Skipping the build process of dealii"
    else
        echo "deal.II* exists in $INSTALL_PREFIX, but 'include' or 'lib' is missing."
        echo "Clean the old directory before re-running this script."
        exit 1
    fi
else
    echo "deal.II* does not exist in $INSTALL_PREFIX"

    # Build necessary dependencies for deal.II using candi
    echo "Starting the build process of dealii"
    
    cd ${CANDI_DIR}
    yes | ./candi.sh --prefix=$INSTALL_PREFIX --platform=deal.II-toolchain/platforms/supported/centos7.platform -j $SLURM_NTASKS
    quit_if_fail "Error: candi.sh encountered an issue during installation. Terminating job."
    
    echo "Installation of dealii succeeds. Cleaning up directories..."
    rm -rf "$DOWNLOAD_PATH" "$UNPACK_PATH" "$BUILD_PATH"
    
    echo "Building of dealii successful"
fi
