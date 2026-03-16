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
#SBATCH -n 8                        # Number of CPUs
#SBATCH --exclusive                  # Exclusive use of node (no sharing with other jobs)
#SBATCH --job-name=candi-toolchain   # Job name

#SBATCH --mail-type=ALL              # Notify user by email on all events
#SBATCH --mail-user=hylli@ucdavis.edu # Email address for notifications

#SBATCH --output=candi_output_%j.out # Redirect standard output to file
#SBATCH --error=candi_error_%j.err   # Redirect standard error to file

set -x  # Enable debug mode (prints commands before executing)

# Build options
# Note:
# set up packages and options in the candi.cfg file beforehand
CANDI_DIR=/quobyte/billengrp/Software/candi_9.5_07182025 # Path to candi (a script for installing deal.ii and dependencies)
INSTALL_PREFIX=/quobyte/billengrp/Software/deal.ii/deal.ii-9.5.1-toolchain-gcc-13.2.0-openmpi5.0.5

# Loading modules
echo "Loading required modules..."
module load gcc/13.2.0
module load cmake/3.28.1

# Path to download, unpack and build. candi will use these path to build deal.ii
export DOWNLOAD_PATH=/tmp/src1
export UNPACK_PATH=/tmp/unpack1
export BUILD_PATH=/tmp/build1


# Function utilities
BAD="\033[1;31m"
GOOD="\033[1;32m"
WARN="\033[1;35m"
INFO="\033[1;34m"
BOLD="\033[1m"

remove_if_exists() {
    # Function to check and remove directories
    if [[ -d "$1" ]]; then
        echo "Removing directory: $1"
        rm -rf "$1"
        if [[ $? -eq 0 ]]; then
            echo "Successfully removed: $1"
        else
            echo "Error: Failed to remove $1" >&2
        fi
    else
        echo "Directory does not exist: $1"
    fi
}

quit_if_fail() {
    STATUS=$?
    if [ ${STATUS} -ne 0 ]; then
        cecho ${BAD} "Failure with exit status: ${STATUS}"
        cecho ${BAD} "Exit message: $1"
        exit ${STATUS}
    fi
}

cecho() {
    COL=$1; shift
    echo -e "${COL}$@\033[0m"
}

# Clean directories
remove_if_exists "$DOWNLOAD_PATH"
remove_if_exists "$UNPACK_PATH"
remove_if_exists "$BUILD_PATH"

# Setting up MPI compilers
echo "Setting up MPI compilers..."
# C compiler
if command -v mpicc >/dev/null 2>&1; then
    export CC=$(command -v mpicc)
else
    echo "Error: mpicc not found."
    exit 1
fi
# Fortran 90 compiler
if command -v mpif90 >/dev/null 2>&1; then
    export FC=$(command -v mpif90)
else
    echo "Error: mpif90 not found."
    exit 1
fi
# Fortran 77 compiler
if command -v mpif77 >/dev/null 2>&1; then
    export FF=$(command -v mpif77)
else
    echo "Error: mpif77 not found."
    exit 1
fi
# C++ compiler: try mpic++ first, fallback to mpicxx
if command -v mpic++ >/dev/null 2>&1; then
    export CXX=$(command -v mpic++)
    export MPI_CXX=$(command -v mpic++)
elif command -v mpicxx >/dev/null 2>&1; then
    export CXX=$(command -v mpicxx)
    export MPI_CXX=$(command -v mpicxx)
else
    echo "Error: No MPI C++ compiler wrapper found (mpic++ or mpicxx)."
    exit 1
fi

# Checking environment
echo "Listing loaded modules:"
module list
echo "Checking compiler paths..."
echo "C compiler (mpicc) located at: $(which mpicc)"
echo "C++ compiler (mpic++) located at: $(which mpic++)"
echo "Fortran compiler (mpif90) located at: $(which mpif90)"
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
    yes | ./candi.sh --prefix=$INSTALL_PREFIX -j $SLURM_NTASKS
    quit_if_fail "Error: candi.sh encountered an issue during installation. Terminating job."
    
    echo "Installation of dealii succeeds. Cleaning up directories..."
    rm -rf "$DOWNLOAD_PATH" "$UNPACK_PATH" "$BUILD_PATH"
    
    echo "Building of dealii successful"
fi