#!/bin/bash
#SBATCH --account=billengrp          # SLURM account to use
#SBATCH --partition=high             # Partition to submit the job
#SBATCH --time=6:00:00               # Max execution time (24 hours)
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks=128                 # Number of CPUs
#SBATCH --mem=512G		     # amount of memory needed so it can use our nodes
#SBATCH --job-name=compile_aspect   # Job name
#SBATCH --output=compile_aspect_output_%j.log # Redirect standard output to file
#SBATCH --error=compile_aspect_error_%j.log   # Redirect standard error to file
#SBATCH --constraint='zen|zen2|zen3' # Do not run or compile on zen4 node

echo SLURM_NTASKS $SLURM_NTASKS

# Loading modules
#echo "Loading required modules..."
module load cmake/3.28.1
module load openmpi/5.0.5
module load gcc/13.2.0

INSTALL_PREFIX=/quobyte/billengrp/Software/deal.ii/deal.ii-9.6.1-toolchain-gcc-13.2.0-openmpi5.0.5

# Must include this line, otherwie aspect will compile but mpi-io will not work. 
source ${INSTALL_PREFIX}/configuration/enable.sh

ASPECT_INSTALL_DIR="/quobyte/billengrp/lochy/Software/aspect"
WORLD_BUILDER_DIR="/quobyte/billengrp/lochy/Software/WorldBuilder"
ASPECT_BUILD_DIR="/quobyte/billengrp/lochy/Software/aspect/build_main_collision"

# Setting up MPI compilers
# Must include these lines otherwise compile will hang and fail in the middle of compiling.
echo "Setting up MPI compilers..."
export CC=$(which mpicc)
export FC=$(which mpif90)
export CXX=$(which mpic++)
export FF=$(which mpif77)
export MPI_CXX=$(which mpic++)

# Checking environment
echo "Listing loaded modules:"
module list
echo "Checking compiler paths..."
echo "C compiler (mpicc) located at: $(which mpicc)"
echo "C++ compiler (mpic++) located at: $(which mpic++)"
echo "Fortran compiler (mpif90) located at: $(which mpif90)"
echo "CMAKE located at: $(which cmake)"

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

remove_if_exists $ASPECT_BUILD_DIR
echo "Creating ASPECT build directory ${ASPECT_BUILD_DIR}"
mkdir -p "${ASPECT_BUILD_DIR}"

cd ${ASPECT_BUILD_DIR}

echo "Configuring ASPECT with CMake..."
echo "Use dealii ${INSTALL_PREFIX}"
cmake -DDEAL_II_DIR="${INSTALL_PREFIX}" -DWORLD_BUILDER_SOURCE_DIR="${WORLD_BUILDER_DIR}" ..
quit_if_fail "Error: Configuring ASPECT repository failed. Terminating job."

echo "Compiling ASPECT in directory ${ASPECT_BUILD_DIR}..."
Lib_build_aspect debugrelease main_collision 0
# make -j $SLURM_NTASKS
quit_if_fail "Error: Building ASPECT repository failed. Terminating job."

[[ -e ${ASPECT_BUILD_DIR}/aspect ]] && echo "ASPECT build completed successfully." || echo "ASPECT failed to build."
~             
