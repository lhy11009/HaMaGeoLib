#!/bin/bash
#SBATCH --account=billengrp          # SLURM account to use
#SBATCH --partition=p-billen         # Partition to submit the job
#SBATCH --time=24:00:00              # Max execution time (24 hours)
#SBATCH -N 1                         # Number of nodes
#SBATCH -n 64                        # Number of CPUs
#SBATCH --exclusive                  # Exclusive use of node (no sharing with other jobs)
#SBATCH --job-name=candi-toolchain   # Job name

#SBATCH --mail-type=ALL              # Notify user by email on all events
#SBATCH --mail-user=hylli@ucdavis.edu # Email address for notifications

#SBATCH --output=candi_output_%j.log # Redirect standard output to file
#SBATCH --error=candi_error_%j.log   # Redirect standard error to file

set -x  # Enable debug mode (prints commands before executing)

# Path to candi (a script for installing deal.ii and dependencies)
CANDI_DIR=/home/lochy/software/candi
INSTALL_PREFIX=/group/billengrp/Software/deal.ii/dealii-9.6.1-toolchain-gcc-13.2.0-openmpi5.0.5
PACKAGE_LIST="openblas sundials p4est trilinos dealii"

# Path to download, unpack and build. candi will use these path to build deal.ii
export DOWNLOAD_PATH=/tmp/src1
export UNPACK_PATH=/tmp/unpack1
export BUILD_PATH=/tmp/build1

# ASPECT_INSTALL_DIR="/group/billengrp/Software/aspect"

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

# Loading modules
echo "Loading required modules..."
module load openmpi/5.0.5
module load gcc/13.2.0


# Setting up MPI compilers
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
    yes | ./candi.sh --prefix=$INSTALL_PREFIX --packages="${PACKAGE_LIST}" -j $SLURM_NTASKS
    quit_if_fail "Error: candi.sh encountered an issue during installation. Terminating job."
    
    echo "Installation of dealii succeeds. Cleaning up directories..."
    rm -rf "$DOWNLOAD_PATH" "$UNPACK_PATH" "$BUILD_PATH"
    
    echo "Building of dealii successful"
fi

# Compile ASPECT
echo "Setting up ASPECT build directory..."
if [[ -z "${ASPECT_INSTALL_DIR}" ]]; then
    ASPECT_INSTALL_DIR="${INSTALL_PREFIX}/aspect"
    echo "ASPECT installation directory set to: ${ASPECT_INSTALL_DIR}"
fi

if [[ -d ${ASPECT_INSTALL_DIR} ]]; then
    echo "Retrieving existing ASPECT directory: ${ASPECT_INSTALL_DIR}"
    cd "${ASPECT_INSTALL_DIR}"
else
    echo "Creating ASPECT directory: ${ASPECT_INSTALL_DIR}"
    mkdir -p "${ASPECT_INSTALL_DIR}"
    cd "${ASPECT_INSTALL_DIR}"
fi

ASPECT_BUILD_DIR=${ASPECT_INSTALL_DIR}/build
remove_if_exists $ASPECT_BUILD_DIR
echo "Creating ASPECT build directory ${ASPECT_BUILD_DIR}"
mkdir -p "${ASPECT_BUILD_DIR}"

cd ${ASPECT_BUILD_DIR}

echo "Configuring ASPECT with CMake..."
echo "Use dealii ${INSTALL_PREFIX}"
cmake -DDEAL_II_DIR="${INSTALL_PREFIX}" ..
quit_if_fail "Error: Configuring ASPECT repository failed. Terminating job."

echo "Compiling ASPECT in directory ${ASPECT_BUILD_DIR}..."
make -j $SLURM_NTASKS
quit_if_fail "Error: Building ASPECT repository failed. Terminating job."

[[ -e ${ASPECT_BUILD_DIR}/aspect ]] && echo "ASPECT build completed successfully." || echo "ASPECT failed to build."