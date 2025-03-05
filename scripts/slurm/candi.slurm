#!/bin/bash
#SBATCH --account billengrp          # SLURM account to use
#SBATCH --partition p-billen              # Partition to submit the job
#SBATCH --time 24:00:00                # Max execution time (24 hours)
#SBATCH -N 1                           # Number of nodes
#SBATCH -n 64                          # Number of cpus
#SBATCH --exclusive                    # Exclusive use of node (no sharing with other jobs)
#SBATCH --job-name=candi-toolchain               # Job name

#SBATCH --mail-type=ALL                 # Notify user by email on all events
#SBATCH --mail-user=hylli@ucdavis.edu # Email address for notifications

#SBATCH --output=candi_output_%j.log       # Redirect standard output to file
#SBATCH --error=candi_error_%j.log         # Redirect standard error to file

set -x	# Enable debug mode (prints commands before executing)

# Path to candi (a script for installing deal.ii and dependencies)
CANDI_DIR=/group/billengrp/Software/candi
INSTALL_PREFIX=/group/billengrp/Software/deal.ii/dealii-9.6.1-toolchain-gcc-13.2.0-openmpi5.0.5
PACKAGE_LIST="openblas sundials hdf5 p4est trilinos dealii"
ASPECT_INTALL_PREFIX=""

echo "Loading required modules..."
module load openmpi/5.0.5
module load gcc/13.2.0

echo "Listing loaded modules:"
module list

# Setting up MPI compilers
echo "Setting up MPI compilers..."
export CC=`which mpicc`
export FC=`which mpif90`
export CXX=`which mpic++`
export FF=`which mpif77`
export MPI_CXX=`which mpic++`

echo "Checking compiler paths..."
echo "C compiler (mpicc) located at: $(which mpicc)"
echo "C++ compiler (mpic++) located at: $(which mpic++)"
echo "Fortran compiler (mpif90) located at: $(which mpif90)"

# build hdf5, p4est, trilinos, dealii:
# Build necessary dependencies for deal.II using candi
echo "Starting the build process for selected packages"

cd ${CANDI_DIR}
yes | ./candi.sh --prefix=$INSTALL_PREFIX --packages="${PACKAGE_LIST}" -j $SLURM_NTASKS

if [[ $? -ne 0 ]]; then
# Capture the exit status of the last command
	    echo "Error: candi.sh encountered an issue during installation. Terminating job." >&2
	        exit 1  # Terminate the job with an error code
fi

echo "Build process completed"


# compile aspect
echo "Setting up ASPECT build directory..."
if [[ -z "${ASPECT_INTALL_PREFIX}" ]]; then
	    ASPECT_BUILD_DIR="${DEALII_DIR}/aspect"
    else
	        ASPECT_BUILD_DIR="${ASPECT_INTALL_PREFIX}/aspect"
fi
echo "ASPECT build directory set to: ${ASPECT_BUILD_DIR}"

if [[ -d ${ASPECT_BUILD_DIR} ]]; then
	    echo "Removing existing ASPECT build directory: ${ASPECT_BUILD_DIR}"
	        rm -rf "${ASPECT_BUILD_DIR}"
fi

echo "Creating ASPECT build directory: ${ASPECT_BUILD_DIR}"
mkdir -p "${ASPECT_BUILD_DIR}"
cd "${ASPECT_BUILD_DIR}"

echo "Cloning ASPECT repository..."
git clone git@github.com:geodynamics/aspect.git

echo "Creating ASPECT build directory..."
mkdir -p "aspect/build"
cd "aspect/build"

echo "Configuring ASPECT with CMake..."
cmake -D DEAL_II_DIR="${INSTALL_PREFIX}" ..

echo "Compiling ASPECT..."
make -j 32

echo "ASPECT build completed successfully."

