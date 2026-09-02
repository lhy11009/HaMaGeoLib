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

# Path to candi (a script for installing deal.ii and dependencies)
INSTALL_PREFIX=/work2/06806/hyli/stampede3/Softwares/dealii/deal.ii-9.8.0-Native-32bits-candi-intel-26.0-impi-21.18-20260831
FASTSCAPE_BUILD_DIR="/work2/06806/hyli/stampede3/Softwares/fastscapelib-fortran/build"
ASPECT_INSTALL_DIR="/work2/06806/hyli/stampede3/Softwares/aspect"
WORLD_BUILDER_DIR="/work2/06806/hyli/stampede3/Softwares/WorldBuilder"
ASPECT_BRANCH="main_collision"

# Build options
# ASPECT_CMAKE_FLAGS - additional cmake flags
# DEBUG_USAGE - true or false, run make verbosely with 1 job.
DEBUG_USAGE=false
ASPECT_CMAKE_FLAGS=(
    "-DDEAL_II_CXX_FLAGS=-fno-finite-math-only"
    "-DASPECT_ADDITIONAL_CXX_FLAGS=-fno-finite-math-only"
)

# Loading modules
echo "Loading required modules and variables..."
source ~/env/aspect_env_intel26.sh
source ~/env/utilities.sh

# Checking environment
echo "Listing loaded modules:"
module list
echo "Checking compiler paths..."
echo "C compiler (mpicc) located at: ${CC}"
echo "C++ compiler (mpic++) located at: ${CXX}"
echo "Fortran compiler (mpif90) located at: ${FC}"
echo "CMAKE located at: $(which cmake)"
echo "Additional checking for icpx: $(which icpx)" # only check this with intel compiler

# Compile ASPECT
echo "Sourcing up DEAL.II configuration..."
source "$INSTALL_PREFIX/configuration/enable.sh"
echo "Checking compiler paths..."
echo "C compiler (mpicc) located at: ${CC}"
echo "C++ compiler (mpic++) located at: ${CXX}"
echo "Fortran compiler (mpif90) located at: ${FC}"
echo "CMAKE located at: $(which cmake)"
echo "Additional checking for icpx: $(which icpx)" # only check this with intel compiler

# Use World Builder if WORLD_BUILDER_DIR is provided.
if [[ -n "${WORLD_BUILDER_DIR:-}" ]]; then
    if [[ ! -d "${WORLD_BUILDER_DIR}" ]]; then
        echo "ERROR: WORLD_BUILDER_DIR does not exist or is not a directory:"
        echo "       ${WORLD_BUILDER_DIR}"
        exit 1
    fi

    ASPECT_CMAKE_FLAGS+=(
        "-DWORLD_BUILDER_SOURCE_DIR=${WORLD_BUILDER_DIR}"
    )
fi

# Enable FastScape if FASTSCAPE_BUILD_DIR is provided.
if [[ -n "${FASTSCAPE_BUILD_DIR:-}" ]]; then
    if [[ ! -d "${FASTSCAPE_BUILD_DIR}" ]]; then
        echo "ERROR: FASTSCAPE_BUILD_DIR does not exist or is not a directory:"
        echo "       ${FASTSCAPE_BUILD_DIR}"
        exit 1
    fi

    ASPECT_CMAKE_FLAGS+=(
        "-DASPECT_WITH_FASTSCAPE=ON"
        "-DFASTSCAPE_DIR=${FASTSCAPE_BUILD_DIR}"
    )
fi

# setting build directory
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

# check aspect branch
ASPECT_BRANCH="${ASPECT_BRANCH:-main}"  

if [ "$ASPECT_BRANCH" = "main" ]; then
	echo "Using default configuration for main branch."
	ASPECT_BUILD_DIR="${ASPECT_INSTALL_DIR}/build_20260831"
else
	echo "Using default configuration for $ASPECT_BRANCH branch."
	# echo "Error: You must specify configuration for branch '$ASPECT_BRANCH' explicitly. Then comment out the exit commend"
	# exit 1
	ASPECT_BUILD_DIR="${ASPECT_INSTALL_DIR}/build_$ASPECT_BRANCH"
fi

# by default, don't remove an existing directory
# remove_if_exists $ASPECT_BUILD_DIR # make build directory
mkdir -p "${ASPECT_BUILD_DIR}"
echo "Using existing ASPECT build directory ${ASPECT_BUILD_DIR}"
cd ${ASPECT_BUILD_DIR}

# checkout the required branch
git checkout $ASPECT_BRANCH 

# configuration
echo "Configuring ASPECT with CMake..."  
echo "Use dealii ${INSTALL_PREFIX}"

CMAKE_COMMAND=(
    cmake
    "${ASPECT_CMAKE_FLAGS[@]}"
    -DDEAL_II_DIR="${INSTALL_PREFIX}"
) # Base CMake command

CMAKE_COMMAND+=(..) # add source

"${CMAKE_COMMAND[@]}"  # run cmake
quit_if_fail "Error: Configuring ASPECT repository failed. Terminating job."

# compile
echo "Compiling ASPECT in directory ${ASPECT_BUILD_DIR}..." 
if [ "$DEBUG_USAGE" = true ]; then
    echo "Debug mode: verbose compilation with a single job"
    make VERBOSE=1 -j1
else
    make -j "$SLURM_NTASKS"
fi
quit_if_fail "Error: Building ASPECT repository failed. Terminating job."
