#!/bin/bash
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
CANDI_DIR=/home/lochy/software/candi # Path to candi (a script for installing deal.ii and dependencies)
INSTALL_PREFIX=/quobyte/billengrp/Software/deal.ii/deal.ii-9.6.1-toolchain-gcc-13.2.0-openmpi5.0.5
PACKAGE_LIST="openblas sundials p4est trilinos dealii"
ASPECT_INSTALL_DIR="/quobyte/billengrp/lochy/Software/aspect"
WORLD_BUILDER_DIR="/quobyte/billengrp/lochy/Software/WorldBuilder"

TASK="aspect" # Different tasks
# choices:
# build_all - build deal.ii, configure and build aspect (recommand starting with the main branch)
# aspect - only configure and build aspect
# aspect_build_only - using existing build directory and build

ASPECT_BRANCH="master_TwoD_rebase" # Choices for aspect
#ASPECT_BRANCH="main"
ASPECT_MODE="debug"
#ASPECT_MODE="debugrelease"
# choices of ASPECT_MODE:
# debugrelease (default), the build directory is named as "build_debugrelease"
# release, the build directory is named as "build_release"
# debug, the build directory is named as "build_debug"

# expicitly set this up for my branch
# source /work2/06806/hyli/frontera/Softwares/aspectLib/enable.sh

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

if [ "$TASK" = "build_all" ]; then
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
else
	echo "Skip building dealii"
fi

# Compile ASPECT
echo "Sourcing up DEAL.II configuration..."
source "$INSTALL_PREFIX/configuration/enable.sh"
export ASPECT_SOURCE_DIR="$ASPECT_INSTALL_DIR"
echo "Setting up ASPECT build directory..."
if [[ -z "${ASPECT_INSTALL_DIR}" ]]; then
    ASPECT_INSTALL_DIR="${INSTALL_PREFIX}/aspect"
    echo "ASPECT installation directory set to: ${ASPECT_INSTALL_DIR}"
fi

if [[ -d ${ASPECT_INSTALL_DIR} ]]; then
    echo "Retrieving existing ASPECT directory: ${ASPECT_INSTALL_DIR}"
else
    echo "Creating ASPECT directory: ${ASPECT_INSTALL_DIR}"
    mkdir -p "${ASPECT_INSTALL_DIR}"
fi

cd "${ASPECT_INSTALL_DIR}" # go into the "aspect" directory

ASPECT_BRANCH="${ASPECT_BRANCH:-main}"  # check aspect branch

echo "Checkout branch" # checkout the branch
git checkout $ASPECT_BRANCH # checkout the branch
quit_if_fail "Error checking out branch $ASPECT_BRANCH"

if [ "$ASPECT_BRANCH" = "main" ]; then
	echo "Using default configuration for main branch."
	ASPECT_BUILD_DIR="${ASPECT_INSTALL_DIR}/build"
else
	echo "Using default configuration for $ASPECT_BRANCH branch."
	# echo "Error: You must specify configuration for branch '$ASPECT_BRANCH' explicitly. Then comment out the exit commend"
	# exit 1
	ASPECT_BUILD_DIR="${ASPECT_INSTALL_DIR}/build_${ASPECT_BRANCH}"
fi

if [ "$ASPECT_MODE" = "release" ]; then
	echo "Building in release mode"
else
	echo "Building in $ASPECT_MODE mode"
	ASPECT_BUILD_DIR="${ASPECT_BUILD_DIR}_${ASPECT_MODE}"
fi


if [ "$TASK" = "aspect_build_only" ]; then
	echo "Using existing ASPECT build directory ${ASPECT_BUILD_DIR}"
	[ -d "$ASPECT_BUILD_DIR" ] || { echo "Error: Directory '$ASPECT_BUILD_DIR' does not exist." >&2; exit 1; }
else
	remove_if_exists $ASPECT_BUILD_DIR # make build directory
	echo "Creating ASPECT build directory ${ASPECT_BUILD_DIR}"
	mkdir -p "${ASPECT_BUILD_DIR}"
fi
quit_if_fail "Error: Setting ASPECT build directory failed. Terminating job."

cd ${ASPECT_BUILD_DIR} # go into the "aspect/build" directory

if [ "$TASK" = "aspect_build_only" ]; then
	echo "Skip ASPECT configuration"
else
	echo "Configuring ASPECT with CMake..."  # configuration
        echo "Use dealii ${INSTALL_PREFIX}"
        if [ -z "$WORLD_BUILDER_DIR" ]; then
        	echo "Use built-in world builder"
        	cmake -DDEAL_II_DIR="${INSTALL_PREFIX}" ..
        elif [ ! -d "$WORLD_BUILDER_DIR" ]; then
        	echo "Error: WORLD_BUILDER_DIR is set to '$WORLD_BUILDER_DIR', but this is not a valid directory."
        	exit 1
        else
        	echo "Using World Builder directory: $WORLD_BUILDER_DIR"
        	cmake -DDEAL_II_DIR="${INSTALL_PREFIX}" -DWORLD_BUILDER_SOURCE_DIR="$WORLD_BUILDER_DIR" ..
        fi
        quit_if_fail "Error: Configuring ASPECT repository failed. Terminating job."
fi

make ${ASPECT_MODE}  # mode of aspect

echo "Compiling ASPECT in directory ${ASPECT_BUILD_DIR}..." # compile
if  [ "$ASPECT_BRANCH" = "master_TwoD_rebase" ]; then
	if [ "$ASPECT_MODE" = "release" ]; then
		Lib_build_aspect ${ASPECT_MODE} master_TwoD_rebase 0
	else
		Lib_build_aspect ${ASPECT_MODE} master_TwoD_rebase 0 $ASPECT_MODE
	fi
else
	make -j $SLURM_NTASKS
fi
quit_if_fail "Error: Building ASPECT repository failed. Terminating job."

[[ -e ${ASPECT_BUILD_DIR}/aspect ]] && echo "ASPECT build completed successfully." || echo "ASPECT failed to build."
