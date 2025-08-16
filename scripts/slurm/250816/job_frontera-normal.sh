#!/bin/bash -l
#SBATCH -N 8
#SBATCH -n 448
#SBATCH --threads-per-core=1
#SBATCH --tasks-per-node=56
#SBATCH -o task-%j.stdout
#SBATCH -e task-%j.stderr
#SBATCH -t 48:00:00
#SBATCH --partition=normal
#SBATCH --switches=1
#SBATCH --mail-user=hylli@ucdavis.edu
#SBATCH -A EAR23027
#SBATCH --job-name=eba3d_width51_c22_AR4

module load gcc/12.2.0
source /work2/06806/hyli/frontera/Softwares/dealii/dealii-9.5.1-Native-32bit-candi-gcc-12.2.0-impi-21.9.0-normal/configuration/enable.sh

>&2 echo "list of modules:"
>&2 module list
>&2 echo "aspect source: ${ASPECT_SOURCE_DIR}"

# Check if the "output" directory exists
if [ ! -d ./output ]; then
    # Create the "output" directory if it doesn't exist
    mkdir -p ./output
    echo "Directory 'output' created"
else
    echo "Directory 'output' already exisits"
fi

# Set file stripping
# 1. restart.mesh_fixed.data
if [ -f ./output/restart.mesh_fixed.data ]; then
    [[ ! -f ./dtemp ]] && mkdir -p ./dtemp
    echo "manage the old restart.mesh_fixed.data file and set stripping"
    cp ./output/restart.mesh_fixed.data ./dtemp/ && rm ./output/restart.mesh_fixed.data
    lfs setstripe -c 8 ./output/restart.mesh_fixed.data
    cp ./dtemp/restart.mesh_fixed.data ./output/
else
    echo "restart.mesh_fixed.data file not existing, set stripping"
    lfs setstripe -c 8 ./output/restart.mesh_fixed.data
fi
# 2. restart.mesh.new_fixed.data
if [ -f ./output/restart.mesh.new_fixed.data ]; then
    [[ ! -f ./dtemp ]] && mkdir -p ./dtemp
    echo "manage the old restart.mesh.new_fixed.data file and set stripping"
    cp ./output/restart.mesh.new_fixed.data ./dtemp/ && rm ./output/restart.mesh.new_fixed.data
    lfs setstripe -c 8 ./output/restart.mesh.new_fixed.data
    cp ./dtemp/restart.mesh.new_fixed.data ./output/
else
    echo "restart.mesh.new_fixed.data file not existing, set stripping"
    lfs setstripe -c 8 ./output/restart.mesh.new_fixed.data
fi
# 3. restart.mesh_fixed.data.old
if [ -f ./output/restart.mesh_fixed.data.old ]; then
    [[ ! -f ./dtemp ]] && mkdir -p ./dtemp
    echo "manage the old restart.mesh_fixed.data.old file and set stripping"
    cp ./output/restart.mesh_fixed.data.old ./dtemp/ && rm ./output/restart.mesh_fixed.data.old
    lfs setstripe -c 8 ./output/restart.mesh_fixed.data.old
    cp ./dtemp/restart.mesh_fixed.data.old ./output/
else
    echo "restart.mesh_fixed.data.old file not existing, set stripping"
    lfs setstripe -c 8 ./output/restart.mesh_fixed.data.old
fi

ibrun ${ASPECT_SOURCE_DIR}/build_master_TwoD/aspect case.prm
