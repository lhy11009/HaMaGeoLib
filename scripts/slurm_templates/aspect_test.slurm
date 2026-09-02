#!/bin/bash
#SBATCH --account billengrp          # SLURM account to use
#SBATCH --partition p-billen              # Partition to submit the job
#SBATCH --time 24:00:00                # Max execution time (24 hours)
#SBATCH -N 1                           # Number of nodes
#SBATCH -n 64                          # Number of cpus
#SBATCH --exclusive                    # Exclusive use of node (no sharing with other jobs)
#SBATCH --job-name=aspect_test               # Job name

#SBATCH --mail-type=ALL                 # Notify user by email on all events
#SBATCH --mail-user=hylli@ucdavis.edu # Email address for notifications

#SBATCH --output=aspect_test_output_%j.log       # Redirect standard output to file
#SBATCH --error=aspect_test_error_%j.log         # Redirect standard error to file

set -x	# Enable debug mode (prints commands before executing)

ASPECT_DIR=/group/billengrp/Software/deal.ii/dealii-9.6.1-toolchain-gcc-13.2.0-openmpi5.0.5/aspect
test_file_path=$ASPECT_DIR/cookbooks/convection_box_3d/convection_box_3d.prm
ASPECT_EXECUTABLE=$ASPECT_DIR/build/aspect-release

if [[ -f "$test_file_path" && -f "$ASPECT_EXECUTABLE" ]]; then
	echo "Both files exist."
else
	echo "One or both files are missing."
	exit 1
fi

module load openmpi/5.0.5

cp ${test_file_path} .

>&2 echo "list of modules:"
>&2 module list
>&2 echo "aspect source: ${ASPECT_SOURCE_DIR}"

srun ${ASPECT_EXECUTABLE} ./convection_box_3d.prm
