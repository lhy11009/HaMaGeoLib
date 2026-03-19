#!/bin/bash
#SBATCH --job-name=copy_selected_folders
#SBATCH --output=copy_selected_folders_%j.out
#SBATCH --error=copy_selected_folders_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=high
#SBATCH --account=billengrp

# A script for preparing Zenodo repository
# Source and destination directories
SRC_DIR="/nfs/peloton/waltz/billenlab/group/lochy/ThDSubduction"
DEST_DIR="/quobyte/billengrp/lochy/ThDSubduction/zenodo_repo"

CASE_DIR_TO_COPY=("chunk_geometry3/eba3d_1_SA50.0_OA20.0_width80_bw4000_sw1000_lwidth9" 
	"chunk_geometry1/eba3d_width80_bw4000_sw1000_yd500.0_AR4_1"
	"chunk_geometry3/eba3d_1_SA80.0_OA40.0_width61_bw4000_sw1000"
	"chunk_geometry3/eba3d_1_SA80.0_OA40.0_width51_bw4000_sw1000"
	"chunk_geometry3/eba3d_1_SA50.0_OA20.0_width80_bw4000_sw1000"
	"chunk_geometry3/eba3d_1_SA50.0_OA20.0_width51_bw4000_sw1000")


for CASE_DIR in "${CASE_DIR_TO_COPY[@]}"; do
    echo "  - $CASE_DIR"
    mkdir -p "${DEST_DIR}/${CASE_DIR}/output/solution"
    if [ -d "$SRC_DIR/${CASE_DIR}" ]; then
        rsync -avu ${SRC_DIR}/${CASE_DIR}/case.* ${DEST_DIR}/${CASE_DIR}/
        rsync -avu ${SRC_DIR}/${CASE_DIR}/*.sh ${DEST_DIR}/${CASE_DIR}/
        rsync -avu ${SRC_DIR}/${CASE_DIR}/output/log.* ${DEST_DIR}/${CASE_DIR}/output/
        rsync -avu ${SRC_DIR}/${CASE_DIR}/output/*statistic* ${DEST_DIR}/${CASE_DIR}/output/
        rsync -avu ${SRC_DIR}/${CASE_DIR}/output/solution.* ${DEST_DIR}/${CASE_DIR}/output/
        rsync -avu ${SRC_DIR}/${CASE_DIR}/output/solution/solution-00034.* ${DEST_DIR}/${CASE_DIR}/output/solution/
        rsync -avu ${SRC_DIR}/${CASE_DIR}/output/solution/solution-00044.* ${DEST_DIR}/${CASE_DIR}/output/solution/
        rsync -avu ${SRC_DIR}/${CASE_DIR}/output/solution/solution-00144.* ${DEST_DIR}/${CASE_DIR}/output/solution/
        rsync -avu ${SRC_DIR}/${CASE_DIR}/output/solution/solution-00154.* ${DEST_DIR}/${CASE_DIR}/output/solution/
    else
        echo "Warning: $SRC_DIR/${CASE_DIR} does not exist"
    fi
done
