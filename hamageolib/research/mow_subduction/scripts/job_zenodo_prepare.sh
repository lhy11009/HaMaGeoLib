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
SRC_DIR="/mnt/lochy/ASPECT_DATA/MOW"  # mow case, 3d
CASE_DIR_TO_COPY=("mow3_00/C_mow_h2890.0_M_gr3_ar4_rf" "mow3_00/C_mow_h2890.0_M_gr3_ar4_rf_gz") # mow case, 3d
# CASE_DIR_TO_COPY=("mow01/C_mow_h2890.0_gr3_ar4" "mow01/C_mow_h2890.0_M_gr3_ar4" "mow01/C_mow_h2890.0_M_gr3_ar4_gz_2") # mow case, 2d


# SRC_DIR="/mnt/lochy/ASPECT_DATA/ThDSubduction"  # non-mow case, 3d
# CASE_DIR_TO_COPY=("EBA_2d_consistent_8_6/eba3d_width80_c22_AR4_yd300")

DEST_DIR="/mnt/lochy/ASPECT_DATA/MOW/zenodo_repo"



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
        rsync -avu ${SRC_DIR}/${CASE_DIR}/output/solution/solution-00124.* ${DEST_DIR}/${CASE_DIR}/output/solution/
        rsync -avu ${SRC_DIR}/${CASE_DIR}/output/solution/solution-00204.* ${DEST_DIR}/${CASE_DIR}/output/solution/
    else
        echo "Warning: $SRC_DIR/${CASE_DIR} does not exist"
    fi
done
