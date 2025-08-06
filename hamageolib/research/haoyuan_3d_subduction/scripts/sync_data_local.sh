#!/bin/bash

case_name="EBA_2d_consistent_8_6/eba3d_width80_c22_AR4_yd300"

# from lochy2 to lochy
source_dir=/mnt/lochy2/ASPECT_DATA/ThDSubduction
target_dir=/mnt/lochy/ASPECT_DATA/ThDSubduction

# from lochy to lochy2
# source_dir=/mnt/lochy/ASPECT_DATA/ThDSubduction
# target_dir=/mnt/lochy2/ASPECT_DATA/ThDSubduction

# first make the target
target_dir="${target_dir}/${case_name}"
mkdir -p ${target_dir}

# now sync
echo "rsync -avu ${source_dir}/${case_name}/* ${target_dir}/"
rsync -avu ${source_dir}/${case_name}/* ${target_dir}/