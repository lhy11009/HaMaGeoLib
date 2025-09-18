#!/bin/bash

# path to
case_dir="/mnt/lochy/ASPECT_DATA/ThDSubduction/EBA_2d_consistent_11/eba3d_1_SA50.0_OA20.0_width80_bw8000_sw2000"

# steps to run (vtu_step)
min_vtu_step=130
max_vtu_step=197

# total number of pieces
n_pieces=16

# cd into work directory
PWD="/home/lochy/ASPECT_PROJECT/HaMaGeoLib"
cd "$PWD"

# activate conda environment
# conda activate hmgeolib

# run python script for all the pieces
for ((vtu_step=min_vtu_step; vtu_step<=max_vtu_step; vtu_step++)); do
    for ((i_piece=0; i_piece<n_pieces; i_piece++)); do
        echo "i_piece=$i_piece"
        command="python hamageolib/research/haoyuan_3d_subduction/scripts/SlabMorphology.py -m piece-bash -d ${case_dir} -s ${vtu_step} -n ${n_pieces} -i ${i_piece}"
        echo "$command"
        eval "$command"
        echo ''
    done

    # assemble and export files (using i_piece=-1)
    i_piece=-1    
    command="python hamageolib/research/haoyuan_3d_subduction/scripts/SlabMorphology.py -m piece-bash -d ${case_dir} -s ${vtu_step} -n ${n_pieces} -i ${i_piece}"
    echo "$command"
    eval "$command"
done
