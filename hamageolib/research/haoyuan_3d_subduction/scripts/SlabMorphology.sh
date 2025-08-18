#!/bin/bash

case_dir="/mnt/lochy/ASPECT_DATA/ThDSubduction/chunk_geometry2/eba3d_SA50.0_OA20.0_width80_bw4000_sw1000_yd500.0_AR4"
vtu_step=10

# cd into work directory
PWD="/home/lochy/ASPECT_PROJECT/HaMaGeoLib"
cd "$PWD"

# activate conda environment
# conda activate hmgeolib

# run python script for all the pieces
n_pieces=2
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
