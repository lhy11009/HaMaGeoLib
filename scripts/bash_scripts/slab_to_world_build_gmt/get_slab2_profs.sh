#!/bin/bash

set -e
#set -x   # uncomment if you want verbose execution like `-v`

# Check arguments
if [ "$#" -eq 0 ]; then
   echo " "
   echo "Usage: get_slab2_profs.gmt region trfile dipgrd depthgrd dphi length-deg"
   echo "trfile is the file with the trench lon-lat values"
   echo "dipfile is the file with the slab dips from Slab 2.0"
   exit 1
fi

# GMT Parameters
gmt gmtset FONT_LABEL 10p,Helvetica,black FONT_TITLE 10p,Helvetica,black
gmt gmtset FONT_ANNOT_PRIMARY 10p,Helvetica,black
gmt gmtset FONT_ANNOT_SECONDARY 10p,Helvetica,black
gmt gmtset MAP_FRAME_TYPE plain MAP_TICK_LENGTH_PRIMARY 0.05i
gmt gmtset PS_MEDIA letter MAP_TITLE_OFFSET 2p MAP_ANNOT_OFFSET_PRIMARY 2p
gmt gmtset PROJ_LENGTH_UNIT INCH MAP_LABEL_OFFSET 2p
gmt gmtset MAP_GRID_PEN_PRIMARY 0.5p,GRAY74
gmt gmtset FORMAT_GEO_MAP +D

# Constants
rad2deg=$(gmt gmtmath -Q 180 PI DIV =)

# Directories
slab2dir="/home/lochy/ASPECT_PROJECT/HaMaGeoLib/dtemp/Slab2WorldBuilder/Slab2WorldBuilder/Slab2Distribute_Mar2018"
trdir="/home/lochy/ASPECT_PROJECT/HaMaGeoLib/dtemp/Slab2WorldBuilder/Slab2WorldBuilder/LocationProfsTrench"
outdir="/home/lochy/ASPECT_PROJECT/HaMaGeoLib/dtemp/Slab2WorldBuilder/Slab2WorldBuilder/LocationProfsTrench"

# Inputs
region="$1"
dipgrd="${slab2dir}/${2}_dip_02.24.18.grd"
depthgrd="${slab2dir}/${2}_dep_02.24.18.grd"
dphi="$3"
lmax="$4"

trdata="${trdir}/${region}_profs.dat"

# Cross section settings
lmin=0

############################
# Number of profiles
nprof=$(wc -l < "$trdata")
echo "$nprof"

i=1

while [ "$i" -le "$nprof" ]; do
    echo "profile $i of $nprof"

    profile="${outdir}/${region}_prof${i}.dat"

    # Extract profile center + azimuth
    clon=$(awk -v i="$i" 'NR==i {print $1}' "$trdata")
    clat=$(awk -v i="$i" 'NR==i {print $2}' "$trdata")
    azm=$(awk -v i="$i" 'NR==i {print $4}' "$trdata")

    # Project dip data
    gmt project -C${clon}/${clat} -A${azm} -L0/10 -G0.1 -V > temp1

    awk '{if($1 < 0) {print 360+$1, $2, $3} else {print $1, $2, $3}}' temp1 > temp2

    gmt grdtrack temp2 -G"$depthgrd" > temp3
    gmt grdtrack temp3 -G"$dipgrd" > temp4

    # Filter NaN and flip depth sign
    awk '($4!="NaN") {print $1, $2, $3, -$4, $5}' temp4 > "$profile"

    ((i++))
done

rm -f temp1 temp2 temp3 temp4