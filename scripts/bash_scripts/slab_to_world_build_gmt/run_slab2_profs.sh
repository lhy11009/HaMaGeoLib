#!/bin/bash

# Exit if any command fails (optional but recommended)
set -e

# Check arguments
if [ "$#" -ne 0 ]; then
   echo " "
   echo "Usage: run_slab2_profs.sh"
   exit 1
fi

# Directory
sdir="/home/lochy/ASPECT_PROJECT/HaMaGeoLib/dtemp/Slab2WorldBuilder/Slab2WorldBuilder"

# Run command
"${sdir}/get_slab2_profs.sh" Kuriles kur_slab2 0.1 10

#"${sdir}/get_slab2_profs.sh" Tonga ker_slab2 0.1 10
#"${sdir}/get_slab2_profs.sh" Japan kur_slab2 0.1 10
#"${sdir}/get_slab2_profs.sh" Marianas izu_slab2 0.1 10
#"${sdir}/get_slab2_profs.sh" JavaSumatra sum_slab2 0.1 10
#"${sdir}/get_slab2_profs.sh" Tonga ker_slab2 0.1 10
#"${sdir}/get_slab2_profs.sh" Kermadec ker_slab2 0.1 10
#"${sdir}/get_slab2_profs.sh" Peru sam_slab2 0.1 10
#"${sdir}/get_slab2_profs.sh" Chile sam_slab2 0.1 10
#"${sdir}/get_slab2_profs.sh" TongaZ ker_slab2 0.1 10
#"${sdir}/get_slab2_profs.sh" KermZ ker_slab2 0.1 10