#!/bin/bash
if [ "$#" -ne 0 ]; then
   echo " "
   echo "Usage: run_trench_profs.sh"
   exit
fi

sdir=/home/lochy/ASPECT_PROJECT/HaMaGeoLib/dtemp/Slab2WorldBuilder/Slab2WorldBuilder
magmin=$1

# Region, trench-file, flip-direction, profile-spacing, profile-length, dxp, dxm, dyp, dym, magmin  
${sdir}/get_trench_perp_data.gmt Kuriles PA-OKnorth 0 200 900 0.5 8.0 0.5 0.5

#${sdir}/get_trench_perp_data.gmt Japan PA-OKPS 0 200 1400 0.5 12.0 5.0 0.5 $magmin
#${sdir}/get_trench_perp_data.gmt Marianas PA-MA 0 200 800 0.5 5.0 0.5 0.5 $magmin
#${sdir}/get_trench_perp_data.gmt JavaSumatra AU-SU 1 200 800 3.0 0.5 2.8 0.5 $magmin
#${sdir}/get_trench_perp_data.gmt Tonga PA-TO 0 200 800 0.5 6.0 0.5 0.5 $magmin
#${sdir}/get_trench_perp_data.gmt Kermadec PA-KE 0 200 800 0.5 6.0 0.5 0.5 $magmin
#${sdir}/get_trench_perp_data.gmt Peru NZ-SAnorth 1 200 1200 5.0 0.5 0.5 0.5 $magmin
#${sdir}/get_trench_perp_data.gmt Chile NZ-SAsouth 1 200 1100 11.0 0.5 5.0 0.5 $magmin
#${sdir}/get_trench_perp_data.gmt TongaZ PA-TO 0 100 800 0.5 6.0 0.5 0.5 $magmin
#${sdir}/get_trench_perp_data.gmt KermZ PA-KE 0 100 800 0.5 6.0 0.5 0.5 $magmin