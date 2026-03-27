#!/bin/csh 
if ($#argv == 0) then
   echo " "
   echo "Usage: run_eq_dip_depth_profs.sh magmin"
   
   exit
endif

set sdir = /Users/billen/Box-Sync/SZ-Earthquake-Profiles/Scripts
set magmin = $1

{$sdir}/get_eq_dip_depth_profs.gmt Kuriles PA-OKnorth kur_izu_slab2 200 900 $magmin
#{$sdir}/get_eq_dip_depth_profs.gmt Japan PA-OKPS kur_izu_slab2 200 1400 $magmin
#{$sdir}/get_eq_dip_depth_profs.gmt Marianas PA-MA kur_izu_slab2 200 800 $magmin
#{$sdir}/get_eq_dip_depth_profs.gmt JavaSumatra AU-SU sum_slab2 200 800 $magmin
#{$sdir}/get_eq_dip_depth_profs.gmt Tonga PA-TO ker_slab2 200 800 $magmin
#{$sdir}/get_eq_dip_depth_profs.gmt Kermadec PA-KE ker_slab2 200 800 $magmin
#{$sdir}/get_eq_dip_depth_profs.gmt Peru NZ-SAnorth sam_slab2 200 1200 $magmin
#{$sdir}/get_eq_dip_depth_profs.gmt Chile NZ-SAsouth sam_slab2 200 1100 $magmin
#{$sdir}/get_eq_dip_depth_profs.gmt TongaZ PA-TO ker_slab2 100 800 $magmin
#{$sdir}/get_eq_dip_depth_profs.gmt KermZ PA-KE ker_slab2 100 800 $magmin