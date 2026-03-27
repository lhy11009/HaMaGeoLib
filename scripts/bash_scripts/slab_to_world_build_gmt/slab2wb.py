#!/Users/billen/opt/miniconda3/bin/python

# This script assumes that you have already used GMT scripts and the mapproject function 
# to extract the dip and depth from Slab2 along a 2D profiles    
# 1. run_trench_profs.sh -> Runs get_trench_perp_data.gmt
# 2. run_slab2_profs.sh  outputs profile lon, lat, distance, depth, dip    

import pygmt
import numpy as np

# Things to Change for different users of this script
ddir = "/home/lochy/ASPECT_PROJECT/HaMaGeoLib/dtemp/Slab2WorldBuilder/Slab2WorldBuilder/LocationProfsTrench/"
region = "Kuriles"    # to-do, at step 2 above, use slab2 3-letter Location, input variable?
profnum = "5"			  # to-do, make this input variable?

thickness = 200  # km  slab thickness

outfile = region + '_prof' + profnum + '_slab_wbsegments.txt'
# Slab 2 locations

loc = ['alu','cal','cam','car','cas','cot','hal','hel','him','hin','izu',
	'ker','kur','mak','man','mue','pam',
    'phi','png','puy','ryu','sam','sco','sol','sul','sum','van'];

# Load slab2 profile data: format is lon, lat, distance, depth, dip   
pfile = ddir + region + '_prof' + profnum + '.dat'
pdat = np.loadtxt(pfile,usecols=(0,1,2,3,4))

lon = pdat[:,0]
lat = pdat[:,1]
d = pdat[:,2]      # distance in degrees along profile
depth = pdat[:,3]
dip = pdat[:,4]

numpts = np.size(dip)
print(numpts)
 
# Create Basemap to show location of profile.  
# TO-DO Modify to set bounds of map to match extent of the slab and only plot the
# slab for this region.
   

# For each segment, calculate arc length 
km2m = 1000 # m/km
Re = 6371.137 # km
d2r = np.pi/180

# this will be a loop
n = range(0,numpts-1)
m = range(1,numpts)

# a. Calculate radius at point i (Re) and point i+1 (Re-depth)
r = Re - depth
		
# b. Convert to cartesian coordinates
# x(i) = r(i)*sin(d_angle)
# y(i) = r(i)*cos(d_angle)

x = r*np.sin(d*d2r)
y = r*np.cos(d*d2r)

print('depth, dist, dip, radius, x, y')
for i in range(numpts):
	print(depth[i], d[i], dip[i], r[i], x[i], y[i])
	

# c. Calculate the chord length from point i to point i+1 
# C = sqrt( (x(i+1)- x(i))^2 + (y(i+1)- y(i))^2  )

C = np.sqrt( (x[m] - x[n])**2 + (y[m] - y[n])**2)

print('C',C)

# d. Calculate the angle between point i and i+1
# theta = dip(i+1) - dip(i)

theta = (dip[m] - dip[n])*d2r

# e. Calculate radius of circle connecting both points
# R  = C/(2*sin(theta/2))
R = C/(2*np.sin(0.5*theta))

# f. Calculate arc length from point i to point i+1
# S = R*theta
S = R*theta

print("C, theta, R, S, dip1, dip2")
for i in range(numpts-1):
	print(C[i], theta[i], R[i],S[i],dip[i],dip[i+1])
	
# Write-out slab segment information Worldbuilder Segment format
#"segments":[
#             {"length":450e3, "thickness":[100e3], "angle":[20]},
#             {"length":450e3, "thickness":[100e3], "angle":[40]}
#           ],

f = open(outfile,'w')
line = ' "segments":[ \n'
f.write(line)

for i in range(numpts-1):
	arclen = "{:.3f}".format(S[i]) + 'e03'   # in meters
	thk = "{:.1f}".format(thickness) + 'e03' # in meteres
	dipn = "{:.3f}".format(dip[i])
	dipm = "{:.3f}".format(dip[i+1])

	line = ' {"length":' + arclen + ', "thickness":[' + thk + '], "angle":[' + dipn + ',' + dipm + ']}, \n'
	f.write(line)
	
line = ' ] \n'
f.write(line)	
f.close()
