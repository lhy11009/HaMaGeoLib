#!/opt/miniconda3/bin/python

# Magali Billen, October 2023
# Utility functions for plotting Aspect VTK model output

# Import Libraries 
import vtk
import os.path  
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy
import pygmt

# Convert time-step number to string with leading zeros to match aspect filename format
def get_timestep_str(timestep, iN):

    str_timestep = str(timestep)
    N = iN - len(str_timestep)
    x = '0'*N
    filenum = x + str_timestep
    #print('Timestep:', filenum)
    
    return filenum

# Opens the vtu file and reads in the contents
# Puts the grid point data into the variable point1
# - in x, y, z coordinates even if mesh is a 2D cylinder
# reader: the object with the information from the file, so it has all the field data

def read_aspect_vtk(indir, timestep, iN):
    
    filenum = get_timestep_str(timestep, iN)
    filein = indir + "/solution/solution-" + filenum + ".pvtu" 
    assert(os.path.isfile(filein))
    
    reader = vtk.vtkXMLPUnstructuredGridReader()
    reader.SetFileName(filein)
    reader.Update()
    grid = reader.GetOutput()
    
    points = grid.GetPoints()
    point1 = vtk_to_numpy(points.GetData())
    y = point1[:,0]
    x = point1[:,1]
    
    return reader, x, y
    
# Even if geometry is chunk or spherical, data is held in a cartesian grid
# converts 2D cartesian to 2D spherical
# radius in km, longitude or latitude degrees

def convert_to_2Dspherical(x,y):    
    # create empty matrices that are the same size/shape as x and y
    r = np.zeros(np.shape(x))
    theta = np.zeros(np.shape(x))
    
    km2m = 1000 # m/km
    r = np.sqrt(np.power(x, 2) + np.power(y, 2))/km2m  # km
    theta = (180/np.pi)*np.arctan2(y,x)  # degrees
    return r, theta
 
# Sets region bounds for 2D plot (cartesian box or polar)
# Needs mid to rotate the data so the middle of the plot is at the top center.
# Spherical: x - longitude, z - radius
# mid: needed for polar slice plot of spherical data
#      rotates the data so the middle of the plot is at the top center.
   
def define_region(z,x):
    x_min = round(x.min(),0)
    x_max = round(x.max(),0)
    z_min = round(z.min())
    z_max = round(z.max())
      
    region = [x_min, x_max, z_min, z_max]
    mid = round(x.min() + 0.5*(x.max() - x.min()),1)
    
    return region, mid
    
# Interpolate field data on to a refined uniform mesh
# 2D Spherical: x - longitude/latitude, z - radius 
# data_str = "T", "viscosity","density, "strain-rate", "szcrust", etc...

def get_data_on_grid(reader,data_str,z,x,dz,dx,this_region):
    data_set = reader.GetOutputAsDataSet()
    point_data = data_set.GetPointData()
    # print(point_data)  # to view types & names of data sets
    temp_array = vtk_to_numpy(point_data.GetArray(data_str)) #getting the desired variable from the point datas
    
    if data_str == 'viscosity':        
        temp_array = np.log10(temp_array)
     
    data = np.zeros((np.size(temp_array),3))
    data[:,0] = x
    data[:,1] = z
    data[:,2] = temp_array
    
    # combines data closer than dx,dz
    data_mean = pygmt.blockmean(data, region=this_region, spacing=[str(dx),str(dz)])
    # other options of possible use: tension, upper, lower
    data_grid = pygmt.surface(data_mean, region=this_region, spacing=[str(dx),str(dz)])
    return data_grid
    

# Convert velocity components into length and angle from horizontal.
# Since data is in x and z components, with units of m/yr or m/s, 
# there's no difference for a 2D cartesian versus 2D spherical reader.
# Will need a different reader for 3D (need to project into cross section for 2D plotting)
# Will need to add options to interpolate to uniform grid, then sample evenly, and
# and then return velocity components

def get_2Dvelocity(reader,data_str):
    
    data_set = reader.GetOutputAsDataSet()
    point_data = data_set.GetPointData()
    tmp_array = vtk_to_numpy(point_data.GetArray(data_str))
    n = np.size(tmp_array)
    
    npts = int(n/3)
    tmp_velo = np.reshape(tmp_array,(npts,3))

    # angle counterclockwise from horizontal 
    r2d = 180/np.pi
    tmp_angle =  np.arctan2(tmp_velo[:,1],tmp_velo[:,0])*r2d 
    
    # Length of the vector
    tmp_length = np.sqrt(tmp_velo[:,0]*tmp_velo[:,0] + tmp_velo[:,1]*tmp_velo[:,1])
    
    
    return tmp_angle, tmp_length