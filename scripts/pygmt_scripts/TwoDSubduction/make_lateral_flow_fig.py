#!/opt/miniconda3/bin/python

# Magali Billen, October 2023
# this script uses
# DATA_OUTPUT_DIR: case output directory

# Import Libraries
import vtk
import os.path  
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy
import pygmt



def plot_step(infile_dir, timestep):
    '''
    Plot time step
    '''
    mod_name = "2dmod"
    
    pygmt_img_dir = os.path.join("IMG_OUTPUT_DIR", "pygmt_output")
    if not os.path.isdir(pygmt_img_dir):
        os.mkdir(pygmt_img_dir)

    # Read in data, convert to spherical and get full plotting region
    reader, x, y = read_aspect_vtk(infile_dir, timestep, INITIAL_ADAPTIVE_REFINEMENT)  # fields, m, m
    r, theta = convert_to_2Dspherical(x,y)   # km, degrees
    region1, mid1 = define_region(r,theta)   # full size of data for interpolating
    
    # choose grid size for interpolated grid, should match smallest mesh size in grid
    dlon = 0.1 # degrees
    drad = 10 # km
    
    # For plotting region smaller than model size, choose plot limits
    lonmin = 40
    lonmax = 60
    zmax = 1200
    erad = 6371  # km
    mid2 = lonmin + 0.5*(lonmax - lonmin)
    region2 = [lonmin, lonmax,erad-zmax,erad]
    
    # make just slightly larger than plotting region so will be smooth at edges
    region1 = [lonmin-1, lonmax+1,erad-zmax-100,erad+100]
    
    # For each field data set that you want to plot, interpolate onto regular grid
    T = get_data_on_grid(reader,'T', r, theta, drad, dlon, region1)
    density = get_data_on_grid(reader,'density', r, theta, drad, dlon, region1)
    visclog = get_data_on_grid(reader,'viscosity', r, theta, drad, dlon, region1)
    
    # Does not interpolate
    velo_angle, velo_length = get_2Dvelocity(reader,'velocity')
    m2cm = 100 # cm/m
    velmax = 0.05 * m2cm
    # velmax = velo_length.max()*m2cm  # cm/yr 
    print('Max length = ', velmax, ' cm/yr')
    #Scale vector: for movies or comparing different time-steps... will want to fix this scaling.
    velo_length = velo_length/velo_length.max()

    # plot the figres
    lonmin = 40
    lonmax = 60
    zmax = 1200
    erad = 6371  # km
    mid2 = lonmin + 0.5*(lonmax - lonmin)
    region2 = [lonmin, lonmax,erad-zmax,erad]
    
    pygmt.config(FONT_LABEL='Times-Roman,10p')
    pygmt.config(FONT_LOGO='Times-Roman,10p')
    
    proj2 = 'P10c+a+t' + str(mid2) + '+z'
    frame2 = ["xa5f","ya200f","WNse+gbisque",] 

    # 1. plot density 
    fig = pygmt.Figure() 
    fig.basemap(region=region2, projection=proj2, frame=frame2)
    pygmt.makecpt(cmap='davos',reverse=True, series=[3025,4225])
    fig.grdimage(density)
    # fig.grdcontour(T,interval=300,limit=[600,2400])
    fig.colorbar(frame=["a200f100", "x+lDensity", "y+lkg/m@+3@+"])
    
    npts = 321 # plot every nth point
    sc = 0.5  # scale vector lengths
    style_w = "v0.08c+e+a40+gwhite+h0+p0.5p,white"
    fig.plot(x=theta[0::npts],y=r[0::npts],direction=[velo_angle[0::npts]+mid2,velo_length[0::npts]*sc], style=style_w, pen="0.5p,white")
    
    # Plot a velocity vector to show scaling
    style_b = "v0.08c+e+a40+gwhite+h0+p0.5p,black" 
    vtext = 'v@-0@- = ' + str(round(velmax,2)) + "cm/yr"
    yvel = erad-zmax-80
    fig.plot(x=[mid2-5, mid2-5], y=[yvel,yvel], direction=[[0,0], [1*0.5, 1*0.5]], style=style_b, pen="0.5p,black", no_clip=True)
    fig.text(text=vtext, x=mid2, y=yvel, font="10p,Times-Roman,black", no_clip=True)
    
    fig.show()
    pdffile_name = mod_name + '_s%04d' % timestep + '_density.pdf'
    pdffile = os.path.join(pygmt_img_dir, pdffile_name)
    print('Figure saved to ', pdffile)
    fig.savefig(pdffile)
    
    # 2. plot viscosity
    fig = pygmt.Figure() 
    fig.basemap(region=region2, projection=proj2, frame=frame2)
    pygmt.makecpt(cmap='roma',reverse=False, series=[18, 24])
    fig.grdimage(visclog)
    # fig.grdcontour(T,interval=300,limit=[600,2400])
    fig.colorbar(frame=["a1f0.5", "x+lViscosity", "y+lPas+"])
    
    npts = 321 # plot every nth point
    sc = 0.5  # scale vector lengths
    style_w = "v0.08c+e+a40+gwhite+h0+p0.5p,white"
    fig.plot(x=theta[0::npts],y=r[0::npts],direction=[velo_angle[0::npts]+mid2,velo_length[0::npts]*sc], style=style_w, pen="0.5p,white")
    
    # Plot a velocity vector to show scaling
    style_b = "v0.08c+e+a40+gwhite+h0+p0.5p,black" 
    vtext = 'v@-0@- = ' + str(round(velmax,2)) + "cm/yr"
    yvel = erad-zmax-80
    fig.plot(x=[mid2-5, mid2-5], y=[yvel,yvel], direction=[[0,0], [1*0.5, 1*0.5]], style=style_b, pen="0.5p,black", no_clip=True)
    fig.text(text=vtext, x=mid2, y=yvel, font="10p,Times-Roman,black", no_clip=True)
    
    fig.show()
    pdffile_name = mod_name + '_s%04d' % timestep + '_viscosity.pdf'
    pdffile = os.path.join(pygmt_img_dir, pdffile_name)
    print('Figure saved to ', pdffile)
    fig.savefig(pdffile)

def main():
    '''
    the main function
    '''
    # Enter information for this particular model and time-step
    infile_dir = "DATA_OUTPUT_DIR"
    timesteps = GRAPHICAL_STEPS
    
    for timestep in timesteps:
        plot_step(infile_dir, timestep)

main()