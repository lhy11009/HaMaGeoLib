#!/opt/miniconda3/bin/python


import pygmt

loc = ['kur'];
      
fig = pygmt.Figure()
pygmt.config(FONT='Times-Roman')
pygmt.config(FONT_LABEL='Times-Roman,12p')

fig.basemap(region="g", projection="W15c", frame=True)
grid = pygmt.datasets.load_earth_relief(resolution="10m",registration="gridline")
fig.grdimage(grid=grid,cmap="gray")
fig.coast(shorelines="1/0.5p,black")

slab2dir ="/Users/billen/Box-Sync/Mybin/Data-Sets/Slab2Distribute_Mar2018/"

grdfile = slab2dir + loc[0] + '_slab2_dep.grd'
#fig.grdimage(grid=grdfile,Q="True",cmap=cptfile)
#fig.grdimage(grid=grdfile,nan_transparent="True",cmap=cptfile)
fig.grdimage(grid=grdfile,nan_transparent=True,cmap="buda")
   
fig.colorbar(cmap="buda",position="JMR",box=False,frame=["x+lDepth", "y+lkm"])
fig.savefig('test_slab2_map.png')
