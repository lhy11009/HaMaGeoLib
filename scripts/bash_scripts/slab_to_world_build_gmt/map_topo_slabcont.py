#!/opt/miniconda3/bin/python

# to use this first:
# > conda activate pygmt

import pygmt

loc = ['alu','cal','cam','car','cas','cot','hal','hel','him','hin','izu',
	'ker','kur','mak','man','mue','pam',
    'phi','png','puy','ryu','sam','sco','sol','sul','sum','van'];
      
fig = pygmt.Figure()
pygmt.config(FONT='Times-Roman')
pygmt.config(FONT_LABEL='Times-Roman,12p')

fig.basemap(region="g", projection="W15c", frame=True)
grid = pygmt.datasets.load_earth_relief(resolution="10m",registration="gridline")
fig.grdimage(grid=grid,cmap="gray")
fig.coast(shorelines="1/0.5p,black")

cptfile ="slabdepth.cpt"
pygmt.makecpt(cmap="buda",series=(-700,0,100), color_model="+c",output=cptfile)

slab2dir ="/Users/billen/Box-Sync/Mybin/Slab2Distribute_Mar2018/"

for i in range(len(loc)):
    #print(loc[i])
    grdfile = slab2dir + loc[i] + '_slab2_dep.grd'
    fig.grdimage(grid=grdfile,Q="True",cmap=cptfile)

    
fig.colorbar(cmap=cptfile,position="JMR",box=False,frame=["x+lDepth", "y+lkm"])
fig.savefig('map_topo_slabs.png')


