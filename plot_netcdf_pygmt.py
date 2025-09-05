import pygmt
import os

# plot shtools grid
megan = pygmt.load_dataarray('/Users/grobert3/Desktop/Desktop/dynamic_topo_obs/holdt_et_al_2022/Residual-depth-topographic-measurements/Holdt_et_al_2022_SupportingInformation/degree_1_to_40.grd')

megan_div = megan / 10. 


fig = pygmt.Figure()
region="g"
proj="N0/8i"
frame=True

with fig.subplot(nrows=3, ncols=1, figsize=("5c", "33c")):

   panel=[0,0] 
   pygmt.makecpt(cmap="polar", series=[-2, 2, 0.01], continuous=True)
   fig.basemap(region=region, projection=proj, frame=frame, panel=panel)
   fig.grdimage(grid=megan,region=region,projection=proj,frame=frame, panel=panel)
   fig.coast(region=region,projection=proj,shorelines="1p,black",area_thresh="100000", panel=panel)

   panel=[1,0] 
   pygmt.makecpt(cmap="polar", series=[-4, 4, 0.01], continuous=True)
   fig.basemap(region=region, projection=proj, frame=frame, panel=panel)
   fig.grdimage(grid=megan_div,region=region,projection=proj,frame=frame, panel=panel)
   fig.coast(region=region,projection=proj,shorelines="1p,black",area_thresh="100000", panel=panel)

   panel=[2,0] 
   pygmt.makecpt(cmap="polar", series=[-4, 4, 0.01], continuous=True)
   fig.basemap(region=region, projection=proj, frame=frame, panel=panel)
   fig.grdimage(grid=megan,region=region,projection=proj,frame=frame, panel=panel)
   fig.coast(region=region,projection=proj,shorelines="1p,black",area_thresh="100000", panel=panel)
   fig.colorbar(region=region,projection=proj,frame=["x+lAmplitude, km"], panel=panel)

fig.show()