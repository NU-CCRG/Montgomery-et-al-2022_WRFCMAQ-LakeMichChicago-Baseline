# plot emissions and NO2:hcho ratios

import pandas as pd
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import pandas as pd; from shapely.geometry import Point, shape, Polygon;import fiona
from shapely.ops import unary_union, cascaded_union; from geopandas.tools import sjoin
import geopandas as gpd; import geoplot; import glob; import os; from datetime import timedelta, date;
from netCDF4 import Dataset
import scipy.ndimage; from cartopy import crs as ccrs; from cartopy.io.shapereader import Reader
import matplotlib.path as mpath; import seaborn as sns
import cartopy.feature as cfeature 
import matplotlib.cm as cm

#


def mask_given_shapefile(lon,lat,shapefile):
   '''
   Make a mask given a shapefile
   lon - array of grid lons
   lat - array of grid lats
   shapefile - geopandas geodataframe shapefile
   '''
   union=gpd.GeoSeries(unary_union(shapefile.geometry))
   mask=np.ones(lon.shape,dtype=bool)
   mask[:] = False
   for i in range(len(lon)):
       for j in range(len(lon[0])):
          pt = Point(lon[i][j],lat[i][j])
          mask[i][j] =  pt.within(union[0])
   #
   return mask

####################################################################################################################################
# model data
dir='/projects/b1045/jschnell/ForStacy/' 
ll='latlon_ChicagoLADCO_d03.nc' 
lon,lat = np.array(Dataset(dir+ll)['lon']),np.array(Dataset(dir+ll)['lat'])

season = ['/AvgBase_emis_mole_all_Summer.nc','_fall/AvgBase_emis_mole_all_Fall.nc','_wint/AvgBase_emis_mole_all_Wint.nc','_spring/AvgBase_emis_mole_all_Spring.nc']
di = ['/projects/b1045/wrf-cmaq/input/emis/Chicago_LADCO/ChicagoLADCO_d03%s'%(season[i]) for i in range(len(season))]
vars = ['NO','NO2','PEC','POC','VOC_INV']

datas = np.array([[np.array(Dataset(di[i])[vars[j]])[0][0] for i in range(len(di))] for j in range(len(vars))])
datas = [datas[0]+datas[1],datas[2]+datas[3],datas[4]]

# Shapefile
path='/projects/b1045/montgomery/shapefiles/Chicago/commareas/geo_export_77af1a6a-f8ec-47f4-977c-40956cd94f97.shp'
chi_shapefile  = gpd.GeoDataFrame.from_file(path)
crs_new = ccrs.PlateCarree()# get shape outside
union=gpd.GeoSeries(unary_union(chi_shapefile.geometry))
outsideofunion=pd.DataFrame([list(union[0][2].exterior.xy)[0], list(union[0][2].exterior.xy)[1]])
mask= mask_given_shapefile(lon,lat,chi_shapefile)


masklake=np.array(pd.read_csv('/projects/b1045/montgomery/lakemask_d03.csv',index_col = 0))
masklaked02=np.array(pd.read_csv('/projects/b1045/montgomery/lakemask_d02.csv',index_col = 0))

#############################################




# create figure
fig, axs = plt.subplots(3,4,subplot_kw={'projection': crs_new},figsize=(10, 6),constrained_layout=False)

vmins,vmaxs = [0]*3,[.06,0.2,5]
titles=vars
edge=0.5
step=0.01
orig_proj = crs_new
season = ['Summer','Fall','Winter','Spring']
label = ['NOx','PM','VOC']

for i in range(len(datas)):
	for j in range(len(datas[0])):
		ax = axs[i][j]
		data = datas[i][j]
		#title = titles[i]
		vmin=vmins[i]; vmax=vmaxs[i]; 
		#cmap = cmaps[i]
		cmap = 'viridis'
		ax.annotate('$\mu$ = %.2f'%(data[mask].mean()),xy=(lon[mask].min()+step, lat[mask].min()+step),transform = crs_new,c='white')
		print('Average Chicago: %.2f'%(data[mask].mean()))
		# plot on axis
		#ax.set_boundary(mpath.Path(outsideofunion.T,closed=True), transform= crs_new, use_as_clip_path=True)
		ax.add_geometries(Reader(path).geometries(), crs=crs_new,facecolor='None', edgecolor='k',linewidth=.5)
		c = ax.pcolormesh(lon,lat,data,vmin=vmin,vmax=vmax,transform=orig_proj,cmap=cmap)
		if (j == 3) & (i < 2):
			plt.colorbar(c,ax=axs[i],fraction=0.04, pad=0.02,label='moles/s',ticks=[vmin, vmax/2, vmax])
		if (j == 3) & (i == 2):
			plt.colorbar(c,ax=axs[i],fraction=0.04, pad=0.02,label='g/s',ticks=[vmin, vmax/2, vmax])
		# configure axis
		ax.set_xlim([outsideofunion.T.min()[0],outsideofunion.T.max()[0]])
		ax.set_ylim([outsideofunion.T.min()[1],outsideofunion.T.max()[1]])
		if i == 0: ax.set_title(season[j])
		if j == 0: ax.text(-0.25, 0.45, label[i], va='bottom', ha='center',transform=ax.transAxes,fontsize = 14)

#plt.tight_layout()
plt.savefig('chicago_seasonal_emissions.png')
#plt.subplots_adjust(left=0.05, right=0.95, bottom = 0.05) # not working
plt.show()



####################################################################################################################################

season = ['/AvgBase_emis_mole_all_Summer.nc','_fall/AvgBase_emis_mole_all_Fall.nc','_wint/AvgBase_emis_mole_all_Wint.nc','_spring/AvgBase_emis_mole_all_Spring.nc']
di = ['/projects/b1045/wrf-cmaq/input/emis/Chicago_LADCO/ChicagoLADCO_d03%s'%(season[i]) for i in range(len(season))]
vars = ['NO','NO2','PEC','POC','VOC_INV']

datas = np.array([[np.array(Dataset(di[i])[vars[j]])[0][0] for i in range(len(di))] for j in range(len(vars))])
datas = [datas[0]+datas[1],datas[2]+datas[3],datas[4]]

np.array(datas).mean(axis=2).mean(axis=2)
np.array(datas)[:,:,mask].mean(axis=2)

# create figure
fig, axs = plt.subplots(3,4,subplot_kw={'projection': crs_new},figsize=(10, 6),constrained_layout=False)

vmins,vmaxs = [0]*3,[.06,0.2,5]
titles=vars
edge=0.5
step=0.01
orig_proj = crs_new
season = ['Summer','Fall','Winter','Spring']
label = ['NOx','PM','VOC']

for i in range(len(datas)):
	for j in range(len(datas[0])):
		ax = axs[i][j]
		data = datas[i][j]
		#title = titles[i]
		vmin=vmins[i]; vmax=vmaxs[i]; 
		#cmap = cmaps[i]
		cmap = 'viridis'
		ax.annotate('$\mu$ = %.2f'%(data[mask].mean()),xy=(lon[mask].min()+step, lat[mask].min()+step),transform = crs_new,c='white')
		print('Average Chicago: %.2f'%(data[mask].mean()))
		# plot on axis
		#ax.set_boundary(mpath.Path(outsideofunion.T,closed=True), transform= crs_new, use_as_clip_path=True)
		ax.add_geometries(Reader(path).geometries(), crs=crs_new,facecolor='None', edgecolor='k',linewidth=.5)
		c = ax.pcolormesh(lon,lat,data,vmin=vmin,vmax=vmax,transform=orig_proj,cmap=cmap)
		if (j == 3) & (i < 2):
			plt.colorbar(c,ax=axs[i],fraction=0.04, pad=0.02,label='moles/s',ticks=[vmin, vmax/2, vmax])
		if (j == 3) & (i == 2):
			plt.colorbar(c,ax=axs[i],fraction=0.04, pad=0.02,label='g/s',ticks=[vmin, vmax/2, vmax])
		# configure axis
		ax.set_xlim([outsideofunion.T.min()[0],outsideofunion.T.max()[0]])
		ax.set_ylim([outsideofunion.T.min()[1],outsideofunion.T.max()[1]])
		if i == 0: ax.set_title(season[j])
		if j == 0: ax.text(-0.25, 0.45, label[i], va='bottom', ha='center',transform=ax.transAxes,fontsize = 14)

#plt.tight_layout()
plt.savefig('chicago_seasonal_dep.png')
#plt.subplots_adjust(left=0.05, right=0.95, bottom = 0.05) # not working
plt.show()

