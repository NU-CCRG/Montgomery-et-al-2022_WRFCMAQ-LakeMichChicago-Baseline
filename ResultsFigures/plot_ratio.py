
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

# Shapefiles
path='/projects/b1045/montgomery/shapefiles/Chicago/commareas/geo_export_77af1a6a-f8ec-47f4-977c-40956cd94f97.shp'
chi_shapefile  = gpd.GeoDataFrame.from_file(path)
crs_new = ccrs.PlateCarree()# get shape outside
union=gpd.GeoSeries(unary_union(chi_shapefile.geometry))
outsideofunion=pd.DataFrame([list(union[0][2].exterior.xy)[0], list(union[0][2].exterior.xy)[1]])
mask= mask_given_shapefile(lon,lat,chi_shapefile)
masklake=np.array(pd.read_csv('/projects/b1045/montgomery/lakemask_d03.csv',index_col = 0))
masklaked02=np.array(pd.read_csv('/projects/b1045/montgomery/lakemask_d02.csv',index_col = 0))


# plot ratio
####################################################################################################################################

# Get the weekend vs weekday hotspots# Get the weekend vs weekday hotspots# Get the weekend vs weekday hotspotsv
# pull cmaq data 
summer = Dataset('/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_FINAL_1.33km_sf_rrtmg_5_8_1_v3852/column/all.nc')
fall = Dataset('/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_FINAL_fall_1.33km_sf_rrtmg_5_8_1_v3852/column/all.nc')
wint = Dataset('/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_FINAL_wint_1.33km_sf_rrtmg_5_8_1_v3852/column/all.nc')
spring = Dataset('/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_FINAL_spring_1.33km_sf_rrtmg_5_8_1_v3852/column/all.nc')

#Make figure
states_provinces = cfeature.NaturalEarthFeature(category='cultural',name='admin_1_states_provinces_lines',edgecolor='black',facecolor='none',scale='10m',alpha = 0.3)
borders = cfeature.NaturalEarthFeature(scale='50m',category='cultural',name='admin_0_countries',edgecolor='black',facecolor='none',alpha=0.6)
land = cfeature.NaturalEarthFeature('physical', 'lakes', '10m', edgecolor='black', facecolor='None')
#


alls = [np.mean(summer['FORM'][:]/summer['NO2'][:],axis=0),np.mean(fall['FORM'][:]/fall['NO2'][:],axis=0),
		np.mean(wint['FORM'][:]/wint['NO2'][:],axis=0),np.mean(spring['FORM'][:]/spring['NO2'][:],axis=0)]

v='FORM'
form = [np.mean([np.mean(summer[v][(24*i+12):(24*i+25)],axis=0) for i in range(30)],axis=0),
		np.mean([np.mean(fall[v][(24*i+12):(24*i+25)],axis=0) for i in range(30)],axis=0),
		np.mean([np.mean(wint[v][(24*i+12):(24*i+25)],axis=0) for i in range(30)],axis=0),
		np.mean([np.mean(spring[v][(24*i+12):(24*i+25)],axis=0) for i in range(29)],axis=0)]

v = 'NO2'
no2 = [np.mean([np.mean(summer[v][(24*i+12):(24*i+25)],axis=0) for i in range(30)],axis=0),
		np.mean([np.mean(fall[v][(24*i+12):(24*i+25)],axis=0) for i in range(30)],axis=0),
		np.mean([np.mean(wint[v][(24*i+12):(24*i+25)],axis=0) for i in range(30)],axis=0),
		np.mean([np.mean(spring[v][(24*i+12):(24*i+25)],axis=0) for i in range(29)],axis=0)]

form = np.array(form)
no2=np.array(no2)
datas = form/no2

import matplotlib as mpl

plt.rcParams['figure.dpi'] = 300

#datas = [summer['FORM'][0]/summer['NO2'][0],fall['FORM'][0]/fall['NO2'][0],wint['FORM'][0]/wint['NO2'][0],spring['FORM'][0]/spring['NO2'][0]] #test
summer = Dataset('/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_FINAL_1.33km_sf_rrtmg_5_8_1_v3852/postprocess/avg.nc')
fall = Dataset('/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_FINAL_fall_1.33km_sf_rrtmg_5_8_1_v3852/postprocess/avg.nc')
wint = Dataset('/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_FINAL_wint_1.33km_sf_rrtmg_5_8_1_v3852/postprocess/avg.nc')
spring = Dataset('/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_FINAL_spring_1.33km_sf_rrtmg_5_8_1_v3852/postprocess/avg.nc')
d1 = [summer['FORM'][0][0]/summer['NO2'][0][0],fall['FORM'][0][0]/fall['NO2'][0][0],wint['FORM'][0][0]/wint['NO2'][0][0],spring['FORM'][0][0]/spring['NO2'][0][0]]
allo = [np.mean(d1,axis=0),np.mean(datas,axis=0)[0]]+list(datas)



# create figure
fig, axs = plt.subplots(3,2,subplot_kw={'projection': crs_new},figsize=(8, 10),constrained_layout=False)
axs = axs.ravel()

vmins,vmaxs = 0,3
titles=['2018 - 2019 Column','2018 - 2019 Surface ','Aug. 2018','Oct. 2018','Jan. 2019','Apr. 2019']
edge=0.5
step=0.01
orig_proj = crs_new
season = ['Surface Annualized','Column Annualized','Aug. 2018','Oct. 2018','Jan. 2019','Apr. 2019'] #['Summer','Fall','Winter','Spring']
s = ['a','(b) ','(c) ','(d) ','(e) ','(f)','(g)']

for i in range(len(axs)):
	ax = axs[i]
	if i < 2: 
		data = allo[i]
	if i >= 2: 
		data = allo[i][0]
	vmin=vmins; 
	vmax=vmaxs; 
	if i == 0:
			vmin = 0
			vmax = 0.5
	# colormap
	label = 'Chicago Ratio = %.2f\nDomain Ratio = %.2f'%(data[mask].mean(),data[~masklake].mean())
	#cmap = plt.get_cmap('viridis', 30)
	cmap = 'viridis'
	# Normalizer
	#norm = mpl.colors.Normalize(vmin=0, vmax=vmax)
	# creating ScalarMappable
	#sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
	#cmap = sm
	c = ax.pcolor(lon,lat,data,vmin=vmin,vmax=vmax,transform=orig_proj,shading='auto')
	# configure axis
	ax.add_feature(land, edgecolor='black')
	ax.add_feature(borders, edgecolor='black')
	ax.add_feature(states_provinces, edgecolor='black')
	ax.set_extent([lon.min()+edge,lon.max()-edge,lat.min()+edge,lat.max()-edge],crs=orig_proj)
	ax.set_title(s[i]+r'HCHO:NO$_2$ Ratio '+season[i])
	ax.text(0.5, -0.22, label, va='bottom', ha='center',transform=ax.transAxes,fontsize = 10)
	plt.colorbar(c,ax=ax,fraction=0.029, pad=0.02,label='Ratio',ticks=[0,1,2,3])
	#plt.tight_layout()
	#plt.savefig('chicago_seasonal_dep.png')
	#plt.subplots_adjust(left=0.05, right=0.95, bottom = 0.05) # not working

plt.savefig('Fig8_chicago_vocratio_seasons.png',dpi=300)#psi=1200)
plt.show()


