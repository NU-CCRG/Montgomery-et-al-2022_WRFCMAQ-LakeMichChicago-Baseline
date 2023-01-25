#get_urban_rural.py
# Make urban/rural averages

# CBSA Data from : https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.2018.html
#--------------------------------------------------------
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import netCDF4
import math
from scipy.interpolate import griddata
import scipy.stats as st
import cartopy.feature as cfeature 
from cartopy import crs as ccrs;
from shapely.ops import unary_union, cascaded_union
from geopandas.tools import sjoin
from shapely.geometry import Point, shape, Polygon
from cartopy import crs as ccrs;
import geopandas as gpd
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
#-------------


#---------------------------------------------------------------------
def mask_given_shapefile(lon,lat,shapefile):
	'''
	Make a mask given a shapefile
	lon - array of grid lons
	lat - array of grid lats
	shapefile - geopandas geodataframe of a shapefile, needs geometry column
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
#---------------------------------------------------------------------

# get cities
cities = gpd.read_file('/projects/b1045/montgomery/urbanfootprints/tl_2019_us_cbsa.shp')
cities = cities[cities.LSAD == 'M1'].reset_index(drop=True) # get areas over 50k
cities['State'] = [cities.NAME[i].split(',')[1][1:3] for i in range(len(cities))] # get state codes
#cities[(cities.State=='17') | (cities.State=='55') | (cities.State=='18') | (cities.State=='26')]  # crop file to get only cities within our domain # WI, IL, IN, MI
cities = cities[(cities.State=='IL') | (cities.State=='IN') | (cities.State=='WI') | (cities.State=='MI')].reset_index(drop=True)  # crop file to get only cities within our domain # WI, IL, IN, MI

#use counties -- weird workaround so i can get county codes to match up in all 3 datasets
cities = gpd.GeoDataFrame.from_file('/projects/b1045/montgomery/nhgis0008_shape/US_county_2018.shp')
cities['COUNTYA'] = [int(cities['COUNTYFP'][i]) for i in range(len(cities))]
cities['STATEA'] = [int(cities['STATEFP'][i]) for i in range(len(cities))]

cities = cities[(cities.STATEFP=='17') | (cities.STATEFP=='55') | (cities.STATEFP=='18') | (cities.STATEFP=='26')].reset_index(drop=True) # crop file to get only cities within our domain # WI, IL, IN, MI

fin = pd.read_csv('/projects/b1045/montgomery/nhgis0010_csv/nhgis0010_ds244_20195_tract.csv',encoding='latin-1')
countpop = fin.groupby(['COUNTY','STATE']).sum()['ALT0E001'].reset_index()
countpop.columns = ['COUNTY','STATE','CountyPopulation']
fin = pd.merge(countpop,fin,on=['COUNTY','STATE'])

shp = gpd.read_file('/projects/b1045/scamilleri/BaseDiff_Annual_allPoll_EC.shp')

fin = pd.merge(fin,shp,on='GISJOIN')
f = gpd.GeoDataFrame(fin,crs=cities.crs)


ftmp = f[f['ALT0E001_x'] >316955.8]
ftmp.plot();plt.show()


# lake michigan
lm = gpd.read_file('/projects/b1045/montgomery/Lake_Michigan_Shoreline.shp')
lm= lm[lm['LAKE_NAME']=='Lake Michigan']
masklake=np.array(pd.read_csv('/projects/b1045/montgomery/lakemask_d03.csv',index_col = 0))

# get lat lon
lon = np.array(Dataset('/projects/b1045/jschnell/ForStacy/'+'latlon_ChicagoLADCO_d03.nc' )['lon'])
lat = np.array(Dataset('/projects/b1045/jschnell/ForStacy/'+'latlon_ChicagoLADCO_d03.nc' )['lat'])

d02 = Dataset('/home/asm0384/lat_lon_chicago_d02.nc')

urban_mask = []

for i in range(len(cities)):
	c = gpd.GeoDataFrame(cities.loc[i]).T
	m1 = mask_given_shapefile(lon,lat,c)
	urban_mask.append(m1)


urban_mask = np.array(urban_mask)
#plt.pcolormesh(urban_mask.sum(axis=0))
mask = ~urban_mask.sum(axis=0)
mask =  masklake+~urban_mask.sum(axis=0)
plt.pcolormesh((mask==0)|(mask==-1))
mask = (mask==0)|(mask==-1)
pd.DataFrame(mask).to_csv('/projects/b1045/montgomery/mask_urban_areas.csv')
np.ma.masked_array(mask,mask=mask)

# ------------------------------------------
# plot the urban mask
from cartopy.io.img_tiles import Stamen
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
tiler = Stamen(style='terrain-background')
ax.add_image(tiler, 9)
ax.coastlines('10m')
ax.add_feature(cfeature.LAND,facecolor='None',edgecolor='k')
states_provinces = cfeature.NaturalEarthFeature(category='cultural',name='admin_1_states_provinces_lines',scale='50m',facecolor='none')
ax.add_feature(states_provinces, edgecolor='gray')

ax.pcolormesh(lon,lat,np.ma.masked_array(mask,mask=mask),alpha=0.3,cmap='tab20b',edgecolor='face')
ax.set_xlim([lon.min(),lon.max()])
ax.set_ylim([lat.min(),lat.max()])
ax.set_title('Counties with >95%ile Population in 1.3 km Domain')
plt.savefig('/projects/b1045/montgomery/paper1/counties_with_95_population.pdf')
plt.show()

# calculate urban rural difference
mask = pd.read_csv('/projects/b1045/montgomery/mask_urban_areas.csv',index_col=0)
mask = np.ma.masked_array(mask,mask=mask)

summer = Dataset('/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_FINAL_1.33km_sf_rrtmg_5_8_1_v3852/postprocess/avg.nc')
fall = Dataset('/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_FINAL_fall_1.33km_sf_rrtmg_5_8_1_v3852/postprocess/avg.nc')
wint = Dataset('/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_FINAL_wint_1.33km_sf_rrtmg_5_8_1_v3852/postprocess/avg.nc')
spring = Dataset('/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_FINAL_spring_1.33km_sf_rrtmg_5_8_1_v3852/postprocess/avg.nc')

var = ['NO2','O3','PM25_TOT','SO2','CO']

ur,ru = [],[]

for i in range(len(var)):
	ur.append([summer[var[i]][0][0][mask.data].mean(),fall[var[i]][0][0][mask.data].mean(),wint[var[i]][0][0][mask.data].mean(),spring[var[i]][0][0][mask.data].mean()])
	ru.append([summer[var[i]][0][0][~mask.data].mean(),fall[var[i]][0][0][~mask.data].mean(),wint[var[i]][0][0][~mask.data].mean(),spring[var[i]][0][0][~mask.data].mean()])
	
diff = pd.DataFrame(ru)-pd.DataFrame(ur)
diff.columns = ['08/18','10/18','01/19','04/19']
diff['var'] = var
diff.to_csv('/projects/b1045/montgomery/paper1/urban_rural_diff.csv')

