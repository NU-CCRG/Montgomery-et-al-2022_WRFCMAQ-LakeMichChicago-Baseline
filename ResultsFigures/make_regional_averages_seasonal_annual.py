#make_regional_averages_seasonal_annual.py

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

####################################################################################################################################

def pull_chem(summer,fall,winter,spring,var):
	hourly =  [np.array(s[var]) for s in summer]+[s[var] for s in fall]+[s[var] for s in winter]+[s[var] for s in spring]
	daily = np.array([np.mean(hourly[l],axis = 0)[0] for l in range(len(hourly))])
	avg = daily.mean(axis=0)
	#avg = hourly.mean(axis=0).mean(axis=0)[0]
	#diurnal = hourly.mean(axis=0)
	seasonal = np.array([np.array(daily[0:30]).mean(axis=0),np.array(daily[30:60]).mean(axis=0),np.array(daily[60:90]).mean(axis=0),np.array(daily[90:120]).mean(axis=0)])
	return avg,daily,seasonal

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


# dir to grid file
# CMAQ RUN things
domain='d03'
time='hourly'
dir_epa = '/projects/b1045/montgomery/CMAQcheck/'
dir='/projects/b1045/jschnell/ForStacy/' 
ll='latlon_ChicagoLADCO_d03.nc' 
lon,lat = np.array(Dataset(dir+ll)['lon']),np.array(Dataset(dir+ll)['lat'])
# dir to grid file
dir='/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_FINAL_4km_sf_rrtmg_10_8_1_v3852/' 
ll='latlon.nc' 
lon2,lat2 = np.array(Dataset(dir+ll)['LON'])[0][0],np.array(Dataset(dir+ll)['LAT'])[0][0]

lond03,latd03 = lon,lat

path='/projects/b1045/montgomery/shapefiles/Chicago/commareas/geo_export_77af1a6a-f8ec-47f4-977c-40956cd94f97.shp'
chi_shapefile  = gpd.GeoDataFrame.from_file(path)
crs_new = ccrs.PlateCarree()# get shape outside
union=gpd.GeoSeries(unary_union(chi_shapefile.geometry))
outsideofunion=pd.DataFrame([list(union[0][2].exterior.xy)[0], list(union[0][2].exterior.xy)[1]])
maskd03 = mask_given_shapefile(lond03,latd03,chi_shapefile)
mask= mask_given_shapefile(lond03,latd03,chi_shapefile)
maskd02 = mask_given_shapefile(lon2,lat2,chi_shapefile)


# lake michigan
lm = gpd.read_file('/projects/b1045/montgomery/Lake_Michigan_Shoreline.shp')
lm= lm[lm['LAKE_NAME']=='Lake Michigan']
#masklake=mask_given_shapefile(lon,lat,lm)
#masklaked02=mask_given_shapefile(lon2,lat2,lm)
#pd.DataFrame(masklake).to_csv('/projects/b1045/montgomery/lakemask_d03.csv')
#pd.DataFrame(masklaked02).to_csv('/projects/b1045/montgomery/lakemask_d02.csv')

masklake=np.array(pd.read_csv('/projects/b1045/montgomery/lakemask_d03.csv',index_col = 0))
masklaked02=pd.read_csv('/projects/b1045/montgomery/lakemask_d02.csv',index_col = 0)

#Make figure
states_provinces = cfeature.NaturalEarthFeature(category='cultural',name='admin_1_states_provinces_lines',edgecolor='black',facecolor='none',scale='10m',alpha = 0.3)
borders = cfeature.NaturalEarthFeature(scale='50m',category='cultural',name='admin_0_countries',edgecolor='black',facecolor='none',alpha=0.6)
land = cfeature.NaturalEarthFeature('physical', 'lakes', '10m', edgecolor='black', facecolor='None')
#
orig_proj = ccrs.PlateCarree()
# create lambeert over  CONUS
standard_parallels = (33.000, 45)
central_longitude = -97
crs_new = ccrs.LambertConformal(central_longitude=central_longitude,standard_parallels=standard_parallels)

#
ll = Dataset('/projects/b1045/TropOMI/new/'+"latlon.nc")
lat,lon = np.array(ll['lat']),np.array(ll['lon'])
#crs_new = orig_proj
####################################################################################################################################


summer = Dataset('/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_FINAL_1.33km_sf_rrtmg_5_8_1_v3852/postprocess/avg.nc')
fall = Dataset('/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_FINAL_fall_1.33km_sf_rrtmg_5_8_1_v3852/postprocess/avg.nc')
wint = Dataset('/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_FINAL_wint_1.33km_sf_rrtmg_5_8_1_v3852/postprocess/avg.nc')
spring = Dataset('/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_FINAL_spring_1.33km_sf_rrtmg_5_8_1_v3852/postprocess/avg.nc')

datas = [[summer['NO2'][0][0],fall['NO2'][0][0],wint['NO2'][0][0],spring['NO2'][0][0],np.mean([summer['NO2'][0][0],fall['NO2'][0][0],wint['NO2'][0][0],spring['NO2'][0][0]],axis=0)],
		[summer['O3'][0][0],fall['O3'][0][0],wint['O3'][0][0],spring['O3'][0][0],np.mean([summer['O3'][0][0],fall['O3'][0][0],wint['O3'][0][0],spring['O3'][0][0]],axis=0)],
		[summer['PM25_TOT'][0][0],fall['PM25_TOT'][0][0],wint['PM25_TOT'][0][0],spring['PM25_TOT'][0][0],np.mean([summer['PM25_TOT'][0][0],fall['PM25_TOT'][0][0],wint['PM25_TOT'][0][0],spring['PM25_TOT'][0][0]],axis=0)]]


summerd02 = Dataset('/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_FINAL_4km_sf_rrtmg_10_8_1_v3852/postprocess/avg.nc')
falld02 = Dataset('/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_fall_4km_sf_rrtmg_10_8_1_v3852/postprocess/avg.nc')
wintd02 = Dataset('/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_FINAL_wint_4km_sf_rrtmg_10_8_1_v3852/postprocess/avg.nc')
springd02 = Dataset('/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_spring_4km_sf_rrtmg_10_8_1_v3852/postprocess/avg.nc')


datasd02 = [[summerd02['NO2'][0][0],falld02['NO2'][0][0],wintd02['NO2'][0][0],springd02['NO2'][0][0]],
		[summerd02['O3'][0][0],falld02['O3'][0][0],wintd02['O3'][0][0],springd02['O3'][0][0]],
		[summerd02['PM25_TOT'][0][0],falld02['PM25_TOT'][0][0],wintd02['PM25_TOT'][0][0],springd02['PM25_TOT'][0][0]]]


diffd03d03 = [np.array(pd.read_csv('../diff_NO2_d02d03.csv')),np.array(pd.read_csv('../diff_O3_d02d03.csv')),np.array(pd.read_csv('../diff_PM25_TOT_d02d03.csv'))] # made in interpolate_d03_to_d02.py
loncut,latcut = np.array(pd.read_csv('~/lon_d02d03.csv')),np.array(pd.read_csv('~/lat_d02d03.csv'))


diffd03d03 = [np.array(pd.read_csv('diff_emis_NO2_d02d03.csv')),np.array(pd.read_csv('diff_emis_POC_d02d03.csv')),np.array(pd.read_csv('diff_emis_PEC_d02d03.csv'))] # made in interpolate_d03_to_d02.py
loncut,latcut = np.array(pd.read_csv('lon_d02d03.csv')),np.array(pd.read_csv('lat_d02d03.csv'))

# Annual averages
annual = np.array(datas).mean(axis=1);
annuald02 = np.array(datasd02).mean(axis=1);
ann = [annual[i].mean() for i in range(3)]
annd02 = [annuald02[i].mean() for i in range(3)]
chi_d03 = [annual[i][mask].mean() for i in range(3)]
chi_d02 = [annuald02[i][maskd02].mean() for i in range(3)]


####################################################################################################################################




# from matplotlib import cm 
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
# make my own colormap
# create yellow colormaps
N = 256
g = np.ones((N, 4))
i,j,k = 0, 80, 49
g[:, 0] = np.flip(np.linspace(i/256, 1, N)) # R = 255
g[:, 1] = np.flip(np.linspace(j/256, 1, N)) # G = 232
g[:, 2] = np.flip(np.linspace(k/256, 1, N))  # B = 11
g_cmp = ListedColormap(g)

g = np.ones((N, 4))
#i,j,k = 0, 39, 80
#i,j,k=0, 34, 120
#i,j,k=46, 120, 0
i,j,k = 36, 36, 135
g[:, 0] = np.flip(np.linspace(i/256, 1, N)) # R = 255
g[:, 1] = np.flip(np.linspace(j/256, 1, N)) # G = 232
g[:, 2] = np.flip(np.linspace(k/256, 1, N))  # B = 11
b_cmp = ListedColormap(g)

g = np.ones((N, 4))
#i,j,k = 80, 41, 0
#i,j,k=140, 129, 0
#i,j,k=135, 0, 0
#i,j,k=88, 45, 112
i,j,k=219, 0, 0
g[:, 0] = np.flip(np.linspace(i/256, 1, N)) # R = 255
g[:, 1] = np.flip(np.linspace(j/256, 1, N)) # G = 232
g[:, 2] = np.flip(np.linspace(k/256, 1, N))  # B = 11
r_cmp = ListedColormap(g)

# Pull data
####################################################################################################################################

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# start obj
fig, axs = plt.subplots(3,5,subplot_kw={'projection': crs_new},figsize=(11, 6))
#datas = np.array([s_no2,s_o3,s_pm])

cmaps = ['viridis','viridis','viridis']
#cmaps = [g_cmp,b_cmp,r_cmp]
#cmaps=['RdYlBu_r','RdYlBu_r','RdYlBu_r']*4
vmins,vmaxs = [0,20,0],[20,55,15]
#titles=['Annual NO2','Annual O3','Annual PM']
edge=0.5
season = ['Aug. 2018','Oct. 2018','Jan. 2019','Apr. 2019','Annualized']
label = ['NO$_2$','O$_3$','PM$_{2.5}$']
units = ['ppb','ppb','$\mu$g/m$^3$']


import string
alphabet = np.array([string.ascii_lowercase[i] for i in range(15)]).reshape(axs.shape)

averageDom = []
for i in range(len(datas)):
	for j in range(len(datas[0])):
		ax = axs[i][j]
#		if j == :
#			data = datas[i]
#		else:
		data = datas[i][j]
		#title = titles[i]
		vmin=vmins[i]; vmax=vmaxs[i]; 
		cmap = cmaps[i]
		# plot on axis
		#if (i==1) & (j == 3): c = ax.pcolormesh(lon,lat,data,vmin=20,vmax=60,transform=orig_proj,cmap=cmap)
		#c = ax.pcolormesh(loncut,latcut,data,vmin=vmin,vmax=vmax,transform=orig_proj,cmap=cmap)
		c = ax.pcolormesh(lon,lat,data,vmin=vmin,vmax=vmax,transform=orig_proj,cmap=cmap)
		if j == 3:
			print('here')
			plt.colorbar(c,ax=axs[i],fraction=0.01, pad=0.02, shrink=0.9,ticks=[vmin, (vmin+vmax)/2, vmax],label=units[i])
		# configure axis
		#ax.set_extent([lon[mask].min()+edge,lon[mask].max()-edge,lat[mask].min()+edge,lat[mask].max()-edge],crs=orig_proj)
		ax.set_extent([lon.min()+edge,lon.max()-edge,lat.min()+edge,lat.max()-edge],crs=orig_proj)
		#ax.set_xlim([lon.min()+edge,lon.max()-edge])
		#ax.set_ylim([lat.min()+edge,lat.max()-edge])
		#add features to map
		ax.add_feature(land, edgecolor='black')
		ax.add_feature(borders, edgecolor='black')
		ax.add_feature(states_provinces, edgecolor='black')
		#ax.set_title(title)
		print('Average Domain: %.2f'%(data.mean()))
		averageDom.append(data.mean())
		if i == 0: ax.set_title(season[j])
		if j == 0: ax.text(-0.25, 0.45, label[i], va='bottom', ha='center',transform=ax.transAxes,fontsize = 14)
		ax.text(0.25, 0.03, '$\mu$ = %.1f'%(data.mean()), va='bottom', ha='center',transform=ax.transAxes,fontsize = 9,c='white')
		ax.text(0.12, 0.8, '(%s)'%(alphabet[i][j]), va='bottom', ha='center',transform=ax.transAxes,fontsize = 9,c='white')
		#ax.set_boundary(mpath.Path(outsideofunion.T,closed=True), transform= crs_new, use_as_clip_path=True)
		#ax.add_geometries(Reader(path).geometries(), crs=crs_new,facecolor='None', edgecolor='black')
		


#plt.tight_layout()
plt.savefig('Fig3_seasonal_and_annual_13domain.png')
plt.show()

# Create annual average plots
####################################################################################################################################
# start obj
fig, axs = plt.subplots(1,3,subplot_kw={'projection': crs_new},figsize=(18, 6))
axs = axs.ravel()
datas = [avg_no2,mdao3,avg_pm]
cmaps = ['g_cmp','b_cmp','b_cmp']
#cmaps=['Purples','Blues','Greens']
#cmaps=['RdYlBu_r','RdYlBu_r','RdYlBu_r']
vmins,vmaxs = [0,40,0],[20,60,15]
titles=['Annual NO2','Warm Season Average MDA8O3','Annual PM']
edge=0.5
#season = ['Summer','Fall','Winter','Spring']
#label = ['NOx','PM','VOC']

for i in range(len(datas)):
	ax = axs[i]
	data = datas[i]
	title = titles[i]
	vmin=vmins[i]; vmax=vmaxs[i]; 
	cmap = cmaps[i]
	# plot on axis
	c = ax.pcolormesh(lon,lat,data,vmin=vmin,vmax=vmax,transform=orig_proj,cmap=cmap)
	plt.colorbar(c,ax=ax,fraction=0.04, pad=0.02)
	# configure axis
	ax.set_extent([lon.min()+edge,lon.max()-edge,lat.min()+edge,lat.max()-edge],crs=orig_proj)
	#ax.set_xlim([lon.min()+edge,lon.max()-edge],crs=orig_proj)
	#ax.set_ylim([lat.min()+edge,lat.max()-edge],crs=orig_proj)
	#add features to map
	ax.add_feature(land, edgecolor='black')
	ax.add_feature(borders, edgecolor='black')
	ax.add_feature(states_provinces, edgecolor='black')
	ax.set_title(title)
	if i == 0: ax.set_title(season[j])
	if j == 0: ax.text(-0.25, 0.45, label[i], va='bottom', ha='center',transform=ax.transAxes,fontsize = 14)


plt.tight_layout()
plt.savefig('annual_mda.png',psi=1200)
plt.show()


# Chicago - specific figure
####################################################################################################################################
# output directory for figs
dir_out = '/projects/b1045/montgomery/citizenscience/'

#read in shapefile to get bounds
shp = gpd.read_file(dir_out+'shapefile/'+'LargeAreaCounties.shp')
union=gpd.GeoSeries(unary_union(shp.geometry))

cook = shp[shp['COUNTYNAME']=='Cook']
#
cook_shapefile = gpd.read_file('/projects/b1045/montgomery/shapefiles/Chicago/cook/Cook_County_Border.shp')
union2 = cook_shapefile.geometry.exterior.unary_union.xy
outsideofunion=pd.DataFrame([union2[0], union2[1]])

# Purple air pr

#
cook_shapefile = gpd.read_file('/projects/b1045/montgomery/shapefiles/Chicago/cook/Cook_County_Border.shp')
union2 = cook_shapefile.geometry.exterior.unary_union.xy
outsideofunion=pd.DataFrame([union2[0], union2[1]])



# shapes and directories == https://www.census.gov/cgi-bin/geo/shapefiles/index.php?year=2019&layergroup=State+Legislative+Districts
path='/projects/b1045/montgomery/shapefiles/Chicago/commareas/geo_export_77af1a6a-f8ec-47f4-977c-40956cd94f97.shp'
chi_shapefile  = gpd.GeoDataFrame.from_file(path)
crs_new = ccrs.PlateCarree()# get shape outside
union=gpd.GeoSeries(unary_union(chi_shapefile.geometry))
outsideofunion=pd.DataFrame([list(union[0][2].exterior.xy)[0], list(union[0][2].exterior.xy)[1]])


summer = Dataset('/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_FINAL_1.33km_sf_rrtmg_5_8_1_v3852/postprocess/avg.nc')
fall = Dataset('/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_FINAL_fall_1.33km_sf_rrtmg_5_8_1_v3852/postprocess/avg.nc')
wint = Dataset('/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_FINAL_wint_1.33km_sf_rrtmg_5_8_1_v3852/postprocess/avg.nc')
spring = Dataset('/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_FINAL_spring_1.33km_sf_rrtmg_5_8_1_v3852/postprocess/avg.nc')


datas = [[summer['NO2'][0][0],spring['NO2'][0][0],fall['NO2'][0][0],wint['NO2'][0][0]],
		[summer['O3'][0][0],spring['O3'][0][0],fall['O3'][0][0],wint['O3'][0][0]],
		[summer['PM25_TOT'][0][0],spring['PM25_TOT'][0][0],fall['PM25_TOT'][0][0],wint['PM25_TOT'][0][0]]]

datas = [[summer['NO2'][0][0],fall['NO2'][0][0],wint['NO2'][0][0],spring['NO2'][0][0]],
		[summer['O3'][0][0],fall['O3'][0][0],wint['O3'][0][0],spring['O3'][0][0]],
		[summer['PM25_TOT'][0][0],fall['PM25_TOT'][0][0],wint['PM25_TOT'][0][0],spring['PM25_TOT'][0][0]]]

orig_proj = crs_new
# start obj

#datas = np.array([[summer['NO2'][0][0],fall['NO2'][0],wint['NO2'][0],spring['NO2'][0]],[summer['O3'][0],fall['O3'][0],wint['O3'][0],spring['O3'][0]],[summer['PM25_TOT'][0],fall['PM25_TOT'][0],wint['PM25_TOT'][0],spring['PM25_TOT'][0]]])
cmaps = ['viridis','viridis','viridis']
#cmaps=['Purples','Blues','Greens']
#cmaps=['RdYlBu_r','RdYlBu_r','RdYlBu_r']*4
#cmaps = ['viridis']*2+['RdBu_r']
#vmins,vmaxs = [15,5,8],[25,35,14]  # cool
#vmins2,vmaxs2 = [5,10,5],[20,40,10] #warm

fig, axs = plt.subplots(3,4,subplot_kw={'projection': crs_new},figsize=(10, 7))

#vmins,vmaxs = [5,15,5],[20,40,14]
vmins,vmaxs = [5,16,5],[25,48,15]

#titles=['Annual NO2','Annual O3','Annual PM']
season = ['Summer','Fall','Winter','Spring']

season = ['Aug. 2018','Oct. 2018','Jan. 2019','Apr. 2019','Annualized']
edge=0.5
import string
alphabet = np.array([string.ascii_lowercase[i] for i in range(12)]).reshape(axs.shape)

averageChi = []

for i in range(len(datas)):
	for j in range(len(datas[0])):
		ax = axs[i][j]
		data = datas[i][j]
		#title = titles[i]
		vmin,vmax=vmins[i],vmaxs[i]
		cmap = cmaps[i]
		# plot on axis
		ax.set_boundary(mpath.Path(outsideofunion.T,closed=True), transform= crs_new, use_as_clip_path=True)
		ax.add_geometries(Reader(path).geometries(), crs=crs_new,facecolor='None', edgecolor='black',linewidth=0.5)
		c = ax.pcolormesh(lon,lat,data,vmin=vmin,vmax=vmax,transform=orig_proj,cmap=cmap)
		if (j == 3):
			plt.colorbar(c,ax=axs[i],fraction=0.04, pad=0.02,ticks=[vmin, (vmin+vmax)/2, vmax],label = units[i],shrink=0.89)
		# configure axis
		ax.set_extent([lon.min()+edge,lon.max()-edge,lat.min()+edge,lat.max()-edge],crs=orig_proj)
		#ax.set_xlim([lon.min()+edge,lon.max()-edge],crs=orig_proj)
		#ax.set_ylim([lat.min()+edge,lat.max()-edge],crs=orig_proj)
		#add features to map
		ax.set_xlim([outsideofunion.T.min()[0],outsideofunion.T.max()[0]])
		ax.set_ylim([outsideofunion.T.min()[1],outsideofunion.T.max()[1]])
		#ax.set_title(title)
		ax.add_feature(land, edgecolor='black',alpha=0.9,linewidth=0.5)
		ax.add_feature(borders, edgecolor='black',alpha=0.5,linewidth=0.5)
		ax.add_feature(states_provinces, edgecolor='black',alpha=0.5,linewidth=0.5)
		print('Average Chicago: %.2f'%(data[mask].mean()))
		averageChi.append(data[mask].mean())
		if i == 0: ax.set_title(season[j])
		if j == 0: ax.text(-0.25, 0.45, label[i], va='bottom', ha='center',transform=ax.transAxes,fontsize = 14,c='k')
		ax.text(0.25, 0.03, '$\mu$ = %.1f'%(data[mask].mean()), va='bottom', ha='center',transform=ax.transAxes,fontsize = 9,c='k')
		#ax.text(0.12, 0.8, '(%s)'%(alphabet[i][j]), va='bottom', ha='center',transform=ax.transAxes,fontsize = 9,c='k')
		ax.text(0.12, 0.15, '(%s)'%(alphabet[i][j]), va='bottom', ha='center',transform=ax.transAxes,fontsize = 9,c='k')
		

#plt.tight_layout()
plt.savefig('Fig5_Chicagodomain_seasonal.png')
plt.show()


#################################################################------------------------------------------------

fig, axs = plt.subplots(3,1,subplot_kw={'projection': crs_new},figsize=(10, 7))

#vmins,vmaxs = [5,15,5],[20,40,14]
vmins,vmaxs = [10,25,8],[22,35,12]

#titles=['Annual NO2','Annual O3','Annual PM']
#season = ['Summer','Fall','Winter','Spring']
edge=0.5
import string
alphabet = np.array([string.ascii_lowercase[i] for i in range(12)]).reshape(axs.shape)
alphabet = ['b','c','d']
averageChi = []

datast = np.mean(datas,axis=1)

for i in range(len(datas)):
		ax = axs[i]
		data = datast[i]
		#title = titles[i]
		vmin,vmax=vmins[i],vmaxs[i]
		cmap = cmaps[i]
		# plot on axis
		ax.set_boundary(mpath.Path(outsideofunion.T,closed=True), transform= crs_new, use_as_clip_path=True)
		ax.add_geometries(Reader(path).geometries(), crs=crs_new,facecolor='None', edgecolor='black',linewidth=0.5)
		c = ax.pcolormesh(lon,lat,data,vmin=vmin,vmax=vmax,transform=orig_proj,cmap=cmap)
		plt.colorbar(c,ax=axs[i],fraction=0.04, pad=0.02,ticks=[vmin, (vmin+vmax)/2, vmax],label = units[i],shrink=0.89)
		# configure axis
		ax.set_extent([lon.min()+edge,lon.max()-edge,lat.min()+edge,lat.max()-edge],crs=orig_proj)
		#ax.set_xlim([lon.min()+edge,lon.max()-edge],crs=orig_proj)
		#ax.set_ylim([lat.min()+edge,lat.max()-edge],crs=orig_proj)
		#add features to map
		ax.set_xlim([outsideofunion.T.min()[0],outsideofunion.T.max()[0]])
		ax.set_ylim([outsideofunion.T.min()[1],outsideofunion.T.max()[1]])
		#ax.set_title(title)
		ax.add_feature(land, edgecolor='black',alpha=0.9,linewidth=0.5)
		ax.add_feature(borders, edgecolor='black',alpha=0.5,linewidth=0.5)
		ax.add_feature(states_provinces, edgecolor='black',alpha=0.5,linewidth=0.5)
		print('Average Chicago: %.2f'%(data[mask].mean()))
		averageChi.append(data[mask].mean())
		#if i == 0: ax.set_title(season[j])
		#if j == 0: ax.text(-0.25, 0.45, label[i], va='bottom', ha='center',transform=ax.transAxes,fontsize = 14,c='k')
		ax.text(0.25, 0.03, '$\mu$ = %.1f'%(data[mask].mean()), va='bottom', ha='center',transform=ax.transAxes,fontsize = 9,c='k')
		#ax.text(0.12, 0.8, '(%s)'%(alphabet[i][j]), va='bottom', ha='center',transform=ax.transAxes,fontsize = 9,c='k')
		ax.text(0.12, 0.15, '(%s)'%(alphabet[i]), va='bottom', ha='center',transform=ax.transAxes,fontsize = 9,c='k')
		

#plt.tight_layout()
plt.savefig('Fig4_Chicagodomain_average.png')
plt.show()

# Calcalations for avg

# Difference in d02 d03 plots @@@@@@!!!!!herhe
####################################################################################################################################
maskcut = mask_given_shapefile(loncut,latcut,chi_shapefile)

#datas = np.array([[summer['NO2'][0],fall['NO2'][0],wint['NO2'][0],spring['NO2'][0]],[summer['O3'][0],fall['O3'][0],wint['O3'][0],spring['O3'][0]],[summer['PM25_TOT'][0],fall['PM25_TOT'][0],wint['PM25_TOT'][0],spring['PM25_TOT'][0]]])
#cmaps = ['viridis','viridis','viridis']
#cmaps=['Purples','Blues','Greens']
#cmaps=['RdYlBu_r','RdYlBu_r','RdYlBu_r']*4
cmaps = ['viridis']+['RdBu_r']
vmins,vmaxs = [10,25,8],[22,35,12]

#dvmins,dvmaxs  = [-3,-3,-3],[3,3,3]
#titles=['1.3 km','4 km','1.3 km - 4 km \nDifference']
edge=0.5
titles=['4 km','1.3 km - 4 km \nDifference']

averageChidiff = []
datas = [annuald02,np.array(diffd03d03)]
masks = [maskd02,maskcut]


fig, axs = plt.subplots(2,3,subplot_kw={'projection': crs_new},figsize=(8, 8))

alphabet = np.array([string.ascii_lowercase[i] for i in range(axs.size)]).reshape(axs.shape)

for i in range(len(datas[0])):
	for j in range(len(datas)):
		ax = axs[i][j]
		data = datas[j][i]
		title = titles[i]
		vmin=vmins[i]; vmax=vmaxs[i]; 
		cmap = cmaps[j]
		mask = masks[j]
		if j == 0: lon,lat = lon2,lat2
		if j == 1: lon,lat = loncut,latcut
		if j == 1: vmin,vmax = dvmins[i],dvmaxs[i]
		# plot on axis
		ax.set_boundary(mpath.Path(outsideofunion.T,closed=True), transform= crs_new, use_as_clip_path=True)
		ax.add_geometries(Reader(path).geometries(), crs=crs_new,facecolor='None', edgecolor='black',linewidth=0.3)
		c = ax.pcolormesh(lon,lat,data,vmin=vmin,vmax=vmax,transform=orig_proj,cmap=cmap)
		#if j == 2:
		#	plt.colorbar(c,ax=axs[i][j],fraction=0.03, pad=0.02)
		#if j == 1:
		plt.colorbar(c,ax=ax,fraction=0.03, pad=0.02,ticks=[vmin, (vmin+vmax)/2, vmax])
		# configure axis
		#ax.set_extent([lon.min()+edge,lon.max()-edge,lat.min()+edge,lat.max()-edge],crs=orig_proj)
		#ax.set_xlim([lon.min()+edge,lon.max()-edge],crs=orig_proj)
		#ax.set_ylim([lat.min()+edge,lat.max()-edge],crs=orig_proj)
		#add features to map
		ax.set_xlim([outsideofunion.T.min()[0],outsideofunion.T.max()[0]])
		ax.set_ylim([outsideofunion.T.min()[1],outsideofunion.T.max()[1]])
		#ax.set_title(title)
		print('Average Chicago: %.2f'%(data[mask].mean()))
		averageChidiff.append(data[mask].mean())
		if (j < 3) and i == 0:
			ax.set_title(titles[j])
		ax.text(0.25, 0.03, '$\mu$ = %.1f'%(data[mask].mean()), va='bottom', ha='center',transform=ax.transAxes,fontsize = 9,c='k')
		#ax.text(0.12, 0.8, '(%s)'%(alphabet[i][j]), va='bottom', ha='center',transform=ax.transAxes,fontsize = 9,c='k')
		ax.text(0.12, 0.15, '(%s)'%(alphabet[i]), va='bottom', ha='center',transform=ax.transAxes,fontsize = 9,c='k')
		if j == 0: ax.text(-0.25, 0.45, label[i], va='bottom', ha='center',transform=ax.transAxes,fontsize = 14,c='k')
		

plt.show()
#plt.tight_layout()
plt.savefig('Fig6_chicago_annual_diffplot_annual13.png')
plt.show()

# Calcalat

# All transport difference plots
####################################################################################################################################

summer = Dataset('/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_FINAL_1.33km_sf_rrtmg_5_8_1_v3852/postprocess/avg.nc')
al = Dataset('/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_Maxime_allTransport_1.33km_sf_rrtmg_5_8_1_v3852/postprocess/avg.nc')


fig, axs = plt.subplots(1,3,subplot_kw={'projection': crs_new},figsize=(16, 9))
#datas = np.array([[summer['NO2'][0],fall['NO2'][0],wint['NO2'][0],spring['NO2'][0]],[summer['O3'][0],fall['O3'][0],wint['O3'][0],spring['O3'][0]],[summer['PM25_TOT'][0],fall['PM25_TOT'][0],wint['PM25_TOT'][0],spring['PM25_TOT'][0]]])
datas = np.array(summer['NO2'][0]-al['NO2'][0]),np.array(summer['O3'][0]-al['O3'][0]),np.array(summer['PM25_TOT'][0]-al['PM25_TOT'][0])
#cmaps=['Purples','Blues','Greens']
#cmaps=['RdYlBu_r','RdYlBu_r','RdYlBu_r']*4
cmaps=['RdBu_r']*3
vmins,vmaxs = [-4,-4,-1],[4,4,1]
titles=['delta NO2','delta O3','delta PM']
edge=0.5

for i in range(len(datas)):
		ax = axs[i]
		data = datas[i][0]*-1
		title = titles[i]
		vmin=vmins[i]; vmax=vmaxs[i]; 
		cmap = cmaps[i]
		# plot on axis
		ax.set_boundary(mpath.Path(outsideofunion.T,closed=True), transform= crs_new, use_as_clip_path=True)
		ax.add_geometries(Reader(path).geometries(), crs=crs_new,facecolor='None', edgecolor='black')
		c = ax.pcolormesh(lon,lat,data,vmin=vmin,vmax=vmax,transform=orig_proj,cmap=cmap)
		plt.colorbar(c,ax=ax,fraction=0.04, pad=0.02)
		# configure axis
		#ax.set_extent([lon.min()+edge,lon.max()-edge,lat.min()+edge,lat.max()-edge],crs=orig_proj)
		#ax.set_xlim([lon.min()+edge,lon.max()-edge],crs=orig_proj)
		#ax.set_ylim([lat.min()+edge,lat.max()-edge],crs=orig_proj)
		#add features to map
		ax.set_xlim([outsideofunion.T.min()[0],outsideofunion.T.max()[0]])
		ax.set_ylim([outsideofunion.T.min()[1],outsideofunion.T.max()[1]])
		ax.add_feature(land, edgecolor='black')
		ax.add_feature(borders, edgecolor='black')
		ax.add_feature(states_provinces, edgecolor='black')
		ax.set_title(title)

plt.tight_layout()
plt.savefig('chicago_diff.png')
plt.show()


# Calculations to get averages of domain
####################################################################################################################################

averageDom, averageChi=[],[]

for i in range(len(datas)):
	for j in range(len(datas[0])):
		data = datas[i][j]
		#print('Average Domain: %.2f'%(data.mean()))
		averageDom.append(data.mean())
		#print('Average Chicago: %.2f'%(data[mask].mean()))
		averageChi.append(data[mask].mean())

print(averageChi)
print(averageDom)

# Figure 7 : Pixel vs. Distance from Lake
####################################################################################################################################

summer = Dataset('/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_FINAL_1.33km_sf_rrtmg_5_8_1_v3852/postprocess/avg.nc')
fall = Dataset('/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_FINAL_fall_1.33km_sf_rrtmg_5_8_1_v3852/postprocess/avg.nc')
wint = Dataset('/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_FINAL_wint_1.33km_sf_rrtmg_5_8_1_v3852/postprocess/avg.nc')
spring = Dataset('/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_FINAL_spring_1.33km_sf_rrtmg_5_8_1_v3852/postprocess/avg.nc')

path='/projects/b1045/montgomery/shapefiles/Chicago/commareas/geo_export_77af1a6a-f8ec-47f4-977c-40956cd94f97.shp'
chi_shapefile  = gpd.GeoDataFrame.from_file(path)
maskd03 = mask_given_shapefile(lon,lat,chi_shapefile)

var = ['NO2','O3','PM25_TOT','SO2','CO']
summer[v][0][0][~masklake].mean()-summer[v][0][0][masklake].mean();fall[v][0][0][~masklake].mean()-fall[v][0][0][masklake].mean();wint[v][0][0][~masklake].mean()-wint[v][0][0][masklake].mean();spring[v][0][0][~masklake].mean()-spring[v][0][0][masklake].mean()

v='O3'
i = 188; s = 10
lon[maskd03][i:i+s],lat[maskd03][i:i+s]
summer[v][0][0][maskd03][i:i+s]

#i = 190
fix,axs = plt.subplots(2,1,sharex=True,figsize=(3,7))
ax= axs[0]
ax.plot(lon[maskd03][i:i+s],summer[var[0]][0][0][maskd03][i:i+s],label=var[0])
ax.plot(lon[maskd03][i:i+s],summer[var[1]][0][0][maskd03][i:i+s],label=var[1])
ax.plot(lon[maskd03][i:i+s],summer[var[2]][0][0][maskd03][i:i+s],label=var[2])
ax.legend()
ax= axs[1]
ax.scatter(lon[maskd03],lat[maskd03]);plt.scatter(lon[maskd03][i:i+s],lat[maskd03][i:i+s]);plt.show()

x,y= np.where(lon == lon[maskd03][i])
x,y = x[0],y[0]

from cartopy.io.img_tiles import Stamen

# create figure object
# original projection so that we can transform it to lambert
crs_new = ccrs.PlateCarree()
crs =  ccrs.PlateCarree()
colors = plt.cm.tab20b.colors
colors = ['k','k','k','k']

import matplotlib.ticker as mticker

fig, ax = plt.subplots(subplot_kw={'projection': crs},figsize=(5,3.5))

tiler = Stamen(style='terrain')
ax.add_image(tiler, 10)
ax.coastlines('50m',linewidth=0.5)
ax.add_feature(cfeature.LAND,facecolor='None',edgecolor='k',linewidth=0.5)
states_provinces = cfeature.NaturalEarthFeature(category='cultural',name='admin_1_states_provinces_lines',scale='50m',facecolor='none')
ax.add_feature(states_provinces, edgecolor='gray',linewidth = 0.5)
ax.add_feature(cfeature.BORDERS, edgecolor='gray',linewidth = 0.5)
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_left = False
gl.xlines = False
gl.ylines = False
#gl.xlocator = mticker.FixedLocator([-180, -45, 0, 45, 180])
#gl.ylocator = mticker.FixedLocator([42.1,42,41.9,41.8,41.7,41.6])

#gl.xlocator = mticker.FixedLocator([-90,-87.5,-85])
#gl.ylocator = mticker.FixedLocator([41,42.5,44])

#gl.xformatter = LONGITUDE_FORMATTER
#gl.yformatter = LATITUDE_FORMATTER

ax.plot(lon[x-z:x+s][:,y-t:y+s][-1],lat[x-z:x+s][:,y-t:y+s][-1],label='Area Average',c='k',transform=crs)
ax.plot(lon[x-z:x+s][:,y-t:y+s][0],lat[x-z:x+s][:,y-t:y+s][0],label='Area Average',c='k',transform=crs)
ax.plot(lon[x-z:x+s][:,y-t:y+s].T[-1],lat[x-z:x+s][:,y-t:y+s].T[-1],label='Area Average',c='k',transform=crs)
ax.plot(lon[x-z:x+s][:,y-t:y+s].T[0],lat[x-z:x+s][:,y-t:y+s].T[0],label='Area Average',c='k',transform=crs)

#ax.set_xlim([lon[x-z:x+s][:,y-t:y+s].mean(axis=0).min()-0.01,lon[x-z:x+s][:,y-t:y+s].mean(axis=0).max()+0.01]);#plt.show()

#ax.plot(lon[-1],lat[-1],label='Area Average',c=colors[3],transform=crs,linewidth=0.5)
#ax.plot(lon[0],lat[0],label='Area Average',c=colors[3],transform=crs,linewidth=0.5)
#ax.plot(lon.T[-1],lat.T[-1],label='Area Average',c=colors[3],transform=crs,linewidth=0.5)
#ax.plot(lon.T[0],lat.T[0],label='Area Average',c=colors[3],transform=crs,linewidth=0.5)

chi_shapefile.plot(ax=ax,facecolor='k',edgecolor='k',alpha=0.15,transform=crs,linewidth=0.5)
#ax.scatter(lon[x-z:x+s][:,y-t:y+s],lat[x-z:x+s][:,y-t:y+s],label='Area Average',c=colors[3])
#gl.xlabel_style = {'size': 15, 'color': 'gray'}
#gl.xlabel_style = {'size': 15,'color': 'gray'}

#st = -.12
#ax.set_xlim([lon[mask].min()+2*st,lon[mask].max()-st])
#ax.set_ylim([lat[mask].min()+2*st,lat[mask].max()-st])

plt.tight_layout()
#plt.scatter(lon[maskd03],lat[maskd03])
#plt.scatter(lon[x:x+s][:,y-5:y+s],lat[x:x+s][:,y-5:y+s]);plt.show()
#plt.savefig('fig4_border.png')
plt.savefig('fig7_b.png',dpi=300)
plt.show()


ann = [(summer[var[i]][0][0]+fall[var[i]][0][0]+wint[var[i]][0][0]+spring[var[i]][0][0])/4 for i in range(len(var))]
annd02 = [(summerd02[var[i]][0][0]+falld02[var[i]][0][0]+wintd02[var[i]][0][0]+springd02[var[i]][0][0])/4 for i in range(len(var))]


################# Full CHI -- d03
s = 17+3 # center?
t = 19 # size of swath l/r
z = 22-9 # size of swath up/down
i = 188 # ind of point in maskd03
x = 118-6
y = 134-1

################# Full CHI -- d02
xx,yy =find_index(lon[x-z:x+s][:,y-t:y+s],lat[x-z:x+s][:,y-t:y+s],lon2,lat2)
xx,yy = pd.DataFrame(xx),pd.DataFrame(yy)
xx,yy = xx[0].unique(),yy[0].unique()
xx,yy = np.meshgrid(xx,yy) # xx = [171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183,184],
# yy = [219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231]


################# South CHI
#s =  15 # center?
#t = 20 # l/r
##z= 19 # up down
#i = 188
#x = 110
#y = 134

colors = plt.cm.tab20b.colors

labels = [r'NO$_2$',r'O$_3$',r'PM$_{2.5}$']

fix,axs = plt.subplots(2,1,sharex=True,figsize=(5,5))
ax= axs[0]
ax2 = ax.twinx()
lns1 = ax.plot(lon[x-z:x+s][:,y-t:y+s].mean(axis=0),ann[0][x-z:x+s][:,y-t:y+s].mean(axis=0),label=labels[0]+', ppb',c=colors[0])
lns2 = ax2.plot(lon[x-z:x+s][:,y-t:y+s].mean(axis=0),ann[1][x-z:x+s][:,y-t:y+s].mean(axis=0),label=labels[1]+', ppb',c=colors[10])
lns3 = ax.plot(lon[x-z:x+s][:,y-t:y+s].mean(axis=0),ann[2][x-z:x+s][:,y-t:y+s].mean(axis=0),label=labels[2]+r', $\mu$g/m$^3$',c=colors[4])

lns12 = ax.plot(lon2[xx,yy].mean(axis=1),annd02[0][xx,yy].mean(axis=1),label='4 km '+labels[0]+', ppb',c=colors[0],linestyle=':')
lns22 = ax2.plot(lon2[xx,yy].mean(axis=1),annd02[1][xx,yy].mean(axis=1),label='4 km '+labels[1]+', ppb',c=colors[10],linestyle=':')
lns32 = ax.plot(lon2[xx,yy].mean(axis=1),annd02[2][xx,yy].mean(axis=1),label='4 km '+labels[2]+r', $\mu$g/m$^3$',c=colors[4],linestyle=':')

ax.grid(zorder=0.1)
# added these three lines
lns = lns1+lns2+lns3
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=(0.02,-0.18),fontsize=10,ncol=3)
#leg = fig.legend(ncol=3, loc='upper center',fancybox=True)


ax.set_title('Annual Concentrations\n Relative to Lakeshore')
ax=axs[1]
ax.scatter(lon[x-z:x+s][:,y-t:y+s],lat[x-z:x+s][:,y-t:y+s],label='d03',c=colors[3]);
ax.scatter(lon2[xx,yy],lat2[xx,yy],label='d02',c=colors[1]);
ax.scatter(lon[maskd03],lat[maskd03],c=colors[4])

#ax.set_xlim([lon[x-z:x+s][:,y-t:y+s].mean(axis=0).min()-0.01,lon[x-z:x+s][:,y-t:y+s].mean(axis=0).max()+0.01]);#plt.show()
plt.tight_layout()
plt.savefig('Fig7_distance_from_lake_withchi.png',dpi=300)
#plt.close()
plt.show()

titles= [r'NO$_2$',r'O$_3$', r'PM$_{2.5}$ ']

fig,axs = plt.subplots(1,1,figsize=(5,4))
ax= axs
ax2 = ax.twinx()
ax.plot(lon[x-z:x+s][:,y-t:y+s][0],ann[0][x-z:x+s][:,y-t:y+s].mean(axis=0),label=titles[0],c=colors[0])
ax2.plot(lon[x-z:x+s][:,y-t:y+s][0],ann[1][x-z:x+s][:,y-t:y+s].mean(axis=0),label=titles[1],c=colors[10])
ax.plot(lon[x-z:x+s][:,y-t:y+s][0],ann[2][x-z:x+s][:,y-t:y+s].mean(axis=0),label=titles[2],c=colors[4])
leg = fig.legend(ncol=3, loc='upper center',fancybox=True)
leg.get_frame().set_alpha(1)

ax2.set_ylabel(r'O$_3$ (ppb)')
ax.set_ylabel(r'NO$_2$ (ppb), PM$_{2.5}$ ($\mu$g/m$^3$)')
#ax2.legend()
ax.set_ylim([6,18])
ax2.set_ylim([33,42])

#ax.set_title('Annual Concentrations\n Relative to Lakeshore')
plt.tight_layout()
#plt.savefig('Figure7b_distance_from_lake.png')
plt.show()





