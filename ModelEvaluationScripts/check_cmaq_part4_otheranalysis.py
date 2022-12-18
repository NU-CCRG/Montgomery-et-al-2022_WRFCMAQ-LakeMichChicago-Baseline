# Make variogram of d02 and d03

import pandas as pd
from scipy.stats import pearsonr
import numpy as np
from sklearn.metrics import mean_squared_error
import scipy.stats as st
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from cartopy import crs as ccrs;
import cartopy.feature as cfeature 
import matplotlib.pyplot as plt
####################################################################################################################################

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

# functions
def stats_normalized(data,prediction):
    x,y=data[~np.isnan(data)],prediction[~np.isnan(data)] # get rid of NaNs
    x,y = x[x>0],y[x>0] # get rid of negatives
    mu_d,mu_p = np.mean(x),np.mean(y)
    nmb = np.sum(y-x)/np.sum(x)*100
    nme = np.sum(np.abs(y-x))/np.sum(x)*100
    nrmse = np.sqrt(1/len(x)*np.sum((y-x)**2))/np.mean(x)
    r,p = st.pearsonr(x,y)
    rms = rmse(y,x)
    mb = np.sum(y-x)/len(x)
    return mu_d,mu_p,nmb,nme,mb,rms,r,p


####################################################################################################################################
d02 = Dataset('/home/asm0384/lat_lon_chicago_d02.nc')
lat,lon = d02['LAT'][0][0].data,d02['LON'][0][0].data

d03 = Dataset('/projects/b1045/wrf-cmaq/output/Chicago_LADCO/mcip/PXLSM/ChicagoLADCO_d03/latlon_ChicagoLADCO_d03.nc')
lat,lon = np.array(d03['lat']),np.array(d03['lon'])

shape = lon.shape

remove_indiana == True

# CMAQ RUN things
domain='d03'
time='hourly'
dir_epa = '/projects/b1045/montgomery/CMAQcheck/'
#epa_code=['42401','42602','44201','42101','88101']; var=['SO2','NO2','O3','CO','PM25_TOT'] #numerical identifiers and corresponding vars
epa_code=['42602','44201','88101','42401','42101']; 
var=['NO2','O3','PM25_TOT','SO2','CO'] 
#epa_code = ['44201']
#var = ['O3']
epa_units = ['Parts per billion']*4+['Parts per million']*4+['Microgram per meters squared']*4+['Parts per billion']*4+['Parts per million']

years=['2018','2018','2019','2019']
months=['8','10','1','4']

# Format file names
epa_files = []
names=[]
for i in range(len(years)):
	year,month = years[i],months[i]
	epa_files.append([dir_epa+'%s_%s_%s_%s_EPA_CMAQ_Combine.csv'%(var[loop],domain,year,month) for loop in range(len(var))])
	names.append(['%s_%s_%s_%s'%(var[loop],domain,year,month) for loop in range(len(var))])

epa_files=[]
names=[]
for i in range(len(var)):
    for m in range(len(months)):
    	month,year = months[m],years[m]
    	epa_files.append([dir_epa+'%s_%s_%s_%s_EPA_CMAQ_Combine.csv'%(var[i],domain,year,month)])
    	names.append('%s_%s_%s_%s'%(var[i],domain,year,month))

epa_files = np.array(epa_files).ravel(); names= np.array(names).ravel()

years=['2018','2018','2019','2019']
months=['8','10','1','4']

epa_filesd02=[]
namesd02=[]
for i in range(len(var)):
    for m in range(len(months)):
    	month,year = months[m],years[m]
    	epa_filesd02.append([dir_epa+'%s_%s_%s_%s_EPA_CMAQ_Combine.csv'%(var[i],'d02',year,month)])
    	namesd02.append('%s_%s_%s_%s'%(var[i],'d02',year,month))

epa_filesd02 = np.array(epa_filesd02).ravel(); names= np.array(names).ravel()


####################################################################################################################################
# Create variogram

d02 = Dataset('/home/asm0384/lat_lon_chicago_d02.nc')
lat2,lon2 = d02['LAT'][0][0].data,d02['LON'][0][0].data
d03 = Dataset('/projects/b1045/wrf-cmaq/output/Chicago_LADCO/mcip/PXLSM/ChicagoLADCO_d03/latlon_ChicagoLADCO_d03.nc')
lat3,lon3 = np.array(d03['lat']),np.array(d03['lon'])


def skg(x,y,N):
	#N = len(x)
	return 1/(2*N)*np.sum((x-y)**2)

from math import sin, cos, sqrt, atan2, radians

def dist(lon1,lat1,lon2,lat2):
	# approximate radius of earth in km
	R = 6373.0
	lat1 = radians(lat1)
	lon1 = radians(lon1)
	lat2 = radians(lat2)
	lon2 = radians(lon2)
	dlon = lon2 - lon1
	dlat = lat2 - lat1
	a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
	c = 2 * atan2(sqrt(a), sqrt(1 - a))
	return R * c

def quick_clean(f,lon,lat,tstep):
#for i in range(1):
	f = f[~np.isnan(f['State Code'])].reset_index(drop=True)
	f.dt = pd.to_datetime(f.dt)
	f['clon'] = [lon[int(f.x[i])][int(f.y[i])] for i in range(len(f))]
	f['clat'] = [lat[int(f.x[i])][int(f.y[i])] for i in range(len(f))]
	f['day'] = [f.dt[i].day for i in range(len(f))]
	f = f.set_index(f.dt)
	favg = f.groupby(['Longitude','Latitude']).resample(tstep).mean().reset_index(drop=True)
	return favg


fig,ax=plt.subplots()

#for i in range(len(epa_files)):
for i in range(4):
	f = pd.read_csv(epa_files[i])
	favg = quick_clean(f,lon3,lat3,'M')
	f2 = pd.read_csv(epa_filesd02[i])
	f2 = f2[f2['Latitude'].isin(f.Latitude.unique())].reset_index(drop=True) # crop d02 to d03 lats
	f2avg = quick_clean(f2,lon2,lat2)
	d_d03 = np.array([[dist(favg.Longitude[i],favg.Latitude[i],favg.clon[j],favg.clat[j]) for i in range(len(favg))] for j in range(len(favg))])
	d_d02 = np.array([[dist(f2avg.Longitude[i],f2avg.Latitude[i],f2avg.clon[j],f2avg.clat[j]) for i in range(len(favg))] for j in range(len(favg))])
	skg_d03 = np.array([[skg(favg['Sample Measurement'][i],favg.CMAQ[j],len(favg)) for i in range(len(favg))] for j in range(len(favg))])
	skg_d02 = np.array([[skg(f2avg['Sample Measurement'][i],f2avg.CMAQ[j],len(favg)) for i in range(len(f2avg))] for j in range(len(f2avg))])
	pwdifd03 = np.array([[favg['Sample Measurement'][i]-favg.CMAQ[j] for i in range(len(favg))] for j in range(len(favg))])
	pwdifd02 = np.array([[f2avg['Sample Measurement'][i]-f2avg.CMAQ[j] for i in range(len(f2avg))] for j in range(len(f2avg))])
	#ax.scatter(d_d03.mean(axis=0),skg_d03.mean(axis=0),c='blue',label='d03')
	#ax.scatter(d_d02.mean(axis=0),skg_d02.mean(axis=0),c='k',label='d02')
	#ax.scatter(d_d03,skg_d03,c='blue',label='d03')
	#ax.scatter(d_d02,skg_d02,c='k',label='d02')
	d3,d2 = pd.DataFrame({'dist':d_d03.ravel(),'diff':pwdifd03.ravel(),'skg':skg_d03.ravel()}),pd.DataFrame({'dist':d_d02.ravel(),'diff':pwdifd02.ravel(),'skg':skg_d02.ravel()})
	d3['bin'] = pd.qcut(d3['dist'], q=25)
	d2['bin'] = pd.qcut(d3['dist'], q=25)
	d3['dt'] = j; d2['dt'] = j;
	if i == 0:
		d3f = d3; d2f = d2
	else:
		d3f = d3f.append(d3); d2f = d2f.append(d2)

	#ax.scatter(d_d03,pwdifd03,c='blue',label='d03')
	#ax.scatter(d_d02,pwdifd02,c='k',label='d02')
	d3.plot('bin','diff',marker='*',linestyle='none',ax=ax,label='d03')
	d2.plot('bin','diff',marker='*',linestyle='none',ax=ax,label='d02')


#plt.legend()
plt.show()

d3 = d3.groupby('bin').mean().reset_index()
d2 = d2.groupby('bin').mean().reset_index()

fig,ax = plt.subplots()
d3.plot('bin','skg',marker='*',linestyle='none',ax=ax,label='d03')
d2.plot('bin','skg',marker='*',linestyle='none',ax=ax,label='d02')

plt.show()
####################################################################################################################################
# Check daily variogram by distance

# 90th Percentile
def q90(x):
	return x.quantile(0.75)

def q10(x):
	return x.quantile(0.25)

tmp3 = d3f.groupby(['bin2']).agg(['mean',q10,q90]).reset_index()
tmp2 = d2f.groupby(['bin2']).agg(['mean',q10,q90]).reset_index()


fig,ax = plt.subplots()
ax.errorbar(x=np.arange(len(tmp3)),y=tmp3['skg']['mean'],yerr=[tmp3['skg']['mean']-tmp3['skg']['q10'],tmp3['skg']['q90']],linestyle='None',marker='*')
ax.errorbar(x=np.arange(len(tmp2)),y=tmp2['skg']['mean'],yerr=[tmp2['skg']['mean']-tmp2['skg']['q10'],tmp2['skg']['q90']],linestyle='None',marker='*')

plt.show()


fig,ax = plt.subplots()
d3f.plot('bin2','skg',marker='*',linestyle='none',ax=ax,label='d03')
d2f.plot('bin2','skg',marker='*',linestyle='none',ax=ax,label='d02')

plt.show()


#for i in range(len(epa_files)):
for i in range(4):
	f = pd.read_csv(epa_files[i])
	favg = quick_clean(f,lon3,lat3,'M')
	f2 = pd.read_csv(epa_filesd02[i])
	f2 = f2[f2['Latitude'].isin(f.Latitude.unique())].reset_index(drop=True) # crop d02 to d03 lats
	f2avg = quick_clean(f2,lon2,lat2)
	d_d03 = np.array([[dist(favg.Longitude[i],favg.Latitude[i],favg.clon[j],favg.clat[j]) for i in range(len(favg))] for j in range(len(favg))])
	d_d02 = np.array([[dist(f2avg.Longitude[i],f2avg.Latitude[i],f2avg.clon[j],f2avg.clat[j]) for i in range(len(favg))] for j in range(len(favg))])
	skg_d03 = np.array([[skg(favg['Sample Measurement'][i],favg.CMAQ[j],len(favg)) for i in range(len(favg))] for j in range(len(favg))])
	skg_d02 = np.array([[skg(f2avg['Sample Measurement'][i],f2avg.CMAQ[j],len(favg)) for i in range(len(f2avg))] for j in range(len(f2avg))])
	pwdifd03 = np.array([[favg['Sample Measurement'][i]-favg.CMAQ[j] for i in range(len(favg))] for j in range(len(favg))])
	pwdifd02 = np.array([[f2avg['Sample Measurement'][i]-f2avg.CMAQ[j] for i in range(len(f2avg))] for j in range(len(f2avg))])
	#ax.scatter(d_d03.mean(axis=0),skg_d03.mean(axis=0),c='blue',label='d03')
	#ax.scatter(d_d02.mean(axis=0),skg_d02.mean(axis=0),c='k',label='d02')
	#ax.scatter(d_d03,skg_d03,c='blue',label='d03')
	#ax.scatter(d_d02,skg_d02,c='k',label='d02')
	d3,d2 = pd.DataFrame({'dist':d_d03.ravel(),'diff':pwdifd03.ravel(),'skg':skg_d03.ravel()}),pd.DataFrame({'dist':d_d02.ravel(),'diff':pwdifd02.ravel(),'skg':skg_d02.ravel()})
	d3['bin'] = pd.qcut(d3['dist'], q=25)
	d2['bin'] = pd.qcut(d3['dist'], q=25)
	d3['dt'] = j; d2['dt'] = j;
	if i == 0:
		d3f = d3; d2f = d2
	else:
		d3f = d3f.append(d3); d2f = d2f.append(d2)

	#ax.scatter(d_d03,pwdifd03,c='blue',label='d03')
	#ax.scatter(d_d02,pwdifd02,c='k',label='d02')
	d3.plot('bin','diff',marker='*',linestyle='none',ax=ax,label='d03')
	d2.plot('bin','diff',marker='*',linestyle='none',ax=ax,label='d02')


####################################################################################################################################
# Check variogram by distance
#

for i in range(4):
	f = pd.read_csv(epa_files[i])
	favg = quick_clean(f,lon3,lat3,'M')
	f2 = pd.read_csv(epa_filesd02[i])
	f2 = f2[f2['Latitude'].isin(f.Latitude.unique())].reset_index(drop=True) # crop d02 to d03 lats
	f2avg = quick_clean(f2,lon2,lat2,'M')
	d_d03 = np.array([[dist(favg.Longitude[i],favg.Latitude[i],favg.Longitude[j],favg.Latitude[j]) for i in range(len(favg))] for j in range(len(favg))])
	d_d02 = np.array([[dist(f2avg.Longitude[i],f2avg.Latitude[i],f2avg.Longitude[j],f2avg.Latitude[j]) for i in range(len(favg))] for j in range(len(favg))])
	skg_sample = np.array([[skg(favg['Sample Measurement'][i],favg['Sample Measurement'][j],len(favg)) for i in range(len(favg))] for j in range(len(favg))])
	skg_d03 = np.array([[skg(favg.CMAQ[i],favg.CMAQ[j],len(favg)) for i in range(len(favg))] for j in range(len(favg))])
	skg_d02 = np.array([[skg(f2avg.CMAQ[i],f2avg.CMAQ[j],len(favg)) for i in range(len(f2avg))] for j in range(len(f2avg))])
	pwdifd03 = np.array([[favg['Sample Measurement'][i]-favg.CMAQ[j] for i in range(len(favg))] for j in range(len(favg))])
	pwdifd02 = np.array([[f2avg['Sample Measurement'][i]-f2avg.CMAQ[j] for i in range(len(f2avg))] for j in range(len(f2avg))])
	d3,d2 = pd.DataFrame({'dist':d_d03.ravel(),'diff':pwdifd03.ravel(),'skg':skg_d03.ravel(),'skgS':skg_sample.ravel()}),pd.DataFrame({'dist':d_d02.ravel(),'diff':pwdifd02.ravel(),'skg':skg_d02.ravel(),'skgS':skg_sample.ravel()})
	#d3['bin'] = pd.qcut(d3['dist'], q=5)
	#d2['bin'] = pd.qcut(d3['dist'], q=5)
	#d3['dt'] = j; d2['dt'] = j;
	if i == 0:
		d3f = d3; d2f = d2
	else:
		d3f = d3f.append(d3); d2f = d2f.append(d2)

d3f = d3f.groupby('dist').mean().reset_index()
d2f = d2f.groupby('dist').mean().reset_index()

d3f['bin'] = pd.qcut(d3f['dist'], q=10)
d2f['bin'] = pd.qcut(d3f['dist'], q=10)

tmp3 = d3f.groupby(['bin']).agg(['mean',q10,q90]).reset_index()
tmp2 = d2f.groupby(['bin']).agg(['mean',q10,q90]).reset_index()
tmp3['bindist'] = [tmp3.bin[i].right for i in range(len(tmp3))]
tmp2['bindist'] = [tmp2.bin[i].right for i in range(len(tmp2))]

fig,ax = plt.subplots()
ax.scatter(tmp3.bindist,tmp3['skg']['mean'],label='d03',zorder=10)
ax.scatter(tmp3.bindist,tmp2['skg']['mean'],label='d02',zorder=10)
ax.scatter(tmp3.bindist,tmp2['skgS']['mean'],c='k',label='sample',zorder=10)
ax.set_xlabel('Distance (km)')
ax.set_ylabel('Semivariogram')
ax.grid()
plt.legend()
plt.show()

####################################################################################################################################
# Check variogram of cmaq files only


def skg(x,y,N):
	N = 1
	return 1/(2*N)*np.sum((x-y)**2)
	#return np.std(x,y)


d02 = Dataset('/home/asm0384/lat_lon_chicago_d02.nc')
lat2,lon2 = d02['LAT'][0][0].data,d02['LON'][0][0].data
d03 = Dataset('/projects/b1045/wrf-cmaq/output/Chicago_LADCO/mcip/PXLSM/ChicagoLADCO_d03/latlon_ChicagoLADCO_d03.nc')
lat3,lon3 = np.array(d03['lat']),np.array(d03['lon'])

d3 = [Dataset('/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_FINAL_1.33km_sf_rrtmg_5_8_1_v3852/postprocess/avg.nc'),
	Dataset('/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_FINAL_fall_1.33km_sf_rrtmg_5_8_1_v3852/postprocess/avg.nc'),
	Dataset('/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_FINAL_wint_1.33km_sf_rrtmg_5_8_1_v3852/postprocess/avg.nc'),
	Dataset('/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_FINAL_spring_1.33km_sf_rrtmg_5_8_1_v3852/postprocess/avg.nc')]

d2 = [Dataset('/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_FINAL_4km_sf_rrtmg_10_8_1_v3852/postprocess/avg.nc'),
	Dataset('/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_fall_4km_sf_rrtmg_10_8_1_v3852/postprocess/avg.nc'),
	Dataset('/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_FINAL_wint_4km_sf_rrtmg_10_8_1_v3852/postprocess/avg.nc'),
	Dataset('/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_spring_4km_sf_rrtmg_10_8_1_v3852/postprocess/avg.nc')]

xy = [(np.random.randint(20+61,260-61),np.random.randint(51+30,270-31)) for i in range(60)]
xy2 = [(np.random.randint(128+20,246-20),np.random.randint(181-20,284-20)) for i in range(60)] # crop sampling indices to d03 domain

def variogram(d3,d2,xy,xy2,v):
	skg_d03,skg_d02 = [],[]
	distance_d02,distance_d03 = [],[]
	for f in range(1):
		tmp3 = d3[f][v][0][0]
		tmp2 = d2[f][v][0][0]
		for x in range(len(xy)):
			skg_d03 = skg_d03 + [skg(tmp3[xy[x]],tmp3[xy[x][0]+i][xy[x][1]+j],100) for i in range(-70,70) for j in range(-70,70)]
			distance_d03 = distance_d03+[dist(lon3[xy[x]], lat3[xy[x]], lon3[xy[x][0]+i][xy[x][1]+j], lat3[xy[x][0]+i][xy[x][1]+j]) for i in range(-70,70) for j in range(-70,70)]
			skg_d02 = skg_d02+[skg(tmp2[xy2[x]],tmp2[xy2[x][0]+i][xy2[x][1]+j],100) for i in range(-25,25) for j in range(-25,25)]
			distance_d02= distance_d02+ [dist(lon2[xy2[x]], lat2[xy2[x]], lon2[xy2[x][0]+i][xy2[x][1]+j], lat2[xy2[x][0]+i][xy2[x][1]+j]) for i in range(-25,25) for j in range(-25,25)]
	#
	d3= pd.DataFrame({'dist':np.array(distance_d03).ravel(),'skg':np.array(skg_d03).ravel()})
	d2= pd.DataFrame({'dist':np.array(distance_d02).ravel(),'skg':np.array(skg_d02).ravel()})
	#
	d3 = d3.groupby('dist').mean().reset_index()
	d2 = d2.groupby('dist').mean().reset_index()
	#
	d3['bin'] = pd.qcut(d3['dist'], q=50)
	d2['bin'] = pd.qcut(d2['dist'], q=50)
	#
	tmp3 = d3.groupby(['bin']).agg(['mean',q10,q90]).reset_index()
	tmp2 = d2.groupby(['bin']).agg(['mean',q10,q90]).reset_index()
	tmp3['bindist'] = [tmp3.bin[i].right for i in range(len(tmp3))]
	tmp2['bindist'] = [tmp2.bin[i].right for i in range(len(tmp2))]
	return tmp3,tmp2


def variogram_obs(n1,n2):
	for i in range(n1,n2):
		f = pd.read_csv(epa_files[i])
		favg = quick_clean(f,lon3,lat3,'M')
		f2 = pd.read_csv(epa_filesd02[i])
		f2 = f2[f2['Latitude'].isin(f.Latitude.unique())].reset_index(drop=True) # crop d02 to d03 lats
		f2avg = quick_clean(f2,lon2,lat2,'M')
		dist_epa = np.array([[dist(favg.Longitude[i],favg.Latitude[i],favg.Longitude[j],favg.Latitude[j]) for i in range(len(favg))] for j in range(len(favg))])
		print(dist_epa)
		skg_sample = np.array([[skg(favg['Sample Measurement'][i],favg['Sample Measurement'][j],len(favg)) for i in range(len(favg))] for j in range(len(favg))])
		d_epa = pd.DataFrame({'dist':np.array(dist_epa).ravel(),'skg':np.array(skg_sample).ravel()})
		if i == n1:
			depa = d_epa; 
		else:
			depa = depa.append(d_epa);
	#
	depa = depa.groupby('dist').mean().reset_index()
	depa['bin'] = pd.qcut(depa['dist'], q=30)
	depa = depa.groupby(['bin']).agg(['mean',q10,q90]).reset_index()
	depa['bindist'] = [depa.bin[i].right for i in range(len(depa))]
	return depa

epano2 = variogram_obs(0,1)
epao3 = variogram_obs(4,5)
epapm25 = variogram_obs(8,9)


tmp3no2,tmp2no2 = variogram(d3,d2,xy,xy2,'NO2')
tmp3o3,tmp2o3 = variogram(d3,d2,xy,xy2,'O3')
tmp3pm25,tmp2pm25 = variogram(d3,d2,xy,xy2,'PM25_TOT')

fig,axs = plt.subplots(3,2,figsize=(8,6))
axs=axs.ravel()

ax = axs[0]
ax.scatter(tmp3no2.bindist,tmp3no2['skg']['mean'],c='blue',label='d03_no2',zorder=10,alpha=0.5)
ax.scatter(tmp2no2.bindist,tmp2no2['skg']['mean'],c='green',label='d02_no2',zorder=10,alpha=0.5)
ax = axs[1]
ax.scatter(epano2.bindist,epano2['skg']['mean']/10,c='black',label='obs_no2',zorder=10,alpha=0.5)


ax = axs[2]
ax.scatter(tmp3o3.bindist,tmp3o3['skg']['mean'],marker='^',c='blue',label='d03_o3',zorder=10,alpha=0.5)
ax.scatter(tmp2o3.bindist,tmp2o3['skg']['mean'],marker='^',c='green',label='d02_o3',zorder=10,alpha=0.5)
ax = axs[3]
ax.scatter(epao3.bindist,epao3['skg']['mean']*10**6,marker='^',c='black',label='obs_o3',zorder=10,alpha=0.5)


ax = axs[4]
ax.scatter(tmp3pm25.bindist,tmp3pm25['skg']['mean'],marker='*',c='blue',label='d03_pm25',zorder=10,alpha=0.5)
ax.scatter(tmp2pm25.bindist,tmp2pm25['skg']['mean'],marker='*',c='green',label='d02_pm25',zorder=10,alpha=0.5)
ax = axs[5]
ax.scatter(epapm25.bindist,epapm25['skg']['mean'],marker='*',c='black',label='obs_pm25',zorder=10,alpha=0.5)

for i in range(6):
	axs[i].set_xlabel('Distance (km)')
	axs[i].set_ylabel('Semivariogram')
	axs[i].grid()
	axs[i].legend()
	#axs[i].set_xlim([0,150])
	#axs[i].set_ylim([0,15])

axs[0].set_xlim([0,150]);axs[0].set_ylim([0,16]);axs[1].set_xlim([0,150]);axs[1].set_ylim([0,5]);
axs[2].set_xlim([0,150]);axs[2].set_ylim([0,31]);axs[3].set_xlim([0,150]);axs[3].set_ylim([0,10]);
axs[4].set_xlim([0,150]);axs[4].set_ylim([0,4]);axs[5].set_xlim([0,150]);axs[5].set_ylim([0,4]);

plt.tight_layout()
plt.savefig('variogram_all.png',dpi=300)
plt.show()

fig,ax = plt.subplots()
ax.scatter(tmp3.dist['mean'],tmp3['skg']['mean'],label='d03',zorder=10,alpha=0.5)
ax.scatter(tmp2.dist['mean'],tmp2['skg']['mean'],label='d02',zorder=10,alpha=0.5)
ax.set_xlabel('Distance (km)')
ax.set_ylabel('Semivariogram')
ax.grid()
plt.legend()
plt.show()



fig,ax = plt.subplots()
ax.scatter(d3.dist,d3.skg,label='d03',zorder=10,alpha=0.5)
ax.scatter(d2.dist,d2.skg,label='d02',zorder=10,alpha=0.5)
ax.set_xlabel('Distance (km)')
ax.set_ylabel('Semivariogram')
ax.grid()
plt.legend()
plt.show()







