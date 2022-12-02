# Model evauation

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


####clean data
def clean(f2,v,mo,shape,ri):
	# f  = file as dataframe
	# v  = variable 
	# mo = month
	# ri = resample type (1h, mo, etc.)
	# shape = shape of grid
	f = f2.copy()
	if (v == 'SO2') & (mo == '8'): # if it's summer 2018, and remove indiana
		if remove_indiana: 
			f = f[f['County Name']!='Lake'].reset_index(drop=True)
			f = f[f['County Name']!='Porter'].reset_index(drop=True)
	if (v == 'SO2') & (mo == '04'): 
		f = f[f['County Name']!='Ingham'].reset_index(drop=True)
		f = f[f['County Name']!='La Salle'].reset_index(drop=True)	
	#
	try:
		if f['Units of Measure'][10] == 'Parts per million': 
			f['Sample Measurement'],f['CMAQ'] = f['Sample Measurement']*1000,f['CMAQ']
		elif f['Units of Measure'][10] == 'ppb': 
			f['Sample Measurement'],f['CMAQ']  = f['Sample Measurement']*1000,f['CMAQ']
		else: 
			f['Sample Measurement'],f['CMAQ']  = f['Sample Measurement'],f['CMAQ']
		#
	except: 
		if epa_units[i] == 'Parts per million':
			f['Sample Measurement'],f['CMAQ']  = f['Sample Measurement']*1000,f['CMAQ']
		else:
			f['Sample Measurement'],f['CMAQ'] = f['Sample Measurement'],f['CMAQ']
	f = f[(f['x'] != 0) & (f['y'] != 0) & (f['y'] != shape[1]) & (f['x'] != shape[0])].reset_index(drop=True)
	try: 
		f['level_0']=pd.to_datetime(f['level_0'])
		f.index = f.level_0
	except: 
		f['dt']=pd.to_datetime(f['dt'])
		f.index = f.dt
	favg = f.groupby(['Longitude','Latitude']).resample(ri).mean().reset_index(drop=True)
	x,y = favg['Sample Measurement'],favg['CMAQ']
	return x,y


####clean data
def clean_dm(f2,v,mo,shape,ri):
	# f  = file as dataframe
	# v  = variable 
	# mo = month
	# ri = resample type (1h, mo, etc.)
	# shape = shape of grid
	f = f2.copy()
	if (v == 'SO2') & (mo == '8'): # if it's summer 2018, and remove indiana
		if remove_indiana: 
			f = f[f['County Name']!='Lake'].reset_index(drop=True)
			f = f[f['County Name']!='Porter'].reset_index(drop=True)
	if (v == 'SO2') & (mo == '04'): 
		f = f[f['County Name']!='Ingham'].reset_index(drop=True)
		f = f[f['County Name']!='La Salle'].reset_index(drop=True)	
	#
	try:
		if f['Units of Measure'][10] == 'Parts per million': 
			f['Sample Measurement'],f['CMAQ'] = f['Sample Measurement']*1000,f['CMAQ']
		elif f['Units of Measure'][10] == 'ppb': 
			f['Sample Measurement'],f['CMAQ']  = f['Sample Measurement']*1000,f['CMAQ']
		else: 
			f['Sample Measurement'],f['CMAQ']  = f['Sample Measurement'],f['CMAQ']
		#
	except: 
		if epa_units[i] == 'Parts per million':
			f['Sample Measurement'],f['CMAQ']  = f['Sample Measurement']*1000,f['CMAQ']
		else:
			f['Sample Measurement'],f['CMAQ'] = f['Sample Measurement'],f['CMAQ']
	f = f[(f['x'] != 0) & (f['y'] != 0) & (f['y'] != shape[1]) & (f['x'] != shape[0])].reset_index(drop=True)
	try: 
		f['level_0']=pd.to_datetime(f['level_0'])
		f.index = f.level_0
	except: 
		f['dt']=pd.to_datetime(f['dt'])
		f.index = f.dt
	favg = f.groupby('Latitude').resample(ri).max().reset_index(drop=True)
	x,y = favg['Sample Measurement'],favg['CMAQ']
	return x,y


####################################################################################################################################
# START CREATING STATS
# Create hourly correlation
####################################################################################################################################

## FOR O3
remove_indiana=True

epa_files  = ['8hrO3_d03_2018_8_EPA_CMAQ_Combine.csv','8hrO3_d03_2018_10_EPA_CMAQ_Combine.csv','8hrO3_d03_2019_1_EPA_CMAQ_Combine.csv','8hrO3_d03_2019_4_EPA_CMAQ_Combine.csv']

epa_files  = [d+'8hrO3_d03_2018_8_EPA_CMAQ_Combine.csv',d+'8hrO3_d03_2018_10_EPA_CMAQ_Combine.csv',d+'8hrO3_d03_2019_1_EPA_CMAQ_Combine.csv',d+'8hrO3_d03_2019_4_EPA_CMAQ_Combine.csv']

corrs = []
for i in range(len(epa_files)):
	print(i)
	f = pd.read_csv(epa_files[i])
	x = f['Mean Including All Data'][~np.isnan(f['Mean Including All Data'])]*1000
	y = f['CMAQ'][~np.isnan(f['Mean Including All Data'])]
	#x,y = x[x>60],y[x>60]
	if len(x) > 0:
		print(len(x))
		mu_d,mu_p,nmb,nme,mb,mse,r,p = stats_normalized(x,y)
		corrs.append([mu_d,mu_p,nmb,nme,mb,mse,r,p])

corrs = pd.DataFrame(corrs)
corrs.columns = ['mu_d','mu_p','nmb','nme','mb','rmse','r','p']


####################################################################################################################################



corrs = []
for i in range(len(epa_files)):
	f = pd.read_csv(epa_files[i])
	f = f[(f['x'] != 0) & (f['y'] != 0) & (f['y'] != shape[1]) & (f['x'] != shape[0])].reset_index(drop=True)
	v = epa_files[i].split(domain)[0].split('/')[-1].split('_')[0]
	mo = epa_files[i].split(domain)[1].split('_')[2]
	ri = '1H'
	x,y = clean(f,v,mo,shape,ri)
	#if (i  == 0) |  (i == 3) : y,x = y[x>60],x[x>60]; #print(np.mean(y-x)/np.mean(x)*100)#;plt.scatter(x,y);print(np.quantile(x,.9))
	print(len(x))
	#y,x = y[y<np.nanquantile(y,.95)],x[y<np.nanquantile(y,.95)]; print(np.mean(y-x)/np.mean(x)*100);plt.scatter(x,y)
	mu_d,mu_p,nmb,nme,mb,rms,r,p = stats_normalized(x,y)
	corrs.append([mu_d,mu_p,nmb,nme,mb,rms,r,p])

corrs = pd.DataFrame(corrs)
corrs.columns = ['mu_d','mu_p','nmb','nme','mb','rmse','r','p']
corrs['Simulation'] = names
corrs
corrs.to_csv('hourly_stn_coefficients_%s.csv'%(domain))

daily_corrs = []
for i in range(len(epa_files)):
	f = pd.read_csv(epa_files[i])
	v = epa_files[i].split(domain)[0].split('/')[-1].split('_')[0]
	mo = epa_files[i].split(domain)[1].split('_')[2]
	ri = '1d'
	x,y = clean(f,v,mo,shape,ri)
	mu_d,mu_p,nmb,nme,r,p = stats_normalized(x,y)
	daily_corrs.append([mu_d,mu_p,nmb,nme,r,p])

daily_corrs = pd.DataFrame(daily_corrs)
daily_corrs.columns = ['mu_d','mu_p','nmb','nme','r','p']
daily_corrs['Simulation'] = names
daily_corrs.to_csv(dir_epa+'daily_stn_coefficients_%s.csv'%(domain))


monthly_corrs = []
for i in range(len(epa_files)):
	f = pd.read_csv(epa_files[i])
	v = epa_files[i].split(domain)[0].split('/')[-1].split('_')[0]
	mo = epa_files[i].split(domain)[1].split('_')[2]
	ri = 'M'
	x,y = clean(f,v,mo,shape,ri)
	mu_d,mu_p,nmb,nme,r,p = stats_normalized(x,y)
	monthly_corrs.append([mu_d,mu_p,nmb,nme,r,p])

monthly_corrs = pd.DataFrame(monthly_corrs)
monthly_corrs.columns = ['mu_d','mu_p','nmb','nme','r','p']
monthly_corrs['Simulation'] = names
monthly_corrs.to_csv(dir_epa+'monthly_stn_coefficients_%s.csv'%(domain))


daily_max_corrs = []
for i in range(len(epa_files)):
	f = pd.read_csv(epa_files[i])
	v = epa_files[i].split(domain)[0].split('/')[-1].split('_')[0]
	mo = epa_files[i].split(domain)[1].split('_')[2]
	ri = 'M'
	x,y = clean_dm(f,v,mo,shape,'1d')
	mu_d,mu_p,nmb,nme,r,p = stats_normalized(x,y)
	daily_max_corrs.append([mu_d,mu_p,nmb,nme,r,p])

daily_max_corrs = pd.DataFrame(daily_max_corrs)
daily_max_corrs.columns = ['mu_d','mu_p','nmb','nme','r','p']
daily_max_corrs['Simulation'] = names
daily_max_corrs.to_csv(dir_epa+'daily_max_stn_coefficients_%s.csv'%(domain))


####################################################################################################################################
# Subset d02 numbers with only d03 stations
# START CREATING STATS
# Create hourly correlation
####################################################################################################################################

# get unique d03 stations
d03lat,d03lon = [],[]
for i in range(len(epa_files)):
	f = pd.read_csv(epa_files[i])
	f['level_0']=pd.to_datetime(f['level_0'])
	f.index = f['level_0']
	d03lat.append(f.Latitude.unique())
	d03lon.append(f.Longitude.unique())

d03lat, d03lon = [d03lat[i][~np.isnan(d03lat[i])] for i in range(len(d03lat))],[d03lon[i][~np.isnan(d03lon[i])] for i in range(len(d03lon))]
nd03 = [len(d03lat[i]) for i in range(len(d03lat))]

s = [len(d03lon[i]) for i in range(len(d03lon))]
nstations = np.array(s).reshape(5,-1).sum(axis=0)

# CMAQ RUN things
domain='d02'
time='hourly'
dir_epa = '/projects/b1045/montgomery/CMAQcheck/'
epa_code=['42401','42602','44201','42101','88101']; var=['SO2','NO2','O3','CO','PM25_TOT'] #numerical identifiers and corresponding vars

years=['2018','2018','2019','2019']
months=['8','10','1','04']

epa_files=[]
names=[]
for i in range(len(var)):
    for m in range(len(months)):
    	month,year = months[m],years[m]
    	epa_files.append([dir_epa+'%s_%s_%s_%s_EPA_CMAQ_Combine.csv'%(var[i],domain,year,month)])
    	names.append('%s_%s_%s_%s'%(var[i],domain,year,month))

epa_files = np.array(epa_files).ravel(); names= np.array(names).ravel()


# get unique d02 stations
d02lon,d02lat = [],[]
for i in range(len(epa_files)):
	f = pd.read_csv(epa_files[i])
	f['level_0']=pd.to_datetime(f['level_0'])
	f.index = f['level_0']
	d02lat.append(f.Latitude.unique())
	d02lon.append(f.Longitude.unique())

d02lat, d02lon = [d02lat[i][~np.isnan(d02lat[i])] for i in range(len(d02lat))],[d02lon[i][~np.isnan(d02lon[i])] for i in range(len(d02lon))]
nd02 = [len(d02lat[i]) for i in range(len(d02lat))]

# get hourly correlations
corrs = []
for i in range(len(epa_files)):
	f = pd.read_csv(epa_files[i])
	if (i == 0) or (i == 1): # if it's summer 2018, and remove indiana
		if remove_indiana: 
			f = f[f['County Name']!='Lake'].reset_index(drop=True)
			f = f[f['County Name']!='Porter'].reset_index(drop=True)
	if i== 3: # if spring	
		f = f[f['County Name']!='Ingham'].reset_index(drop=True)
		f = f[f['County Name']!='La Salle'].reset_index(drop=True)
	f['level_0']=pd.to_datetime(f['level_0'])
	f = f[f['Latitude'].isin(d03lat[i])].reset_index(drop=True)
	if f['Units of Measure'][10] == 'Parts per million': x,y = f['Sample Measurement']*1000,f['CMAQ']
	else: x,y = f['Sample Measurement'],f['CMAQ']
	mu_d,mu_p,nmb,nme,r,p = stats_normalized(x,y)
	corrs.append([mu_d,mu_p,nmb,nme,r,p])

corrs = pd.DataFrame(corrs)
corrs.columns = ['mu_d','mu_p','nmb','nme','r','p']
corrs['Simulation'] = names
corrs.to_csv(dir_epa+'hourly_stn_coefficients_%s_withind03.csv'%(domain))

# create daily correlation
daily_corrs = []
for i in range(len(epa_files)):
	f = pd.read_csv(epa_files[i])
	if (i == 0) or (i == 1): # if it's summer 2018, and remove indiana
		if remove_indiana:
			f = f[f['County Name']!='Lake'].reset_index(drop=True)
			f = f[f['County Name']!='Porter'].reset_index(drop=True)
	if i== 3: # if spring	
		f = f[f['County Name']!='Ingham'].reset_index(drop=True)
		f = f[f['County Name']!='La Salle'].reset_index(drop=True)
	f = f[f['Latitude'].isin(d03lat[i])].reset_index(drop=True)
	f['level_0']=pd.to_datetime(f['level_0'])
	if f['Units of Measure'][10] == 'Parts per million': f['Sample Measurement'],f['CMAQ'] = f['Sample Measurement']*1000,f['CMAQ']
	f.index = f['level_0']
	favg = f.groupby('Latitude').resample('D').mean()
	favg=favg.reset_index(level=0, drop=True).reset_index()
	x,y = favg['Sample Measurement'],favg['CMAQ']
	mu_d,mu_p,nmb,nme,r,p = stats_normalized(x,y)
	daily_corrs.append([mu_d,mu_p,nmb,nme,r,p])

daily_corrs = pd.DataFrame(daily_corrs)
daily_corrs.columns = ['mu_d','mu_p','nmb','nme','r','p']
daily_corrs['Simulation'] = names
daily_corrs.to_csv(dir_epa+'daily_stn_coefficients_%s_withind03.csv'%(domain))

# create month correlation
monthly_corrs = []
for i in range(len(epa_files)):
	f = pd.read_csv(epa_files[i])
	if (i == 0) or (i == 1): # if it's summer 2018, and remove indiana
		if remove_indiana: 
			f = f[f['County Name']!='Lake'].reset_index(drop=True)
			f = f[f['County Name']!='Porter'].reset_index(drop=True)
	if i== 3: # if spring	
		f = f[f['County Name']!='Ingham'].reset_index(drop=True)
		f = f[f['County Name']!='La Salle'].reset_index(drop=True)
	f = f[f['Latitude'].isin(d03lat[i])].reset_index(drop=True)
	f['level_0']=pd.to_datetime(f['level_0'])
	if f['Units of Measure'][10] == 'Parts per million': f['Sample Measurement'],f['CMAQ'] = f['Sample Measurement']*1000,f['CMAQ']
	f.index = f['level_0']
	favg = f.groupby('Latitude').resample('M').mean()
	favg=favg.reset_index(level=0, drop=True).reset_index()
	x,y = favg['Sample Measurement'],favg['CMAQ']
	mu_d,mu_p,nmb,nme,r,p = stats_normalized(x,y)
	monthly_corrs.append([mu_d,mu_p,nmb,nme,r,p])

monthly_corrs = pd.DataFrame(monthly_corrs)
monthly_corrs.columns = ['mu_d','mu_p','nmb','nme','r','p']
monthly_corrs['Simulation'] = names
monthly_corrs.to_csv(dir_epa+'monthly_stn_coefficients_%s_withind03.csv'%(domain))

# create daily max correlation
daily_max_corrs = []
for i in range(len(epa_files)):
	f = pd.read_csv(epa_files[i])
	if (i == 0) or (i == 1): # if it's summer 2018, and remove indiana
		if remove_indiana: 
			f = f[f['County Name']!='Lake'].reset_index(drop=True)
			f = f[f['County Name']!='Porter'].reset_index(drop=True)
	if i== 3: # if spring	
		f = f[f['County Name']!='Ingham'].reset_index(drop=True)
		f = f[f['County Name']!='La Salle'].reset_index(drop=True)
	f = f[f['Latitude'].isin(d03lat[i])].reset_index(drop=True)
	f['level_0']=pd.to_datetime(f['level_0'])
	if f['Units of Measure'][10] == 'Parts per million': f['Sample Measurement'],f['CMAQ'] = f['Sample Measurement']*1000,f['CMAQ']
	f.index = f['level_0']
	favg = f.groupby('Latitude').resample('D')
	x,y = favg['Sample Measurement'].max(),favg['CMAQ'].max()
	x,y = x.reset_index(),y.reset_index()
	x,y = x['Sample Measurement'],y['CMAQ']
	mu_d,mu_p,nmb,nme,r,p = stats_normalized(x,y)
	daily_max_corrs.append([mu_d,mu_p,nmb,nme,r,p])

daily_max_corrs = pd.DataFrame(daily_max_corrs)
daily_max_corrs.columns = ['mu_d','mu_p','nmb','nme','r','p']
daily_max_corrs['Simulation'] = names
daily_max_corrs.to_csv(dir_epa+'daily_max_stn_coefficients_%s_withind03.csv'%(domain))

#
# combine
corrs = pd.read_csv(dir_epa+'hourly_stn_coefficients_d03.csv',index_col=0)
corrsd02 = pd.read_csv(dir_epa+'hourly_stn_coefficients_d02_withind03.csv',index_col=0)
corrs.Simulation = corrsd02.Simulation

####################################################################################################################################
# Make figures of correlations
# Read in stats 
d03='hourly_stn_coefficients_d03.csv','daily_stn_coefficients_d03.csv','monthly_stn_coefficients_d03.csv','daily_max_stn_coefficients_d03.csv'
d02='hourly_stn_coefficients_d02_withind03.csv','daily_stn_coefficients_d02_withind03.csv','monthly_stn_coefficients_d02_withind03.csv','daily_max_stn_coefficients_d02_withind03.csv'

corrs = pd.read_csv(dir_epa+d02[0]); daily_corrs = pd.read_csv(dir_epa+d02[1]); daily_max_corrs = pd.read_csv(dir_epa+d02[3]);  monthly_corrs = pd.read_csv(dir_epa+d02[2]); 
titles = ['avg','pearsonr','nmb','nme']

corrs.mu_d[12:16]=corrs.mu_d[12:16]/10;corrs.mu_p[12:16]=corrs.mu_p[12:16]/10
daily_corrs.mu_d[12:16]=corrs.mu_d[12:16]/10;daily_corrs.mu_p[12:16]=corrs.mu_p[12:16]/10
daily_max_corrs.mu_d[12:16]=corrs.mu_d[12:16]/10;daily_max_corrs.mu_p[12:16]=corrs.mu_p[12:16]/10
monthly_corrs.mu_d[12:16]=corrs.mu_d[12:16]/10;monthly_corrs.mu_p[12:16]=corrs.mu_p[12:16]/10

import matplotlib.markers as markers
a,b,aa,bb = markers.CARETUPBASE,markers.CARETDOWNBASE,markers.CARETLEFTBASE,markers.CARETRIGHTBASE
a,b,aa,bb = 'h','H','o','.'
a,b,aa,bb = '^','<','>','v'
a,b,aa,bb = 'o','o','o','o'

s1,s2,s3,s4 = 40,32,23,15
x= corrs.Simulation
ticklabels = ['Summer','Fall','Winter','Spring']*5

colors = ['#601204', '#c75f24', '#868569', '#617983']
colors = ['#ba5126','#ffd255','#53a9b6','k']
fig,ax = plt.subplots(1,3,figsize=(10,3.5))
ax.ravel()

#ax[0].scatter(x,corrs.mu_d,label='EPA Average',marker='o',alpha=1,c='#ba5226')
#ax[0].scatter(x,corrs.mu_p,label='Pixel Average',marker='o',alpha=1,c='#11537a')
#ax[0].set_xticklabels(x,rotation = 90)
#ax[0].legend()
#ax[0].set_title('d03')
#ax[0].set_title(titles[0])

ax[0].scatter(x,corrs.r, label='hourly',alpha=1,marker=a,c=colors[0],s=s1)
ax[0].scatter(x,daily_corrs.r,label='daily',alpha=1,marker=b,c=colors[1],s=s2)
ax[0].scatter(x,daily_max_corrs.r,label='daily max',alpha=1,marker=aa,c=colors[2],s=s3)
ax[0].scatter(x,monthly_corrs.r,label='monthly',alpha=1,marker=bb,c=colors[3],s=s4)
ax[0].set_xticklabels(ticklabels,rotation = 90)
#ax[1].set_yticks([])
#ax[0].legend()
#ax[1].set_title('d03')
#ax[0].set_title('pearson r')

ax[1].scatter(x,corrs.nmb, label='hourly',alpha=1,marker=a,c=colors[0],s=s1)
ax[1].scatter(x,daily_corrs.nmb,label='daily',marker=b,c=colors[1],s=s2)
ax[1].scatter(x,daily_max_corrs.nmb,label='daily max',alpha=1,marker=aa,c=colors[2],s=s3)
ax[1].scatter(x,monthly_corrs.nmb,label='monthly',alpha=1,marker=bb,c=colors[3],s=s4)
ax[1].set_xticklabels(ticklabels,rotation = 90)
#ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)
#ax[2].set_yticks([])
#ax[2].set_title('d03')
#ax[1].set_title('nmb')

ax[2].scatter(x,corrs.nme,label='hourly',alpha=1,marker=a,c=colors[0],s=s1)
ax[2].scatter(x,daily_corrs.nme,label='daily',marker=b,c=colors[1],s=s2)
ax[2].scatter(x,daily_max_corrs.nme,label='daily max',alpha=1,marker=aa,c=colors[2],s=s3)
ax[2].scatter(x,monthly_corrs.nme,label='monthly',alpha=1,marker=bb,c=colors[3],s=s4)
ax[2].set_xticklabels(ticklabels,rotation = 90)
#ax[2].legend()
#ax[3].set_title('d03')
#ax[2].set_title('nme')


#ylims = [[0,50],[0,1],[-25,100],[0,100]]
ylims = [[0,1],[-25,100],[0,100]]
q = 0
for ax in ax: 
	ax.set_xlim([-.5,19.5])
	a = np.arange(ylims[q][0]-1,ylims[q][1]+1)
	for i in range(5): ax.plot([15.5-i*4]*len(a),a,alpha=0.5,c='gray');
	ax.set_ylim(ylims[q])
	#ax.set_xticks([])
	q=q+1

# add line to demarcate groupings
#for ax in ax: ax.plot(np.arange(-1,51),[15.5]*52,alpha=0.5,c='gray');
plt.tight_layout()
plt.savefig('r_nme_nmb_d02-LEGEND.pdf')
plt.show()



# Compare do2 do3 pearsonr
labels=['Hourly ','Daily ','Monthly ']
fig,ax = plt.subplots(3,1,figsize=(6,12))
for i in range(len(d03)):
	d03hourly,d02hourly=pd.read_csv(dir_epa+d03[i],index_col=0),pd.read_csv(dir_epa+d02[i],index_col=0)
	ax[i].scatter(d03hourly.Simulation, d03hourly.r,label=labels[i]+'d03')
	ax[i].scatter(d03hourly.Simulation,d02hourly.r,label=labels[i]+'d02',marker='^')
	#ax[i].quiver(np.arange(0,len(d03hourly.r)), d02hourly.r, [0]*len(d03hourly.r), (d03hourly.r-d02hourly.r), angles='xy', scale_units='xy', scale=1)
	ax[i].legend()
	ax[i].set_ylabel('pearson r')

empty_string_labels = ['']*len(d02hourly)
ax[0].set_xticklabels(empty_string_labels);ax[1].set_xticklabels(empty_string_labels);
[ax[i].set_ylim([0,1]) for i in range(len(ax))]
plt.xticks(rotation = 90)
#plt.tight_layout()
plt.savefig(dir_epa+'compared02_d03_pearsonr_noarrows.png')
plt.show()

# Compare do2 do3 averages
labels=['Hourly ','Daily ','Monthly ']
fig,ax = plt.subplots(3,1,figsize=(6,12))
for i in range(len(d03)-1):
	d03hourly,d02hourly=pd.read_csv(dir_epa+d03[i],index_col=0),pd.read_csv(dir_epa+d02[i],index_col=0)
	ax[i].scatter(d03hourly.Simulation, d03hourly.nmb,label=labels[i]+'d03')
	ax[i].scatter(d03hourly.Simulation,d02hourly.nmb,label=labels[i]+'d02',marker='^')
	#ax[i].quiver(np.arange(0,len(d03hourly.r)), d02hourly.r, [0]*len(d03hourly.r), (d03hourly.r-d02hourly.r), angles='xy', scale_units='xy', scale=1)
	ax[i].legend()
	ax[i].set_ylabel('nmb')

empty_string_labels = ['']*len(d02hourly)
ax[0].set_xticklabels(empty_string_labels);ax[1].set_xticklabels(empty_string_labels);
[ax[i].set_ylim([-50,50]) for i in range(len(ax))]
plt.xticks(rotation = 90)
plt.tight_layout()
plt.savefig(dir_epa+'compared02_d03_nmb_noarrows.png')
plt.show()


# Compare do2 do3 averages
labels=['Hourly ','Daily ','Monthly ']
fig,ax = plt.subplots(3,1,figsize=(6,12))
for i in range(len(d03)-1):
	d03hourly,d02hourly=pd.read_csv(dir_epa+d03[i],index_col=0),pd.read_csv(dir_epa+d02[i],index_col=0)
	ax[i].scatter(d03hourly.Simulation, d03hourly.nme,label=labels[i]+'d03')
	ax[i].scatter(d03hourly.Simulation,d02hourly.nme,label=labels[i]+'d02',marker='^')
	#ax[i].quiver(np.arange(0,len(d03hourly.r)), d02hourly.r, [0]*len(d03hourly.r), (d03hourly.r-d02hourly.r), angles='xy', scale_units='xy', scale=1)
	ax[i].legend()
	ax[i].set_ylabel('nme')

empty_string_labels = ['']*len(d02hourly)
ax[0].set_xticklabels(empty_string_labels);ax[1].set_xticklabels(empty_string_labels);
[ax[i].set_ylim([0,100]) for i in range(len(ax))]
plt.xticks(rotation = 90)
plt.tight_layout()
plt.savefig(dir_epa+'compared02_d03_nme_noarrows.png')
plt.show()


# Look at station covergage
# d02 vs d03 stations
crs_new = ccrs.PlateCarree()# get shape outside
fig,ax = plt.subplots(5,4,subplot_kw={'projection': crs_new},figsize=(10,10))
ax = ax.ravel()
states = cfeature.NaturalEarthFeature(category='cultural',name='admin_1_states_provinces_lines',edgecolor='gray',facecolor='none',scale='10m',alpha = 0.3)
borders = cfeature.NaturalEarthFeature(scale='50m',category='cultural',name='admin_0_countries',edgecolor='gray',facecolor='lightgray',alpha=0.3)
lakes = cfeature.NaturalEarthFeature('physical', 'lakes', '50m', facecolor=cfeature.COLORS['water'])
edge=0.5
for i in range(len(ax)):
	ax[i].scatter(d02lon[i],d02lat[i],s=order=10,c='k')
	ax[i].scatter(d03lon[i],d03lat[i],marker='^',s=8,zorder=10,c='crimson')
	ax[i].add_feature(states)
	ax[i].add_feature(borders)
	ax[i].add_feature(lakes)
	ax[i].set_xlim([lon.min()+edge,lon.max()-edge])
	ax[i].set_ylim([lat.min()+edge,lat.max()-edge])
#

plt.tight_layout()
plt.savefig('d02_d03_station_coverage.png')
plt.show()


####################################################################################################################################
# Make diurnal profile
def makefig(t,x3,x2,y3,y2,var):
    fig,ax = plt.subplots(figsize = (4,3))
    ax.plot(t,x3,linewidth = 1, label='d03')
    ax.plot(t,x2,linewidth = 1, label='d02')
    ax.plot(t,y2,linestyle='--',linewidth = 1, label = 'AQSd02')
    ax.plot(t,y3,linestyle='--',linewidth = 1, label = 'AQSd03')
    ax.grid()
    ax.set_xlabel('Hour of Day (GMT)')
    ax.set_ylabel(var)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    #plt.legend()
    plt.savefig(var+'_alld03_diurnal_CMAQ_avg.png',dpi = 300)
    plt.close()

for i in range(len(epa_files)):
	f = pd.read_csv(epa_files[i])
	f2 = pd.read_csv(epa_filesd02[i])
	f = f[(f['x'] != 0) & (f['y'] != 0) & (f['y'] != shape[1]) & (f['x'] != shape[0])].reset_index(drop=True)
	f2 = f2[(f2['x'] != 0) & (f2['y'] != 0) & (f2['y'] != shape[1]) & (f2['x'] != shape[0])].reset_index(drop=True)
	v = epa_files[i].split(domain)[0].split('/')[-1].split('_')[0]
	f.dt = pd.to_datetime(f.dt); f2.dt = pd.to_datetime(f2.level_0)
	f['hr'] = [f.dt[i].hour for i in range(len(f))]
	f2['hr'] = [f2.dt[i].hour for i in range(len(f2))]
	fh = f.groupby('hr').mean()
	fh2 = f2.groupby('hr').mean()
	if v == 'O3': fh['Sample Measurement'],fh2['Sample Measurement'] = fh['Sample Measurement']*1000,fh2['Sample Measurement']*1000
	if v == 'CO': fh['Sample Measurement'],fh2['Sample Measurement'] = fh['Sample Measurement']*1000,fh2['Sample Measurement']*1000
	makefig(np.arange(0,24),fh.CMAQ,fh2.CMAQ,fh['Sample Measurement'],fh2['Sample Measurement'],v)
