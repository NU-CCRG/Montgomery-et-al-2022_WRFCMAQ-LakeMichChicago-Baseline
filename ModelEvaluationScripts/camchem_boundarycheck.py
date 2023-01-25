#check_CAMChem_ozonesonde.py

# data = https://gml.noaa.gov/aftp/data/ozwv/Ozonesonde/Boulder,%20Colorado/100%20Meter%20Average%20Files/
# 

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
import matplotlib
matplotlib.use('Agg')
####################################################################################################################################



# adapted from : http://kbkb-wx-python.blogspot.com/2016/08/find-nearest-latitude-and-longitude.html
def find_index(stn_lon, stn_lat, wrf_lon, wrf_lat):
# stn -- points 
# wrf -- list
#for iz in range(1):
        xx=[];yy=[]
        for i in range(len(stn_lat)):
                #for i in range(1):
                abslat = np.abs(wrf_lat-stn_lat[i])
                abslon= np.abs(wrf_lon-stn_lon[i])
                c = np.maximum(abslon,abslat)
                latlon_idx = np.argmin(c)
                x, y = np.where(c == np.min(c))
                #add indices of nearest wrf point station
                xx.append(x[0])
                yy.append(y[0])
        #
        #xx=[xx[i][0] for i in range(len(xx))];
        #yy=[yy[i][0] for i in range(len(yy))]
        #return indices list
        return xx, yy

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
        p,r = np.corrcoef(x,y)[0]
        rms = rmse(y,x)
        mb = np.sum(y-x)/len(x)
        return mu_d,mu_p,nmb,nme,mb,rms,r,p


# Start
####################################################################################################################################

# CMAQ RUN things
domain='d02'
time='hourly'
dir_epa = '/projects/b1045/montgomery/CMAQcheck/'
epa_code=['44201']; var=['O3'] #numerical identifiers and corresponding vars

years=['2018','2018','2019','2019']
months=['8','10','1','04']
days= ['31','31','31','30']


# Start
####################################################################################################################################
d02 = Dataset('/home/asm0384/lat_lon_chicago_d02.nc')
lat,lon = d02['LAT'][0][0].data,d02['LON'][0][0].data

latedges = list(lat[0])+list(lat.T[0])+list(lat[-1]) + list(lat.T[-1])
lonedges = list(lon[0])+list(lon.T[0])+list(lon[-1]) + list(lon.T[-1])
lledge = [(np.min(latedges),np.min(lonedges)),(np.min(latedges),np.max(lonedges)),(np.max(latedges),np.max(lonedges)),(np.max(latedges),np.min(lonedges))]
lledgeUS = [(np.min(latedges)-1,np.min(lonedges)-1),(np.min(latedges)-1,np.max(lonedges)+1),(np.max(latedges)+1,np.max(lonedges)+1),(np.max(latedges)+1,np.min(lonedges)-1)]
def compute_stats_cam(epa_hourly_fname,month,year,day,cam_chem_file):
        epa_hourly = pd.read_csv(epa_hourly_fname)
        #
        st = 5 # degrees in which to test the edges
        epa_hourly = epa_hourly[epa_hourly['Qualifier'].isna()].reset_index(drop=True) # remove any flagged data
        epa_hourly_US = epa_hourly[((epa_hourly.Latitude > np.max(latedges)) | (epa_hourly.Latitude < np.min(latedges))) | ((epa_hourly.Longitude > np.max(lonedges)) | (epa_hourly.Longitude < np.min(lonedges)))].reset_index(drop=True)
        epa_hourly_US = epa_hourly_US[((epa_hourly_US.Latitude < np.max(latedges)+st) & (epa_hourly_US.Latitude > np.min(latedges)-st)) & ((epa_hourly_US.Longitude < np.max(lonedges)+st) & (epa_hourly_US.Longitude > np.min(lonedges)-st))].reset_index(drop=True)
        epa_hourly_US['dt'] = [epa_hourly_US['Date GMT'][i] + ' ' + epa_hourly_US['Time GMT'][i] for i in range(len(epa_hourly_US))]
        epa_hourly_US['dt'] = pd.to_datetime(epa_hourly_US['dt'])
        epa_hourly_US['lonlat'] = [(epa_hourly_US.Longitude[i],epa_hourly_US.Latitude[i]) for i in range(len(epa_hourly_US))]
        tmp = epa_hourly_US['lonlat'].unique()
        epalat = [tmp[i][1] for i in range(len(tmp))]
        epalon = [tmp[i][0] for i in range(len(tmp))]
        #
        # make lines around
        #
        times_perf = pd.date_range(start='%s-%s-%s'%(month,'01',year),end='%s-%s-%s 23:00:00'%(month,day,year),freq='6H')
        epa_hourly_US = epa_hourly_US.set_index('dt').groupby(['Latitude','Longitude']).resample('6h').mean()
        #
        # read in cam-chem
        c8 = Dataset(cam_chem_file)
        c8o3 = c8['O3']
        clat,clon = np.meshgrid(c8['lat'],c8['lon'])
        xx,yy = find_index(epalat,epalon,clat,clon-360)
        ok = pd.DataFrame([xx,yy,[1]*len(xx)]).T
        print('Number of Unique Pixels: '+str(ok.groupby([0,1]).sum().shape[0]))
        print('Number of Unique EPA: '+str(len(xx)))
        # Correct for time slicing smh
        if len(times_perf) < len(c8o3):
                endidx = len(times_perf)
        else:
                endidx = len(c8o3)
                times_perf = times_perf[:len(c8o3)]
        #
        for i in range(len(epalat)):
                depa = epa_hourly_US['Sample Measurement'][epalat[i], epalon[i]].reindex(times_perf).fillna(np.nan)
                depa = np.array(depa)*1000
                co3 = np.array(c8o3[0:endidx,0,yy[i],xx[i]])*10**7
                data = pd.DataFrame({'dt':times_perf,'epa':depa,'cam':co3,'lat':epalat[i],'lon':epalon[i]})
                if i == 0: data_full = data
                else: data_full = data_full.append(data)
        #
        data_full.loc[data_full.epa < 0,'epa'] = np.nan
        data_full = data_full[~np.isnan(data_full.epa)].reset_index(drop=True)
        d = data_full.set_index('dt').resample('d').mean().reset_index(drop=True)
        dma = data_full.set_index('dt').resample('d').max().reset_index(drop=True)
        dmi = data_full.set_index('dt').resample('d').min().reset_index(drop=True)
        mo = data_full.set_index('dt').groupby(['lat','lon']).resample('M').mean().reset_index(drop=True)
        #
        hourly6 = stats_normalized(data_full.epa,data_full.cam)
        daily = stats_normalized(d.epa,d.cam)
        dailymax = stats_normalized(dma.epa,dma.cam)
        dailymin = stats_normalized(dmi.epa,dmi.cam)
        #
        return daily,dailymax,dailymin,monthly,hourly6,epalat,epalon,epa_hourly_US.Longitude.unique(),epa_hourly_US.Latitude.unique()

# run for all
epa_hourly_fname = '/projects/b1045/montgomery/CMAQcheck/hourly_%s_2018.csv'%(epa_code[0])
cam_chem_file = '/projects/b1045/wrf-cmaq/input/CAM-CHEM/ChicagoLADCO/camchem-20221205113358656561.nc'
cam_chem_file = '/projects/b1045/wrf-cmaq/input/CAM-CHEM/ChicagoLADCO/camchem-20221218162651779761.nc'

# August
daily8,dailymax8,dailymin8,monthly8,hourly8,epalat,epalon,clon,clat = compute_stats_cam(epa_hourly_fname,months[0],years[0],days[0],cam_chem_file)

#Oct
cam_chem_file = '/projects/b1045/wrf-cmaq/input/CAM-CHEM/ChicagoLADCO/camchem-20221218162659630768.nc'
#cam_chem_file = '/projects/b1045/wrf-cmaq/input/CAM-CHEM/ChicagoLADCO/camchem-20221218162651779761.nc'
daily10,dailymax10,dailymin10,monthly10,hourly10,epalat,epalon,clon,clat = compute_stats_cam(epa_hourly_fname,months[1],years[1],days[1],cam_chem_file)

# january 
cam_chem_file = '/projects/b1045/wrf-cmaq/input/CAM-CHEM/ChicagoLADCO/camchem-20221218162712127679.nc'
epa_hourly_fname = '/projects/b1045/montgomery/CMAQcheck/hourly_%s_2019.csv'%(epa_code[0])
daily1,dailymax1,dailymin1,monthly1,hourly1,epalat,epalon,clon,clat = compute_stats_cam(epa_hourly_fname,months[2],years[2],days[2],cam_chem_file)

#april
#cam_chem_file = '/projects/b1045/wrf-cmaq/input/CAM-CHEM/ChicagoLADCO/camchem-20221218162659630768.nc'
cam_chem_file='/projects/b1045/wrf-cmaq/input/MOZART/Chicago_LADCO/camchem-20211101120011669984.april.nc'
daily4,dailymax4,dailymin4,monthly4,hourly4,epalat,epalon,clon,clat = compute_stats_cam(epa_hourly_fname,months[3],years[3],days[3],cam_chem_file)


hourly = pd.DataFrame([hourly8,hourly10,hourly1,hourly4])
daily = pd.DataFrame([daily8,daily10,daily1,daily4])
dailymax = pd.DataFrame([dailymax8,dailymax10,dailymax1,dailymax4])
dailymin = pd.DataFrame([dailymin8,dailymin10,dailymin1,dailymin4])
monthly = pd.DataFrame([monthly8,monthly10,monthly1,monthly4])

columns = ['epa','cam','nmb','nme','mb','rms','r','p']
hourly.columns = columns; daily.columns = columns; dailymax.columns = columns;dailymin.columns = columns; monthly.columns = columns
hourly.to_csv('hourly_camchem_o3.csv');daily.to_csv('hourly_camchem_o3.csv');dailymax.to_csv('hourly_camchem_o3.csv');
dailymin.to_csv('hourly_camchem_o3.csv');monthly.to_csv('hourly_camchem_o3.csv');


#----------------------------------------------------------------------------------------------------
# create figure object
# original projection so that we can transform it to lambert

from cartopy import crs as ccrs;
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.io.img_tiles import Stamen

crs_new = ccrs.PlateCarree()
crs =  ccrs.PlateCarree()

fig, ax = plt.subplots(subplot_kw={'projection': crs},figsize=(4.5,4))

tiler = Stamen(style='terrain-background')
ax.add_image(tiler, 8)
ax.coastlines('10m',linewidth=0.5)
ax.add_feature(cfeature.LAND,facecolor='None',edgecolor='k',linewidth=0.5)
states_provinces = cfeature.NaturalEarthFeature(category='cultural',name='admin_1_states_provinces_lines',scale='50m',facecolor='none')
ax.add_feature(states_provinces, edgecolor='gray',linewidth = 0.5)
ax.add_feature(cfeature.BORDERS, edgecolor='gray',linewidth = 0.5)

#ax.scatter(clon,clat,label='O3 Monitors')
ax.scatter(epalon,epalat,label='Comparison O3 Monitors',alpha=0.6, c='k', s=10,marker='^',zorder=10)
ax.scatter(lonedges,latedges,c='k',label='d02 Domain Edge',zorder=10,s=20)

ax.set_title(r'O$_3$'+' Stations (n = '+ str(len(epalon)) + ') \n around 4 km Domain for CAM-Chem Boundaries')

gl = ax.gridlines(crs=crs_new, draw_labels=True,linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size': 8, 'color': 'gray'}
gl.ylabel_style = {'size': 8, 'color': 'gray'}
gl.xlabels_top = False
gl.ylabels_left = False
gl.xlines = False
gl.ylines = False

plt.savefig('o3_stations_camchem.png',dpi = 300)
#plt.show()
