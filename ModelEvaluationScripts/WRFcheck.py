
import pandas as pd
from scipy.stats import pearsonr
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from netCDF4 import Dataset
####################################################################################################################################

# functions
def stats_normalized(data,prediction):
    x,y=data[~np.isnan(data)],prediction[~np.isnan(data)] # get rid of NaNs
    mu_d,mu_p = np.mean(x),np.mean(y)
    nmb = np.sum(y-x)/np.sum(x)*100
    nme = np.sum(np.abs(y-x))/np.sum(x)*100
    nrmse = np.sqrt(1/len(x)*np.sum((y-x)**2))/np.mean(x)
    r,p = st.pearsonr(x,y)
    return mu_d,mu_p,nmb,nme,r,p

def find_index(stn_lon, stn_lat, wrf_lon, wrf_lat):
# stn -- points 
# wrf -- list
        stn_lon=stn_lon[~np.isnan(stn_lon)]
        stn_lat=stn_lat[~np.isnan(stn_lat)]
        #for iz in range(1):
        xx=[];yy=[]
        for i in range(len(stn_lat)):
                abslat = np.abs(wrf_lat-stn_lat[i])
                abslon= np.abs(wrf_lon-stn_lon[i])
                c = np.maximum(abslon,abslat)
                latlon_idx = np.argmin(c)
                x, y = np.where(c == np.min(c))
                #add indices of nearest wrf point station
                xx.append(x)
                yy.append(y)
        #
        xx=[xx[i][0] for i in range(len(xx))];yy=[yy[i][0] for i in range(len(yy))]
        #return indices list
        return xx, yy


# functions # NOT normalized
def stats_normalized(data,prediction):
    x,y=data[~np.isnan(data)],prediction[~np.isnan(data)] # get rid of NaNs
    mu_d,mu_p = np.mean(x),np.mean(y)
    mb = np.sum(y-x)/len(x)
    ge = np.mean(np.abs(x-y))
    rmse = np.sqrt(np.square(np.subtract(x,y)).mean())
    r,p = st.pearsonr(x,y)
    return mu_d,mu_p,mb,ge,rmse,r,p


def makefig(obs_d02,obs_d03,d02,d03,var):
    var = var.split('output_BASE_FINAL_')[1]
    obs_d02 = obs_d02.reset_index()
    obs_d03 = obs_d03.reset_index()
    d02 = d02.reset_index()
    d03 = d03.reset_index()
    obs_d02.columns = ['dt','x']; obs_d03.columns = ['dt','x']
    d02.columns = ['dt','x']; d03.columns = ['dt','x']
    obs_d02['hr'] = [obs_d02.dt[i].hour for i in range(len(obs_d02))]
    obs_d03['hr'] = [obs_d03.dt[i].hour for i in range(len(obs_d03))]
    d03['hr'] = [d03.dt[i].hour for i in range(len(d03))]
    d02['hr'] = [d02.dt[i].hour for i in range(len(d02))]
    fig,ax = plt.subplots(figsize = (4,3))
    ax.plot(obs_d03.groupby('hr').mean()['x'],linestyle='--',linewidth = 1, label='NCDC')
    ax.plot(d02.groupby('hr').mean()['x'],linewidth = 1, label = 'd02')
    ax.plot(d03.groupby('hr').mean()['x'],linewidth = 1, label='d03')
    ax.grid()
    ax.set_xlabel('Hour of Day (GMT)')
    ax.set_ylabel(var)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    #plt.legend()
    plt.savefig(var+'_ohare_diurnal_avg.png',dpi = 300)
    plt.close()
    fig,ax = plt.subplots(figsize = (6,3))
    ax.plot(obs_d03['x'], linewidth = 1, linestyle='--', label='NCDC')
    ax.plot(d02['x'], linewidth = 1, label = 'd02')
    ax.plot(d03['x'], linewidth = 1, label='d03')
    ax.set_xlabel('Hour of Simulation')
    ax.set_ylabel(var)
    ax.grid()
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(var+'_ohare_diurnal_all.png',dpi = 300)
    plt.close()
d02 = Dataset('/home/asm0384/lat_lon_chicago_d02.nc')
lat,lon = d02['LAT'][0][0].data,d02['LON'][0][0].data

# CMAQ RUN things
domain='d03'
time='hourly'
dir_WRF = '/projects/b1045/montgomery/WRFcheck/'
#epa_code=['42401','42602','44201','42101','88101']; var=['SO2','NO2','O3','CO','PM25_TOT'] #numerical identifiers and corresponding vars
sim = ['output_BASE_FINAL_1.33km_sf_rrtmg_5_8_1_v3852','output_BASE_FINAL_fall_1.33km_sf_rrtmg_5_8_1_v3852','output_BASE_FINAL_wint_1.33km_sf_rrtmg_5_8_1_v3852','output_BASE_FINAL_spring_1.33km_sf_rrtmg_5_8_1_v3852']
var = ['t2','rh','winds','winddir']
v2 = ['RH','Wind','WindDir']

years=['2018','2018','2019','2019']
months=['8','10','1','04']
days=['31','31','31','30']*4
months_start=['7','9','12','03']*4
years_start = ['2018','2018','2018','2019']*4
yy = years*4; mm = months*4
ss = sim*4

col = ['mu_d','mu_p','b','ge','rmse','r','p']

# Format file names
wrf_files_d03 = [dir_WRF+sim[i]+'/'+var+'d03.csv' for var in var for i in range(len(sim))]
wrf_files_d02 = [dir_WRF+sim[i]+'/'+var+'d02.csv' for var in var for i in range(len(sim))]
station_files = [dir_WRF+sim[i]+'/wrfcheck_withstations_%s_%s.csv'%(sim[i],var) for var in v2 for i in range(len(sim))]
station_files=['/projects/b1045/montgomery/WRFcheck/output_BASE_FINAL_1.33km_sf_rrtmg_5_8_1_v3852/wrfcheck_withstations_output_BASE_FINAL_1.33km_sf_rrtmg_5_8_1_v3852_082018.csv',
                                '/projects/b1045/montgomery/WRFcheck/output_BASE_FINAL_fall_1.33km_sf_rrtmg_5_8_1_v3852/wrfcheck_withstations_output_BASE_FINAL_fall_1.33km_sf_rrtmg_5_8_1_v3852_102018.csv',
                                '/projects/b1045/montgomery/WRFcheck/output_BASE_FINAL_wint_1.33km_sf_rrtmg_5_8_1_v3852/wrfcheck_withstations_output_BASE_FINAL_wint_1.33km_sf_rrtmg_5_8_1_v3852_012019.csv',
                                '/projects/b1045/montgomery/WRFcheck/output_BASE_FINAL_spring_1.33km_sf_rrtmg_5_8_1_v3852/wrfcheck_withstations_output_BASE_FINAL_spring_1.33km_sf_rrtmg_5_8_1_v3852_042019.csv']+station_files


fin = Dataset('/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_FINAL_1.33km_sf_rrtmg_5_8_1_v3852/wrfout_d01_2018-08-01_00:00:00')
shd03 = fin['T2'][0].shape

fin=Dataset('/projects/b1045/wrf-cmaq/output/Chicago_LADCO/output_BASE_FINAL_4km_sf_rrtmg_10_8_1_v3852/wrfout_d01_2018-08-01_00:00:00')
shd02 = fin['T2'][0].shape


d02_hourly,d03_hourly = [],[]
d02_daily,d03_daily = [],[]
d02_monthly,d03_monthly = [],[]
d02_dailymax,d03_dailymax = [],[]

for f in range(len(wrf_files_d02)):
        stn = pd.read_csv(station_files[f],index_col=0)
        stn['ohare'] = [False for i in range(len(stn))]
        #stn.ohare[89] = True
        m = (stn.yy_d02 > 0) & (stn.xx_d02 > 0) & (stn.yy_d02 < shd02[1]) & (stn.xx_d02 < shd02[0])
        #m = stn.ohare
        #stn=stn[stn['yy_d02']>0].reset_index(drop=True)#stn=stn[stn['xx_d02']>0].reset_index(drop=True)#stn=stn[stn['yy_d02']<shd02[1]].reset_index(drop=True)#stn=stn[stn['xx_d02']<shd02[0]].reset_index(drop=True)i
        stn = stn[m].reset_index(drop=True)
        times = pd.read_csv(dir_WRF+ss[f]+'/completeddata_mini_extras2.csv',index_col=0)
        times_perf = pd.date_range(start='%s-%s-%s'%(mm[f],'01',yy[f]),end='%s-%s-%s 23:00:00'%(mm[f],days[f],yy[f]),freq='H')
        stn_d03 = stn[stn.in_d03==True].reset_index()
        obs_d02 = pd.DataFrame([stn[str(i)].tolist() for i in range(len(times))]).T
        obs_d03 = pd.DataFrame([stn_d03[str(i)].tolist() for i in range(len(times))]).T
        d03 = pd.read_csv(wrf_files_d03[f],index_col=0).T
        d02 = pd.read_csv(wrf_files_d02[f],index_col=0).T
        d02 = d02[list(m)].reset_index(drop=True)
        obs_d02=obs_d02[stn.in_d03==True].reset_index(drop=True)
        d02=d02[stn.in_d03==True].reset_index(drop=True)
        d02=d02.T; d03=d03.T#format for time column
        # subsetting to make 3rd column == o'hare 
        d03=pd.DataFrame(d03['3']); d02=pd.DataFrame(d02[3]); obs_d02 = pd.DataFrame(obs_d02.T[3].T); obs_d03= pd.DataFrame(obs_d03.T[3].T)
        # convert K to F
        if f < 4:
                #d02 = (d02 - 273.15)*9/5 + 32
                #d03 = (d03 - 273.15)*9/5 + 32  
        #
                obs_d02 = (obs_d02 - 32)*5/9 + 273.15
                obs_d03 = (obs_d03 - 32)*5/9 + 273.15
        #
        if (f >= 8) & (f < 12):
                d02 = 2.236936*d02
                d03 = 2.236936*d03
        #
        #
        times = np.array(pd.to_datetime(times['0'])[0:len(times)])
#       obs_d03=obs_d03.T; obs_d02=obs_d02.T; 
        d02['dt']=np.array(times_perf)[0:len(d02)+1]; d03['dt']=np.array(times_perf)[0:len(d03)+1]
        obs_d02['dt']=times; obs_d03['dt']=times
        #pd.merge(obs_d02,d02,on='dt', how='outer', indicator=True)
        # Mask dor times
        intimemask = (obs_d02['dt']>=times_perf[0]) & (obs_d02['dt']<=times_perf[-1])
        obs_d02=obs_d02[intimemask].reset_index(drop=True); obs_d03=obs_d03[intimemask].reset_index(drop=True);
        intimemask = (d02['dt']>=times[0]) & (d02['dt']<=times[-1])
        d02=d02[intimemask].reset_index(drop=True); d03=d03[intimemask].reset_index(drop=True);
        # cross check
        intimemask = (obs_d02['dt']<=np.array(d02.dt)[-1])
        obs_d02=obs_d02[intimemask].reset_index(drop=True); obs_d03=obs_d03[intimemask].reset_index(drop=True);
        #
        obs_d02,obs_d03,d02,d03=obs_d02.set_index('dt'),obs_d03.set_index('dt'),d02.set_index('dt'),d03.set_index('dt');
        #resample if missing observation data: 
        obs_d02=obs_d02.resample('h').mean();obs_d03=obs_d03.resample('h').mean();
        # make figure with diurnal profiles
        vari =wrf_files_d02[f].split('/')[-2].split('1.33km')[0]+ wrf_files_d02[f].split('/')[-1].split('d02')[0]
        makefig(obs_d02,obs_d03,d02,d03,vari)
        #
        d02_hourly.append(stats_normalized(np.array(obs_d02).ravel(),np.array(d02).ravel()))
        d03_hourly.append(stats_normalized(np.array(obs_d03).ravel(),np.array(d03).ravel()))
        #
        #
        x,y = np.array(obs_d02.resample('D').mean()).ravel(),np.array(d02.resample('D').mean()).ravel()
        d02_daily.append(stats_normalized(x,y))
        x,y = np.array(obs_d03.resample('D').mean()).ravel(),np.array(d03.resample('D').mean()).ravel()
        d03_daily.append(stats_normalized(x,y))
        d02_hourly.append(stats_normalized(np.array(obs_d02).ravel(),np.array(d02).ravel()))
        d03_hourly.append(stats_normalized(np.array(obs_d03).ravel(),np.array(d03).ravel()))
        #
        #
        x,y = np.array(obs_d02.resample('D').mean()).ravel(),np.array(d02.resample('D').mean()).ravel()
        d02_daily.append(stats_normalized(x,y))
        x,y = np.array(obs_d03.resample('D').mean()).ravel(),np.array(d03.resample('D').mean()).ravel()
        d03_daily.append(stats_normalized(x,y))
        x,y = np.array(obs_d02.resample('M').mean()).ravel(),np.array(d02.resample('M').mean()).ravel()
        d02_monthly.append(stats_normalized(x,y))
        x,y = np.array(obs_d03.resample('M').mean()).ravel(),np.array(d03.resample('M').mean()).ravel()
        d03_monthly.append(stats_normalized(x,y))
        x,y = np.array(obs_d02.resample('D').max()).ravel(),np.array(d02.resample('D').max()).ravel()
        d02_dailymax.append(stats_normalized(x,y))
        x,y = np.array(obs_d03.resample('D').max()).ravel(),np.array(d03.resample('D').max()).ravel()
        d03_dailymax.append(stats_normalized(x,y))
        #

domain = 'd03'
names=[]
for i in range(len(var)):
    for m in range(len(months)):
        month,year = months[m],years[m]
        names.append('%s_%s_%s_%s_ohare'%(var[i],domain,year,month))


d = [d02_hourly,d03_hourly,d02_daily,d03_daily,d02_monthly,d03_monthly,d02_dailymax,d03_dailymax]
nameout=['hourly_stn_coefficients']*2+['daily_stn_coefficients']*2+['monthly_stn_coefficients']*2+['dailymax_stn_coefficients']*2
domain=['d02','d03']*4
for i in range(len(d)):
        dout = pd.DataFrame(d[i])
        dout.columns = col # ['mu_d','mu_p','nmb','nme','r','p']
        dout['Simulation'] = names
        dout.to_csv(dir_WRF+'%s_%s_ohare.csv'%(nameout[i],domain[i]))

#
d = [d02_hourly,d03_hourly,d02_daily,d03_daily,d02_monthly,d03_monthly,d02_dailymax,d03_dailymax]
nameout=['hourly_stn_coefficients']*2+['daily_stn_coefficients']*2+['monthly_stn_coefficients']*2+['dailymax_stn_coefficients']*2
domain=['d02','d03']*4
for i in range(len(d)):
        dout = pd.DataFrame(d[i])
        dout.columns = col # ['mu_d','mu_p','nmb','nme','r','p']
        dout['Simulation'] = names


