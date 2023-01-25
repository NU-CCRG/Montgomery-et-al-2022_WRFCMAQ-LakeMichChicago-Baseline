#model_evaluation_plots.py
#funfetti plot

####################################################################################################################################

import pandas as pd
from scipy.stats import pearsonr
import numpy as np
from sklearn.metrics import mean_squared_error
import scipy.stats as st
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from cartopy import crs as ccrs;
import cartopy.feature as cfeature 
#
####################################################################################################################################

# Make figures of correlations
# Read in stats 
dir_epa = '/projects/b1045/montgomery/CMAQcheck/'
d03='hourly_stn_coefficients_d03.csv','daily_stn_coefficients_d03.csv','monthly_stn_coefficients_d03.csv','daily_max_stn_coefficients_d03.csv'
d02='hourly_stn_coefficients_d02_withind03.csv','daily_stn_coefficients_d02_withind03.csv','monthly_stn_coefficients_d02_withind03.csv','daily_max_stn_coefficients_d02_withind03.csv'

# model evaluation but 2nd way
dir_epa = '/projects/b1045/montgomery/CMAQcheck/'
dir='/projects/b1045/jschnell/ForStacy/' 
ll='latlon_ChicagoLADCO_d03.nc' 
lon,lat = np.array(Dataset(dir+ll)['lon']),np.array(Dataset(dir+ll)['lat'])

corrs = pd.read_csv(dir_epa+d03[0]); daily_corrs = pd.read_csv(dir_epa+d03[1]); daily_max_corrs = pd.read_csv(dir_epa+d03[3]);  monthly_corrs = pd.read_csv(dir_epa+d03[2]); 
titles = ['avg','pearsonr','nmb','nme']
corrsd02 = pd.read_csv(dir_epa+d02[2]);daily_corrsd02 = pd.read_csv(dir_epa+d02[1]); daily_max_corrsd02 = pd.read_csv(dir_epa+d02[3]);  monthly_corrsd02 = pd.read_csv(dir_epa+d02[2]); 

#t = corrs.T;t2 = corrsd02.T
#cols = t.columns.tolist()
#cols = cols[4:8] + cols[8:12]+cols[16:20]+cols[0:4] + cols[12:16]
#corrs,corrsd02 = t[cols].T,t2[cols].T
#daily_corrs,daily_corrsd02 = daily_corrs.T[cols].T,daily_corrsd02.T[cols].T
#daily_max_corrs,daily_max_corrsd02 = daily_max_corrs.T[cols].T,daily_max_corrsd02.T[cols].T
#monthly_corrs,monthly_corrsd02=monthly_corrs.T[cols].T,monthly_corrsd02.T[cols].T


yearly_corrs = [np.mean(np.array(monthly_corrs.r).reshape(-1,4),axis=1),np.mean(np.array(monthly_corrs.nmb).reshape(-1,4),axis=1),np.mean(np.array(monthly_corrs.nme).reshape(-1,4),axis=1)]+ [np.mean(np.array(corrsd02.r).reshape(-1,4),axis=1),np.mean(np.array(corrsd02.nmb).reshape(-1,4),axis=1),np.mean(np.array(corrsd02.nme).reshape(-1,4),axis=1)]


# CMAQ RUN things
domain='d03'
time='hourly'
vv = ['NO2']*4+['O3']*4+['PM25_TOT']*4+['SO2']*4+['CO']*4
var = ['NO2','O3','PM25_TOT','SO2','CO']
years=['2018','2018','2019','2019']
months=['8','10','1','04']
days=['31','31','31','30']*4
months_start=['7','9','12','03']*4
years_start = ['2018','2018','2018','2019']*4
yy = years*4; mm = months*4
ss = sim*4

domain = 'd03'
names=[]
for i in range(len(var)):
    for m in range(len(months)):
    	month,year = months[m],years[m]
    	names.append('%s_%s_%s_%s'%(var[i],domain,year,month))


#corrs.mu_d[12:16]=corrs.mu_d[12:16]/10;corrs.mu_p[12:16]=corrs.mu_p[12:16]/10
#daily_corrs.mu_d[12:16]=corrs.mu_d[12:16]/10;daily_corrs.mu_p[12:16]=corrs.mu_p[12:16]/10
#daily_max_corrs.mu_d[12:16]=corrs.mu_d[12:16]/10;daily_max_corrs.mu_p[12:16]=corrs.mu_p[12:16]/10
#monthly_corrs.mu_d[12:16]=corrs.mu_d[12:16]/10;monthly_corrs.mu_p[12:16]=corrs.mu_p[12:16]/10

import matplotlib.markers as markers
a,b,aa,bb = markers.CARETUPBASE,markers.CARETDOWNBASE,markers.CARETLEFTBASE,markers.CARETRIGHTBASE
a,b,aa,bb = 'h','H','o','.'
a,b,aa,bb = '^','<','>','v'
a,b,aa,bb = 'o','o','o','o'

s1,s2,s3,s4 = 50-9,40-10,30-9,15-6
x= np.array(corrs.Simulation)
x=names

colors = ['#601204', '#c75f24', '#868569', '#617983']
colors = ['#ba5126','#ffd255','#53a9b6','k']

fig,ax = plt.subplots(3,2,figsize=(8,6))
ax=ax.T.ravel()
title = r'NO$_2$       O$_3$       PM$_{2.5}$          SO$_2$       CO'

axs=ax[0]
axs.scatter(x,corrs.r, label='hourly',alpha=1,marker=a,c=colors[0],s=s1)
axs.scatter(x,daily_corrs.r,label='daily',alpha=1,marker=b,c=colors[1],s=s2)
axs.scatter(x,daily_max_corrs.r,label='daily max',alpha=1,marker=aa,c=colors[2],s=s3)
axs.scatter(x,monthly_corrs.r,label='monthly',alpha=1,marker=bb,c=colors[3],s=s4)
#ax[1].set_xticklabels(x,rotation = 90)
#ax[1].set_yticks([])
#axs.legend()
#ax[1].set_title('d03')r'PM$_{2.5}$']
axs.set_ylabel('(a) pearson r')
axs.set_title(title, fontsize=10)

axs=ax[1]
axs.scatter(x,corrs.nmb, label='hourly',alpha=1,marker=a,c=colors[0],s=s1)
axs.scatter(x,daily_corrs.nmb,label='daily',marker=b,c=colors[1],s=s2)
axs.scatter(x,daily_max_corrs.nmb,label='daily max',alpha=1,marker=aa,c=colors[2],s=s3)
axs.scatter(x,monthly_corrs.nmb,label='monthly',alpha=1,marker=bb,c=colors[3],s=s4)
#ax[2].set_xticklabels(x,rotation = 90)
#axs.legend()
#ax[2].set_yticks([])
#ax[2].set_title('d03')
axs.set_ylabel('(b) nmb (%)')

axs=ax[2]
axs.scatter(x,corrs.nme,label='hourly',alpha=1,marker=a,c=colors[0],s=s1)
axs.scatter(x,daily_corrs.nme,label='daily',marker=b,c=colors[1],s=s2)
axs.scatter(x,daily_max_corrs.nme,label='daily max',alpha=1,marker=aa,c=colors[2],s=s3)
axs.scatter(x,monthly_corrs.nme,label='monthly',alpha=1,marker=bb,c=colors[3],s=s4)
#ax[3].set_xticklabels(x,rotation = 90)
#axs.legend()
#ax[3].set_title('d03')
axs.set_ylabel('(c) nme (%)')


corrs = corrsd02
titles = ['avg','pearsonr','nmb','nme']

#corrs.mu_d[12:16]=corrs.mu_d[12:16]/10;corrs.mu_p[12:16]=corrs.mu_p[12:16]/10
#daily_corrs.mu_d[12:16]=corrs.mu_d[12:16]/10;daily_corrs.mu_p[12:16]=corrs.mu_p[12:16]/10
#daily_max_corrs.mu_d[12:16]=corrs.mu_d[12:16]/10;daily_max_corrs.mu_p[12:16]=corrs.mu_p[12:16]/10
#monthly_corrs.mu_d[12:16]=corrs.mu_d[12:16]/10;monthly_corrs.mu_p[12:16]=corrs.mu_p[12:16]/10

axs=ax[3]
axs.scatter(x,corrs.r, label='hourly',alpha=1,marker=a,c=colors[0],s=s1)
axs.scatter(x,daily_corrs.r,label='daily',alpha=1,marker=b,c=colors[1],s=s2)
axs.scatter(x,daily_max_corrs.r,label='daily max',alpha=1,marker=aa,c=colors[2],s=s3)
axs.scatter(x,monthly_corrs.r,label='monthly',alpha=1,marker=bb,c=colors[3],s=s4)
#ax[1].set_xticklabels(x,rotation = 90)
#ax[1].set_yticks([])
#axs.legend()
#ax[1].set_title('d03')
axs.set_ylabel('pearson r')
axs.set_title(title, fontsize=10)

axs=ax[4]
axs.scatter(x,corrs.nmb, label='hourly',alpha=1,marker=a,c=colors[0],s=s1)
axs.scatter(x,daily_corrs.nmb,label='daily',marker=b,c=colors[1],s=s2)
axs.scatter(x,daily_max_corrs.nmb,label='daily max',alpha=1,marker=aa,c=colors[2],s=s3)
axs.scatter(x,monthly_corrs.nmb,label='monthly',alpha=1,marker=bb,c=colors[3],s=s4)
#ax[2].set_xticklabels(x,rotation = 90)
#axs.legend()
#ax[2].set_yticks([])
#ax[2].set_title('d03')
axs.set_ylabel('nmb (%)')

axs=ax[5]
axs.scatter(x,corrs.nme,label='hourly',alpha=1,marker=a,c=colors[0],s=s1)
axs.scatter(x,daily_corrs.nme,label='daily',marker=b,c=colors[1],s=s2)
axs.scatter(x,daily_max_corrs.nme,label='daily max',alpha=1,marker=aa,c=colors[2],s=s3)
axs.scatter(x,monthly_corrs.nme,label='monthly',alpha=1,marker=bb,c=colors[3],s=s4)
#ax[3].set_xticklabels(x,rotation = 90)
#axs.legend()
#ax[3].set_title('d03')
axs.set_ylabel('nme (%)')

# set up axis
#ylims = np.array([[0,50],[0,1],[-25,100],[0,100],[0,50],[0,1],[-25,100],[0,100]]).T

ylims = np.array([[0,1],[-130,130],[0,130],[0,1],[-130,130],[0,130]])

q = 0

#axs=ax
for q in range(len(ax)):
	x=ax[q] 
	x.set_xlim([-.5,19.5])
	t = np.arange(ylims[q][0]-1,ylims[q][1]+1)
	for i in range(len(ax)): x.plot([15.5-i*4]*len(t),t,alpha=0.5,c='gray');
	x.set_ylim(ylims[q])
	print(ylims[q])
	x.set_xticks([])
	#q=q+1

for i in range(len(ax)):
	x = np.arange(0,20).reshape(-1,4)
	y = yearly_corrs[i]
	for j in range(len(x)):
		ax[i].plot(x[j],[y[j]]*len(x[j]),c='k')


ax[2].set_xticks(np.arange(0,20)); ax[2].set_xticklabels(['Aug. 18','Oct. 18','Jan. 19','Apr. 19']*5,rotation = 90,fontsize=9)
ax[5].set_xticks(np.arange(0,20)); ax[5].set_xticklabels(['08/18','10/18','01/19','04/19']*5,rotation = 90,fontsize=9)


for i in range(len(ax)):
	ax[i].set_ylim(ylims[i])

# add line to demarcate groupings
#for ax in ax: ax.plot(np.arange(-1,51),[15.5]*52,alpha=0.5,c='gray');

ax[1].axhline(0,c='lightgrey',zorder=0.1);ax[4].axhline(0,c='lightgrey',zorder=0.1);
#corrs.mu_d[12:16]=corrs.mu_d[12:16]*10;corrs.mu_p[12:16]=corrs.mu_p[12:16]*10
#corrsd02.mu_d[12:16]=corrsd02.mu_d[12:16]*10;corrsd02.mu_p[12:16]=corrs.mu_p[12:16]*10

plt.tight_layout()
plt.savefig('Fig2_CMAQcheck.png',dpi=300)
plt.show()



