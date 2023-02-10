# -*- coding: utf-8 -*-
#==============================================================================
#--- @author: Honda, R. H., on: Fri Feb 10 15:17:08 2023
#==============================================================================
import RINEX2_Pandas_spyder as rnx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime

# =============================================================================
# Definindo um filtro passa banda ButterWorth
# =============================================================================
from scipy.signal import butter, lfilter, filtfilt

import IPP3 as iopp
import read_sp3d as sp3
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

#Plotting
plt.rcParams.update({'font.size': 20})
fig = plt.figure(figsize=(20,20))


ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines(resolution='110m',color='k')
ax.add_feature(cfeature.BORDERS)
extend = [-70,-30,-50,-10]
ax.set_extent(extend, crs=ccrs.PlateCarree())

ax.text(0.5, 1.02, 'Porto Alegre-RS, 16/09/2015 - 16-20h', va='bottom', ha='center',
        rotation='horizontal', rotation_mode='anchor',
        transform=ax.transAxes)
ax.text(-0.07, 0.55, 'Latitude', va='bottom', ha='center',
        rotation='vertical', rotation_mode='anchor',
        transform=ax.transAxes)
ax.text(0.5, -0.05, 'Longitude', va='bottom', ha='center',
        rotation='horizontal', rotation_mode='anchor',
        transform=ax.transAxes)


gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2,color='gray', alpha=0.5, linestyle=':')
gl.top_labels = False
gl.right_labels = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

# prn_rtp=sp3.coord_spherical(Inter_prn,T_input,long_sort=False,cartesian=False)

for Prn_name in sta_prns:
    # Prn_name='G01'
    min_wanted=7
    
    Inter_prn=sp3.data_interpol(Df_sp3d,Prn_name)

    prn_data=Df_RINEX[Df_RINEX['prn']==Prn_name]
    
    TEC_obs = prn_data['TEC'].values
    TEC_fit = prn_data['TEC'].rolling(2*min_wanted*4,center=True).mean().values
    dTEC = TEC_obs - TEC_fit
    
    T_input=prn_data['horario_s']
    
    prn_xyz=sp3.coord_spherical(Inter_prn,T_input,long_sort=False,cartesian=True)


    # df_ipp=calculating_ipp(sta_xyz,prn_xyz)
    lat,lon,elev=iopp.ipp_calculator(prn_xyz,sta_xyz,sta_rtp)
    
    #Indices para se calcular as trajetorias.
    idx=np.isfinite(dTEC)
    lon=lon[idx]
    lat=lat[idx]
    dTEC_Butter=butter_bandpass_filter(dTEC[idx],15,30,240,order=5)
    x_axis=T_input[idx]
    
    
    
    
    idxa=abs(dTEC_Butter)>0.025
    
    elev = elev[idx]
    elev = elev[idxa]
    
    idxe = elev>20
    
    # print(len(idxe)==len(idxa))
    
    # ax.scatter(x_axis[idxa],y_axis[idxa],marker='.',s=1,c='r')



    x_axis = lon
    y_axis = lat
    ax.scatter(x_axis,y_axis, marker ='.',s=1,c='k',alpha=.1)
    ax.scatter(x_axis[idxa],y_axis[idxa],marker='x',s=1,c='r')

x = sta_rtp['lon_deg']
y = sta_rtp['lat_deg']

ax.scatter(x, y,color='b',s=200)
plt.savefig('POAL15_259_displacement_{}.png'.format(min_wanted), dpi=300)