# #Add Tolls folder into library
# import sys
# # sys.path.insert(0, 'D:\Dropbox\Python\spyder_home\Tools') #Windows
# sys.path.insert(0, '/home/hisashi/Desktop/Dropbox/Python/spyder_home/Tools/') #Linux

import RINEX2_Pandas_spyder as rnx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime

# =============================================================================
# Definindo um filtro passa banda ButterWorth
# =============================================================================
from scipy.signal import butter, lfilter, filtfilt

def butter_bandpass(lowcut, highcut, frequency_sample, order=5):
    nyq = 0.5 * frequency_sample
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, frequency_sample, order=5):
    b, a = butter_bandpass(lowcut, highcut, frequency_sample, order=order)
    y = lfilter(b, a, data)
#     y = filtfilt(b, a, data)
    return y



L1=1575.42e6
L2=1227.60e6
C_Middle = (L1**2)*(L2**2)/((L1**2) - (L2**2))

CTE=299792458.*C_Middle/(10**16 * 40.3)




# rinex_file='D:\Dropbox\Python\spyder_home\Files\poal2591.15o' #Windows
rinex_file='/home/hisashi/Dropbox/Python/spyder_home/Files/poal2591.15o' #Linux

#Lendo o arquivo RINEX
Df_RINEX,sta_rtp,sta_xyz,sta_prns = rnx.rinex2_csv(rinex_file, generate_csv=False)

# Extraindo os valores das colunas
cols=Df_RINEX.columns.drop(['horario','horario_s','prn'])

# Convertendo os valores para float
Df_RINEX[cols]=Df_RINEX[cols].apply(pd.to_numeric, errors = 'coerce')

# Calculando os valores de TEC
Df_RINEX['TEC'] = CTE*(Df_RINEX['L1']/L1-Df_RINEX['L2']/L2)


# =============================================================================
# 
# =============================================================================
# R_names = 'R01 R02 R03 R04 R05 R06 R07 R08 R09 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27'.split()
channel = [1, -4, 5, 6, 1, -4, 5,6, -2, -7, 0,-1,-2,-7,0,-1,4,-3,3,2,4,-3,3,2,np.nan,-6]

# idx=Df_RINEX['prn'].str[:]==R_names[0]
Df_index=Df_RINEX.index

for i in Df_index:
    # if Df_RINEX.loc[i]['prn'][0]=='G':
    #     L1=1575.42e6
    #     L2=1227.60e6
    #     C_Middle = (L1**2)*(L2**2)/((L1**2) - (L2**2))

    #     CTE=299792458.*C_Middle/(10**16 * 40.3)
        
    #     Df_RINEX.loc[i]['TEC']=CTE*(Df_RINEX.loc[i]['L1']/L1-Df_RINEX.loc[i]['L2']/L2)
        
    #     print(Df_RINEX.loc[i]['TEC'])
        
    if Df_RINEX.loc[i]['prn'][0]=='R':
        num=int(Df_RINEX.loc[i]['prn'][1:])-1
        
        # print(num)
        
        L1 = 1602e6 + channel[num] * 562.5e3
        L2 = 1246e6 + channel[num] * 437.5e3
        
        # print(L1,L2)
        C_Middle = (L1**2)*(L2**2)/((L1**2) - (L2**2))

        CTE=299792458.*C_Middle/(10**16 * 40.3)
        
        # print(Df_RINEX.loc[i]['TEC'])
        
        Df_RINEX.loc[i,'TEC']=CTE*(Df_RINEX.loc[i,'L1']/L1-Df_RINEX.loc[i,'L2']/L2)
        # print(i)
        
        # print(CTE*(Df_RINEX.loc[i]['L1']/L1-Df_RINEX.loc[i]['L2']/L2))






# Filtrando os horarios de interesse
t_min = 16
t_max = 20

t_min = datetime(2015, 9, 16, t_min, 0, 0)
t_max = datetime(2015, 9, 16, t_max, 0, 0)

Df_RINEX = Df_RINEX[(Df_RINEX['horario'] >= t_min) & (Df_RINEX['horario'] <= t_max)]

#Filtrando apenas os valores de G
# sta_prns=sta_prns[pd.Series(sta_prns).str.contains('G').values] #o sinal de ~ inverte os valores booleanos

#Eliminar os PRNs que estao com informação quebrada
sta_prns=sta_prns[sta_prns != 'G06']
sta_prns=sta_prns[sta_prns != 'G27']
sta_prns=sta_prns[sta_prns != 'G08']
sta_prns=sta_prns[sta_prns != 'G16']
sta_prns=sta_prns[sta_prns != 'G11']
sta_prns=sta_prns[sta_prns != 'G29']
sta_prns=sta_prns[sta_prns != 'G14']
sta_prns=sta_prns[sta_prns != 'G12']
sta_prns=sta_prns[sta_prns != 'G31']
sta_prns=sta_prns[sta_prns != 'G21']
sta_prns=sta_prns[sta_prns != 'G25']
sta_prns=sta_prns[sta_prns != 'G26']
sta_prns=sta_prns[sta_prns != 'G18']
sta_prns=sta_prns[sta_prns != 'G22']
sta_prns=sta_prns[sta_prns != 'G32']
sta_prns=sta_prns[sta_prns != 'G04']
sta_prns=sta_prns[sta_prns != 'G02']
sta_prns=sta_prns[sta_prns != 'G24']
sta_prns=sta_prns[sta_prns != 'G20']
sta_prns=sta_prns[sta_prns != 'G15']
sta_prns=sta_prns[sta_prns != 'G01']
sta_prns=sta_prns[sta_prns != 'G03']

sta_prns=sta_prns[sta_prns != 'R20']
sta_prns=sta_prns[sta_prns != 'R10']
sta_prns=sta_prns[sta_prns != 'R21']
sta_prns=sta_prns[sta_prns != 'R04']
sta_prns=sta_prns[sta_prns != 'R11']
sta_prns=sta_prns[sta_prns != 'R22']
sta_prns=sta_prns[sta_prns != 'R12']
sta_prns=sta_prns[sta_prns != 'R08']
sta_prns=sta_prns[sta_prns != 'R23']
sta_prns=sta_prns[sta_prns != 'R01']
sta_prns=sta_prns[sta_prns != 'R02']




# Figure 1

# plt.rcParams.update({'font.size': 20})
# fig,axs = plt.subplots(len(sta_prns),1,figsize=(20,30),sharex=True)
# #-------------------------------
# for i in range(len(sta_prns)):
#     prn_data=Df_RINEX[Df_RINEX['prn']==sta_prns[i]]
#     # print(i)
    
#     x_axis=prn_data['horario_s']
#     y_axis=prn_data['TEC']

#     axs[i].plot(x_axis,y_axis,label=sta_prns[i])
#     axs[i].grid(visible=True)
#     axs[i].legend(loc='upper center')

# plt.tight_layout()
# plt.savefig('POAL15_259_TEC.pdf', dpi=300)



# # =============================================================================
# # #Polinomial Detrend
# # =============================================================================

# # plt.rcParams.update({'font.size': 14})
# fig,axs = plt.subplots(len(sta_prns),1,figsize=(20,60),sharex=True)
# #-------------------------------
# for i in range(len(sta_prns)):
#     prn_data=Df_RINEX[Df_RINEX['prn']==sta_prns[i]]
#     # print(i)
#     #-----1st plot
#     x_obs = prn_data['horario_s'].values
#     y_obs = prn_data['TEC'].values
    
#     x_axis=x_obs
#     y_axis=y_obs

#     axs[i].plot(x_axis,y_axis,label=sta_prns[i])
 
#     #-----2st plot
#     #-Excluding NaN Values
#     idx = np.isfinite(x_obs) & np.isfinite(y_obs)

#     #-Extracting the polinomials coefficients
#     poly = np.polyfit(x_obs[idx], y_obs[idx], 2)
#     y_fit = np.polyval(poly, x_obs)

#     dy_a=y_obs-y_fit
#     x_axis=x_obs
#     y_axis=y_fit

#     axs[i].plot(x_axis,y_axis)
    
#     axs[i].grid(visible=True)
#     axs[i].legend(loc='upper center')

# plt.tight_layout()
# plt.savefig('POAL15_259_TEC_fit.pdf', dpi=300)

# # # =============================================================================
# # # # Aqui eu passo os filtros tanto o ButterWorth quanto o Threshold
# # # =============================================================================
# plt.rcParams.update({'font.size': 18})
# # fig,axs = plt.subplots(len(sta_prns),1,figsize=(20,30),sharex=True,sharey=True)
# fig,axs = plt.subplots(len(sta_prns),1,figsize=(20,60))
# #-------------------------------
# for i in range(len(sta_prns)):
#     prn_data=Df_RINEX[Df_RINEX['prn']==sta_prns[i]]
#     # print(i)
#     #-----1st plot
#     min_wanted=7
#     x_obs = prn_data['horario_s'].values
#     y_obs = prn_data['TEC'].values
#     y_fit = prn_data['TEC'].rolling(2*min_wanted*4,center=True).mean().values
    
#     x_axis=x_obs
#     y_axis=y_obs

#     dy_a=y_obs-y_fit
#     x_axis=x_obs
#     # y_axis=dy_a
    
#     # y_axis=y_obs
#     # axs[i].plot(x_axis,y_axis,label=sta_prns[i])
#     # y_axis=y_fit
#     # axs[i].plot(x_axis,y_axis,label=sta_prns[i])
    
#     # y_axis=dy_a
#     # axs[i].plot(x_axis,y_axis,label=sta_prns[i])
    
#     idx=np.isfinite(dy_a)
#     y_axis=butter_bandpass_filter(dy_a[idx],15,30,240,order=5)
#     idxa=abs(y_axis)>0.00025
#     x_axis=x_axis[idx]
#     axs[i].plot(x_axis[idxa],y_axis[idxa],'-o',color='k',label=sta_prns[i])
    
#     axs[i].grid(visible=True)
#     axs[i].legend()

# plt.tight_layout()
# plt.savefig('POAL15_259_TEC_{}min.pdf'.format(min_wanted), dpi=300)

# # =============================================================================
# # # Figura com o mapa
# # =============================================================================
import IPP3 as iopp
import read_sp3d as sp3
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
# =============================================================================
# Lendo arquivo sp3
# =============================================================================
# file_path='D:\Dropbox\Python\spyder_home\sp3_cache\ESA0MGNFIN_20152590000_01D_05M_ORB.sp3'
File_path='/home/hisashi/Dropbox/Python/spyder_home/sp3_cache/ESA0MGNFIN_20152590000_01D_05M_ORB.sp3'
Df_sp3d, prn_names_sp3 = sp3.read_sp3(File_path)


# #Plotting
# plt.rcParams.update({'font.size': 20})
# fig = plt.figure(figsize=(20,20))


# ax = plt.axes(projection=ccrs.PlateCarree())
# ax.coastlines(resolution='110m',color='k')
# ax.add_feature(cfeature.BORDERS)
# extend = [-70,-30,-50,-10]
# ax.set_extent(extend, crs=ccrs.PlateCarree())

# ax.text(0.5, 1.02, 'Porto Alegre-RS, 16/09/2015 - 16-20h', va='bottom', ha='center',
#         rotation='horizontal', rotation_mode='anchor',
#         transform=ax.transAxes)
# ax.text(-0.07, 0.55, 'Latitude', va='bottom', ha='center',
#         rotation='vertical', rotation_mode='anchor',
#         transform=ax.transAxes)
# ax.text(0.5, -0.05, 'Longitude', va='bottom', ha='center',
#         rotation='horizontal', rotation_mode='anchor',
#         transform=ax.transAxes)


# gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
#                   linewidth=2,color='gray', alpha=0.5, linestyle=':')
# gl.top_labels = False
# gl.right_labels = False
# gl.xformatter = LONGITUDE_FORMATTER
# gl.yformatter = LATITUDE_FORMATTER

# # prn_rtp=sp3.coord_spherical(Inter_prn,T_input,long_sort=False,cartesian=False)

# for Prn_name in sta_prns:
#     # Prn_name='G01'
#     min_wanted=7
    
#     Inter_prn=sp3.data_interpol(Df_sp3d,Prn_name)

#     prn_data=Df_RINEX[Df_RINEX['prn']==Prn_name]
    
#     TEC_obs = prn_data['TEC'].values
#     TEC_fit = prn_data['TEC'].rolling(2*min_wanted*4,center=True).mean().values
#     dTEC = TEC_obs - TEC_fit
    
#     T_input=prn_data['horario_s']
    
#     prn_xyz=sp3.coord_spherical(Inter_prn,T_input,long_sort=False,cartesian=True)


#     # df_ipp=calculating_ipp(sta_xyz,prn_xyz)
#     lat,lon,elev=iopp.ipp_calculator(prn_xyz,sta_xyz,sta_rtp)
    
#     #Indices para se calcular as trajetorias.
#     idx=np.isfinite(dTEC)
#     lon=lon[idx]
#     lat=lat[idx]
#     dTEC_Butter=butter_bandpass_filter(dTEC[idx],15,30,240,order=5)
#     x_axis=T_input[idx]
    
    
    
    
#     idxa=abs(dTEC_Butter)>0.025
    
#     elev = elev[idx]
#     elev = elev[idxa]
    
#     idxe = elev>20
    
#     # print(len(idxe)==len(idxa))
    
#     # ax.scatter(x_axis[idxa],y_axis[idxa],marker='.',s=1,c='r')



#     x_axis = lon
#     y_axis = lat
#     ax.scatter(x_axis,y_axis, marker ='.',s=1,c='k',alpha=.1)
#     ax.scatter(x_axis[idxa],y_axis[idxa],marker='x',s=1,c='r')

# x = sta_rtp['lon_deg']
# y = sta_rtp['lat_deg']

# ax.scatter(x, y,color='b',s=200)
# plt.savefig('POAL15_259_displacement_{}.png'.format(min_wanted), dpi=300)



    # prn_data=Df_RINEX[Df_RINEX['prn']==sta_prns[i]]
    # # print(i)
    # #-----1st plot
    # min_wanted=7
    # x_obs = prn_data['horario_s'].values
    # y_obs = prn_data['TEC'].values
    # y_fit = prn_data['TEC'].rolling(2*min_wanted*4,center=True).mean().values
    
    # x_axis=x_obs
    # y_axis=y_obs

    # dy_a=y_obs-y_fit
    # x_axis=x_obs
    # # y_axis=dy_a
    
    # # y_axis=y_obs
    # # axs[i].plot(x_axis,y_axis,label=sta_prns[i])
    # # y_axis=y_fit
    # # axs[i].plot(x_axis,y_axis,label=sta_prns[i])
    
    # # y_axis=dy_a
    # # axs[i].plot(x_axis,y_axis,label=sta_prns[i])
    
    # idx=np.isfinite(dy_a)
    # y_axis=butter_bandpass_filter(dy_a[idx],15,30,240,order=5)
    # idxa=abs(y_axis)>0.025
    # x_axis=x_axis[idx]
    # axs[i].plot(x_axis[idxa],y_axis[idxa],'-o',label=sta_prns[i])


























# def interpolate_data(prn_data):
#     #-Changing the PRN name for the iteration
#     df_prn=prn_data

#     x = df_prn['time'].dt.hour.values + (df_prn['time'].dt.minute.values + df_prn['time'].dt.second.values/60)/60
#     y = ct*(df_prn[ini]-df_prn[fin])
#     y = y.values

#     #-Excluding NaN Values
#     idx = np.isfinite(x) & np.isfinite(y)

#     #-Extracting the polinomials coefficients
#     poly = np.polyfit(x[idx], y[idx], 6)


#     #-Calculating TEC
#     y_obs = ct*(df_prn[ini]-df_prn[fin])

#     #-Calculating the fit curve.
#     y_obs = y_obs.values
#     y_fit = np.polyval(poly, x)

#     #-taking the difference of observable and the fit
#     y=y_obs-y_fit

#     #-interpolating the values (to make the fourrier transform)
#     y_interpolated = np.interp(np.arange(len(y)), 
#                      np.arange(len(y))[np.isnan(y) == False], 
#                      y[np.isnan(y) == False])
#     return x, y_interpolated



# def fourrier_extract(prn_data,ini,fin):
#     x,y_interpolated = interpolate_data(prn_data)
    
# #     #-Changing the PRN name for the iteration
# #     df_prn=prn_data

# #     x = df_prn['time'].dt.hour.values + (df_prn['time'].dt.minute.values + df_prn['time'].dt.second.values/60)/60
# #     y = ct*(df_prn[ini]-df_prn[fin])
# #     y = y.values

# #     #-Excluding NaN Values
# #     idx = np.isfinite(x) & np.isfinite(y)

# #     #-Extracting the polinomials coefficients
# #     poly = np.polyfit(x[idx], y[idx], 6)


# #     #-Calculating TEC
# #     y_obs = ct*(df_prn[ini]-df_prn[fin])

# #     #-Calculating the fit curve.
# #     y_obs = y_obs.values
# #     y_fit = np.polyval(poly, x)

# #     #-taking the difference of observable and the fit
# #     y=y_obs-y_fit

# #     #-interpolating the values (to make the fourrier transform)
# #     y_interpolated = np.interp(np.arange(len(y)), 
# #                      np.arange(len(y))[np.isnan(y) == False], 
# #                      y[np.isnan(y) == False])

#     #Fazendo a transformada de Fourier dos dados acima.
#     dt=1887/9
# #     t=x
# #     f=y_interpolated

#     n=len(x)
#     fhat=np.fft.fft(y_interpolated,n) # O metodo usado para se fazer a transformada de Fourier é o FFT
#     PSD= fhat * np.conj(fhat)/n # Tomando os valores ao quadrado, esse passo é importante porque os valores podem ser numeros complexos. (Power Spectrum)
#     PSD=PSD.real
#     # freq = (1/(dt*n))*np.arange(n) #criando o eixo-x de frequencias
#     L = np.arange(1,np.floor(n/2),dtype='int') #Plotando apenas a primeira metade da transformada

#     timestep = 15/3600
#     freq = np.fft.fftfreq(n, d=timestep)

#     #Aplicando um filtro nesse espectro de forma que apenas as frequencias acima de um limiar continuem
#     #Entao sera apagado manualmente todas as frequencias que tem um espectro indesejado
#     # indices = PSD > 0.05e10            #Essa parte vai dar trabalho
#     indices = (freq >= 15.12) & (freq <= 29.88)
#     #------------------------------------------------------------
#     PSD_clean = PSD * indices       # A matriz de indices é zero para todos os logares que nao foram verdadeiros na condição acima
#     # fhat = indices * fhat           #fhat e a transformada de fourier, que agora tera todos os valores zerados, com excecao dos valores que foram verdadeiros cna condicao acima.
#     # ffilt = np.fft.ifft(fhat)       #Inverse FFT para os sinais filtrados
#     # fhat
#     return PSD, PSD_clean, freq, L, x, y_interpolated


# # In[59]:


# #This is just a copy
# def interpolate_data(prn_data):
#     #-Changing the PRN name for the iteration
#     df_prn=prn_data

#     x = df_prn['time'].dt.hour.values + (df_prn['time'].dt.minute.values + df_prn['time'].dt.second.values/60)/60
#     y = ct*(df_prn[ini]-df_prn[fin])
#     y = y.values

#     #-Excluding NaN Values
#     idx = np.isfinite(x) & np.isfinite(y)

#     #-Extracting the polinomials coefficients
#     poly = np.polyfit(x[idx], y[idx], 6)


#     #-Calculating TEC
#     y_obs = ct*(df_prn[ini]-df_prn[fin])

#     #-Calculating the fit curve.
#     y_obs = y_obs.values
#     y_fit = np.polyval(poly, x)

#     #-taking the difference of observable and the fit
#     y=y_obs-y_fit

#     #-interpolating the values (to make the fourrier transform)
#     y_interpolated = np.interp(np.arange(len(y)), 
#                      np.arange(len(y))[np.isnan(y) == False], 
#                      y[np.isnan(y) == False])
#     return x, y_interpolated



# def fourrier_extract(prn_data,ini,fin):
#     x,y_interpolated = interpolate_data(prn_data)
    
# #     #-Changing the PRN name for the iteration
# #     df_prn=prn_data

# #     x = df_prn['time'].dt.hour.values + (df_prn['time'].dt.minute.values + df_prn['time'].dt.second.values/60)/60
# #     y = ct*(df_prn[ini]-df_prn[fin])
# #     y = y.values

# #     #-Excluding NaN Values
# #     idx = np.isfinite(x) & np.isfinite(y)

# #     #-Extracting the polinomials coefficients
# #     poly = np.polyfit(x[idx], y[idx], 6)


# #     #-Calculating TEC
# #     y_obs = ct*(df_prn[ini]-df_prn[fin])

# #     #-Calculating the fit curve.
# #     y_obs = y_obs.values
# #     y_fit = np.polyval(poly, x)

# #     #-taking the difference of observable and the fit
# #     y=y_obs-y_fit

# #     #-interpolating the values (to make the fourrier transform)
# #     y_interpolated = np.interp(np.arange(len(y)), 
# #                      np.arange(len(y))[np.isnan(y) == False], 
# #                      y[np.isnan(y) == False])

#     #Fazendo a transformada de Fourier dos dados acima.
#     dt=1887/9
# #     t=x
# #     f=y_interpolated

#     n=len(x)
#     fhat=np.fft.fft(y_interpolated,n) # O metodo usado para se fazer a transformada de Fourier é o FFT
#     PSD= fhat * np.conj(fhat)/n # Tomando os valores ao quadrado, esse passo é importante porque os valores podem ser numeros complexos. (Power Spectrum)
#     PSD=PSD.real
#     # freq = (1/(dt*n))*np.arange(n) #criando o eixo-x de frequencias
#     L = np.arange(1,np.floor(n/2),dtype='int') #Plotando apenas a primeira metade da transformada

#     timestep = 15/3600
#     freq = np.fft.fftfreq(n, d=timestep)

#     #Aplicando um filtro nesse espectro de forma que apenas as frequencias acima de um limiar continuem
#     #Entao sera apagado manualmente todas as frequencias que tem um espectro indesejado
#     # indices = PSD > 0.05e10            #Essa parte vai dar trabalho
#     indices = (freq >= 15.12) & (freq <= 29.88)
#     #------------------------------------------------------------
#     PSD_clean = PSD * indices       # A matriz de indices é zero para todos os logares que nao foram verdadeiros na condição acima
#     # fhat = indices * fhat           #fhat e a transformada de fourier, que agora tera todos os valores zerados, com excecao dos valores que foram verdadeiros cna condicao acima.
#     # ffilt = np.fft.ifft(fhat)       #Inverse FFT para os sinais filtrados
#     # fhat
#     return PSD, PSD_clean, freq, L, x, y_interpolated


# # In[54]:


# # def extract_data_from_prn(prn_data):
# #     df_prn=prn_data

# #     x = df_prn['time'].dt.hour.values + (df_prn['time'].dt.minute.values + df_prn['time'].dt.second.values/60)/60
# #     y = ct*(df_prn[ini]-df_prn[fin])
# #     y = y.values
# #     return x, y


# # def interpolate_data(prn_data):
# #     #-Changing the PRN name for the iteration
# #     prn_data=prn_data
# #     x,y = extract_data_from_prn(prn_data)

# #     #-Excluding NaN Values
# #     idx = np.isfinite(x) & np.isfinite(y)

# #     #-Extracting the polinomials coefficients
# #     poly = np.polyfit(x[idx], y[idx], 6)


# #     #-Calculating TEC
# #     y_obs = ct*(df_prn[ini]-df_prn[fin])

# #     #-Calculating the fit curve.
# #     y_obs = y_obs.values
# #     y_fit = np.polyval(poly, x)

# #     #-taking the difference of observable and the fit
# #     y=y_obs-y_fit

# #     #-interpolating the values (to make the fourrier transform)
# #     y_interpolated = np.interp(np.arange(len(y)), 
# #                      np.arange(len(y))[np.isnan(y) == False], 
# #                      y[np.isnan(y) == False])
# #     return x, y_interpolated



# # def fourrier_extract(prn_data,ini,fin):
# #     x,y_interpolated = interpolate_data(prn_data)
    
# #     dt=1887/9
# # #     t=x
# # #     f=y_interpolated

# #     n=len(x)
# #     fhat=np.fft.fft(y_interpolated,n) # O metodo usado para se fazer a transformada de Fourier é o FFT
# #     PSD= fhat * np.conj(fhat)/n # Tomando os valores ao quadrado, esse passo é importante porque os valores podem ser numeros complexos. (Power Spectrum)
# #     PSD=PSD.real
# #     # freq = (1/(dt*n))*np.arange(n) #criando o eixo-x de frequencias
# #     L = np.arange(1,np.floor(n/2),dtype='int') #Plotando apenas a primeira metade da transformada

# #     timestep = 15/3600
# #     freq = np.fft.fftfreq(n, d=timestep)

# #     #Aplicando um filtro nesse espectro de forma que apenas as frequencias acima de um limiar continuem
# #     #Entao sera apagado manualmente todas as frequencias que tem um espectro indesejado
# #     # indices = PSD > 0.05e10            #Essa parte vai dar trabalho
# #     indices = (freq >= 15.12) & (freq <= 29.88)
# #     #------------------------------------------------------------
# #     PSD_clean = PSD * indices       # A matriz de indices é zero para todos os logares que nao foram verdadeiros na condição acima
# #     # fhat = indices * fhat           #fhat e a transformada de fourier, que agora tera todos os valores zerados, com excecao dos valores que foram verdadeiros cna condicao acima.
# #     # ffilt = np.fft.ifft(fhat)       #Inverse FFT para os sinais filtrados
# #     # fhat
# #     return PSD, PSD_clean, freq, L, x, y_interpolated


# # In[60]:


# def extract_data_from_prn(prn_data):
#     df_prn=prn_data

#     x = df_prn['time'].dt.hour.values + (df_prn['time'].dt.minute.values + df_prn['time'].dt.second.values/60)/60
#     y = ct*(df_prn[ini]-df_prn[fin])
#     y = y.values
#     return x, y


# # In[61]:


# x,y = extract_data_from_prn(prn_data)

# x


# # In[62]:


# arr=[]
# prn_PSD=[]


# # In[63]:


# ini='L1'
# fin='L2'
# arr=[]

# n_prn=0

# for i in range(len(prn_name)):
#     prn_data=df[df['prn']==prn_name[i]]
#     data_size=len(prn_data['time'])
#     if data_size >= 256:
#         n_prn+=1
# #         print(prn_name[i])
#         prn_PSD.append(prn_name[i])
#         PDS, PSD_filtered, freq, L, x, y_interpolated = fourrier_extract(prn_data,ini,fin)
#         arr.append(PSD_filtered)


# # In[64]:


# n_prn


# # In[65]:


# df_PDS=pd.DataFrame(arr)


# # In[66]:


# df_PDS = df_PDS.assign(prn=prn_PSD)

# df_PDS=df_PDS.set_index('prn')


# # In[67]:


# #Periodo
# n_axis=0
# fig,axs = plt.subplots(n_prn,1,figsize=(20,60))
# #-------------------------------
# for i in range(len(prn_name)):
#     prn_data=df[df['prn']==prn_name[i]]
#     data_size=len(prn_data['time'])
#     if data_size >= 256:
#         #Fazendo os varios plots
# #         print(len(prn_data['time']))
#         plt.sca(axs[n_axis])

#         PDS, PSD_filtered, freq, L, x, y_interpolated = fourrier_extract(prn_data,ini,fin)
#         x_axis=60/freq[L]
#         y_axis=PSD_filtered[L]

#         plt.plot(x_axis,y_axis,'.-',color='m',linewidth=1,label=prn_name[i])
#         plt.xlim(0,15)
#         plt.legend()
#         plt.grid()
        
#         n_axis += 1
        
# # ax[n_prn].set_xlabel('x')

# # ax[n_prn].set(xlabel='x-axis', ylabel='y-axis',
# #        title='Title of plot')

# axs[n_prn-1].set_xlabel('Periodo (1/h)')
# # ax.set_ylabel('common ylabel')
# fig.tight_layout()

# plt.savefig('POAL16_Periodo.pdf', dpi=600)
# plt.show()


# # In[68]:


# #Frequencia
# n_axis=0
# fig,axs = plt.subplots(n_prn,1,figsize=(20,60))

# #-------------------------------
# for i in range(len(prn_name)):
#     prn_data=df[df['prn']==prn_name[i]]
#     data_size=len(prn_data['time'])
#     if data_size >= 256:
#         #Fazendo os varios plots
# #         print(len(prn_data['time']))
#         plt.sca(axs[n_axis])

#         PDS, PSD_filtered, freq, L, x, y_interpolated = fourrier_extract(prn_data,ini,fin)
#         x_axis=freq[L]/3.6
#         y_axis=PSD_filtered[L]

#         plt.plot(x_axis,y_axis,'.-',color='b',linewidth=1,label=prn_name[i])
# #         plt.xlim(0,9)
#         plt.legend()
#         plt.grid()
        
#         n_axis += 1
        
# # ax[n_prn].set_xlabel('x')

# # ax[n_prn].set(xlabel='x-axis', ylabel='y-axis',
# #        title='Title of plot')

# axs[n_prn-1].set_xlabel('Frequencia (mHz)')
# # ax.set_ylabel('common ylabel')
# fig.tight_layout()

# plt.savefig('POAL16_PSD.pdf', dpi=600)
# plt.show()


# # In[ ]:





# # In[70]:


# n_axis=0
# fig,axs = plt.subplots(n_prn,1,figsize=(20,60))

# #-------------------------------

# freq,PSD,L,f_hat=fft_data(x, y_interpolated_hamming)
# # PDS, PSD_filtered, freq, L, xx, yy = fourrier_extract(prn_data,ini,fin)
# #Aplicando um filtro nesse espectro de forma que apenas as frequencias acima de um limiar ocntinuem
# #Entao sera apagado manualmente todas as frequencias que tem um espectro menor que 100
# # indices = PSD > 0.05e10            #Essa parte vai dar trabalho
# # indices = (freq >= 15.12) & (freq <= 29.88)
# indices = (abs(freq) >= 0.) & (abs(freq) < 12.15)
# #------------------------------------------------------------
# PSD_clean = PSD * indices       # A matriz de indices é zero para todos os logares que nao foram verdadeiros na condição acima
# f_hat = indices * f_hat           #fhat e a transformada de fourier, que agora tera todos os valores zerados, com excecao dos valores que foram verdadeiros cna condicao acima.
# ffilt = np.fft.ifft(f_hat)       #Inverse FFT para os sinais filtrados
# ffilt = ffilt.real
# # fhat


# for i in range(len(prn_name)):
#     prn_data=df[df['prn']==prn_name[i]]
#     data_size=len(prn_data['time'])
#     if data_size >= 256:
#         #Fazendo os varios plots
# #         print(len(prn_data['time']))
#         plt.sca(axs[n_axis])

#         PDS, PSD_filtered, freq, L, xx, yy = fourrier_extract(prn_data,ini,fin)
#         t=xx
#         f=yy

#         plt.plot(t,f,'.-',color='k',linewidth=1,label=prn_name[i])
#         plt.xlim(15.8,20.2)
#         plt.legend()
#         plt.grid()
        
#         n_axis += 1
        
# # ax[n_prn].set_xlabel('x')

# # ax[n_prn].set(xlabel='x-axis', ylabel='y-axis',
# #        title='Title of plot')

# axs[n_prn-1].set_xlabel('Horas (h)')
# # ax.set_ylabel('common ylabel')
# fig.tight_layout()

# plt.savefig('POAL16_dTEC.pdf', dpi=300)
# plt.show()


# # In[71]:


# n_axis=0
# fig,axs = plt.subplots(n_prn,1,figsize=(20,60))

# #-------------------------------
# for i in range(len(prn_name)):
#     prn_data=df[df['prn']==prn_name[i]]
#     data_size=len(prn_data['time'])
#     if data_size >= 256:
#         #Fazendo os varios plots
# #         print(len(prn_data['time']))
#         plt.sca(axs[n_axis])

#         PDS, PSD_filtered, freq, L, xx, yy = fourrier_extract(prn_data,ini,fin)
#         t=xx
#         f=np.hamming(len(yy))*yy

#         plt.plot(t,f,'.-',color='c',linewidth=1,label=prn_name[i])
#         plt.xlim(15.8,20.2)
#         plt.legend()
#         plt.grid()
        
#         n_axis += 1
        
# # ax[n_prn].set_xlabel('x')

# # ax[n_prn].set(xlabel='x-axis', ylabel='y-axis',
# #        title='Title of plot')

# axs[n_prn-1].set_xlabel('Horas (h)')
# # ax.set_ylabel('common ylabel')
# fig.tight_layout()

# plt.savefig('POAL16_dTEC.pdf', dpi=300)
# plt.show()


# # In[ ]:


# # y_interpolated_hamming = np.hamming(len(y_interpolated))*y_interpolated


# # In[72]:


from scipy.signal import butter, lfilter, filtfilt

def butter_bandpass(lowcut, highcut, frequency_sample, order=5):
    nyq = 0.5 * frequency_sample
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, frequency_sample, order=5):
    b, a = butter_bandpass(lowcut, highcut, frequency_sample, order=order)
    y = lfilter(b, a, data)
#     y = filtfilt(b, a, data)
    return y


# # In[ ]:





# # In[ ]:


# sns.heatmap(df_PDS,cmap='crest')


# # In[ ]:





# # In[ ]:





# # In[ ]:





# # In[ ]:





# # In[ ]:





# # In[ ]:





# # In[ ]:





# # In[919]:


# # fig, ax = plt.subplots(figsize=(20,5))

# # #------------------------
# # # x = df_R09['time'].dt.hour.values + df_R09['time'].dt.minute.values/60
# # y_obs = df_prn[ini]-df_prn[fin]

# # y_obs = y_obs.values
# # y_fit = np.polyval(poly, x)

# # y=y_obs-y_fit

# # ax.scatter(x, y_obs,c='black',label=prn_name[i])
# # ax.plot(x, y_fit,color='red',label='trend_fit')

# # # ax.plot(x, y,label='fit')

# # # ax.plot(x,np.polyval(poly, x), label='fit')
# # #------------------------
# # # ax.yaxis.set_major_locator(plt.MaxNLocator(10))
# # ax.set(xlabel='x-axis', ylabel='y-axis',
# #        title='Title of plot')
# # ax.grid()

# # plt.legend(loc='best')
# # # plt.xticks(rotation=30)
# # plt.show()


# # In[920]:


# y_interpolated = np.interp(np.arange(len(y)), 
#           np.arange(len(y))[np.isnan(y) == False], 
#           y[np.isnan(y) == False])


# # In[921]:


# fig, ax = plt.subplots(figsize=(20,5))

# #------------------------
# # x = df_R09['time'].dt.hour.values + df_R09['time'].dt.minute.values/60
# # y_obs = df_prn[ini]-df_prn[fin]

# # y_obs = y_obs.values
# # y_fit = np.polyval(poly, x)

# # y=y_obs-y_fit


# ax.plot(x, y,'.-',color='k',label='dTEC')
# ax.plot(x, y_interpolated,color='r',label='interpolated')

# # ax.plot(x,np.polyval(poly, x), label='fit')
# #------------------------
# # ax.yaxis.set_major_locator(plt.MaxNLocator(10))
# ax.set(xlabel='x-axis', ylabel='y-axis',
#        title='Title of plot')
# ax.grid()

# # plt.xlim([0.5,1.5])
# plt.legend(loc='best')
# # plt.xticks(rotation=30)
# plt.show()


# # In[922]:


# #Fazendo a transformada de Fourier dos dados acima.
# dt=1887/9
# t=x
# f=y_interpolated

# n=len(t)
# fhat=np.fft.fft(f,n) # O metodo usado para se fazer a transformada de Fourier é o FFT
# PSD= fhat * np.conj(fhat)/n # Tomando os valores ao quadrado, esse passo é importante porque os valores podem ser numeros complexos. (Power Spectrum)
# PSD=PSD.real
# # freq = (1/(dt*n))*np.arange(n) #criando o eixo-x de frequencias
# L = np.arange(1,np.floor(n/2),dtype='int') #Plotando apenas a primeira metade da transformada

# timestep = 15/3600
# freq = np.fft.fftfreq(n, d=timestep)
# # freq


# fig,axs = plt.subplots(3,1,figsize=(20,15))

# plt.sca(axs[0])
# plt.plot(t,f,'.-',color='k',linewidth=1,label='Signal')
# #plt.plot(t,f_clean,color='r',linewidth=1,label='Clean') 
# plt.xlim(t[0],t[-1])
# plt.legend()
# plt.grid()

# plt.sca(axs[1])
# plt.stem(freq[L],PSD[L],basefmt='C6-', linefmt='blue', markerfmt='D', bottom=10.1,label='PSD (1/h)')
# # plt.xlim(0,30)
# plt.legend()
# plt.grid()

# plt.sca(axs[2])
# plt.stem(freq[L]/3.6,PSD[L],basefmt='C6-', linefmt='red', markerfmt='D', bottom=10.1,label='PSD (mHz)')
# # plt.xlim(0,30)
# plt.legend()
# plt.grid()

# fig.tight_layout()
# plt.show()


# # In[924]:


# # (freq >= 15.12) & (freq <= 29.88)


# # In[925]:


# # fhat


# # In[926]:


# #Aplicando um filtro nesse espectro de forma que apenas as frequencias acima de um limiar ocntinuem
# #Entao sera apagado manualmente todas as frequencias que tem um espectro menor que 100
# # indices = PSD > 0.05e10            #Essa parte vai dar trabalho
# indices = (freq >= 15.12) & (freq <= 29.88)
# #------------------------------------------------------------
# PSD_clean = PSD * indices       # A matriz de indices é zero para todos os logares que nao foram verdadeiros na condição acima
# fhat = indices * fhat           #fhat e a transformada de fourier, que agora tera todos os valores zerados, com excecao dos valores que foram verdadeiros cna condicao acima.
# ffilt = np.fft.ifft(fhat)       #Inverse FFT para os sinais filtrados
# # fhat


# # In[927]:


# fig,axs = plt.subplots(3,1,figsize=(20,20))

# plt.sca(axs[0])
# plt.plot(t,f,'.-',color='k',linewidth=1,label='Noisy')
# # plt.plot(t,f_clean,color='r',linewidth=1,label='Clean') 
# plt.plot(t,ffilt.real,color='r',linewidth=1,label='Clean') 
# # plt.xlim(t[0],t[-1])
# plt.legend()
# plt.grid()

# plt.sca(axs[1])
# plt.plot(t,ffilt.real,'.-',color='r',linewidth=1,label='Filtered')
# # plt.xlim(t[0],t[-1])
# plt.legend()
# plt.grid()

# plt.sca(axs[2])
# plt.plot(freq[L]/3.6,PSD[L],color='k',linewidth=1,label='PSD_FULL')
# plt.plot(freq[L]/3.6,PSD_clean[L],'.-',color='r',linewidth=1,label='PSD_CLEAN')
# # plt.xlim(0.001,0.0015)

# # Create a Rectangle patch
# # rect = patches.Rectangle((50, 100), 40, 30, linewidth=1, edgecolor='r', facecolor='none')

# # Add the patch to the Axes
# # plt.add_patch(rect)



# plt.legend()
# plt.grid()

# fig.tight_layout()
# plt.show()


# # In[ ]:


# ###################################################################################3


# # In[928]:


# plt.rcParams['figure.figsize'] = [20,5]
# plt.rcParams.update({'font.size':18})

# Spectrum, freq, time, iAxis = plt.specgram(y_interpolated,
#                                            NFFT=256,
#                                            Fs=len(y_interpolated)/t_max, #esse numero nao esta correto
# #                                            Fs=240/3.6, #esse numero nao esta correto
#                                            mode = 'magnitude',
#                                            noverlap=128,
#                                            cmap='gnuplot')
# plt.grid()
# plt.xlabel('Time (h)')
# plt.colorbar()

# plt.ylabel('Frequency (1/h)')
# # plt.colorbar()

# plt.show()


# # In[861]:


# time


# # In[797]:


# len(y_interpolated)
