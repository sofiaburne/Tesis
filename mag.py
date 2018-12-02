# 0 uso modulo desde otro modulo
# 1 uso modulo y quiero que me haga plots y los guarde
MODO_mag = 0  

import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import os
#%%

#Datos MAG

path_mag = r'C:\Users\sofia\Documents\Facultad\Tesis\Datos Maven\MAG/'
d, h, m, day_frac, Bx, By, Bz, X, Y, Z = np.loadtxt(path_mag+'alta_frec/2014/12/mvn_mag_l2_2014359pc_20141225_v01_r01.sts', skiprows = 147, usecols = (1,2,3,6,7,8,9,11,12,13),unpack = True)

#paso a unidades de radios marciano
x, y, z = X/3390, Y/3390, Z/3390

#%%

#vector de tiempos en hora decimal
t_mag = (day_frac-d[5])*24

#fecha del shock
shock_date = dt.date(2014,12,25) #* cambiar a mano

#modulo de B en coordenadas MSO
B = np.sqrt(Bx**2 + By**2 + Bz**2)

#%%
if MODO_mag == 1:
    
    '''
    plot de intervalo entre mediciones para checkear si hay agujeros donde la nave no haya medido
    (Si algun intervalo es >1s entonces ahi la nave no midio)
    '''
    
    Dt = np.empty([len(t_mag)-1])     
    for i in range(len(t_mag)):
        Dt[i-1] = t_mag[i]-t_mag[i-1]
    Dt = Dt*3600 #expreso los intervalos en segundos
    
    #plt.figure(0, figsize = (30,20), tight_layout = True)
    #plt.title('Intervalos temporales entre mediciones MAG', fontsize = 30)
    #plt.plot(t_mag[:-1], Dt, 'o-', linewidth = 2)
    #plt.ylabel('Intervalo entre mediciones [s]', fontsize = 20)
    #plt.xlabel('Tiempo [hora decimal]', fontsize = 20)
    #plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
    #plt.grid(axis = 'both', which = 'both', alpha = 0.8, linewidth = 2, linestyle = '--')
    #plt.show()
    
    #carpeta para guardar plot
    #ojo: para dia con mas de un shock hacer subcarpetas a mano
    path_analisis = r'C:\Users\sofia\Documents\Facultad\Tesis\Analisis\{}/'.format(shock_date)
    if not os.path.exists(path_analisis):
        os.makedirs(path_analisis)
    
    #plt.savefig(path_analisis+'intervalos_mediciones_MAG_{}'.format(shock_date))
        
    
    #lo veo en forma de histograma para mostrar que la mayoria de las mediciones respetan la frecuencia de muestreo
    
    plt.figure(1, figsize = (30,20), tight_layout = True)
    plt.title('Intervalos temporales entre mediciones MAG', fontsize = 30)   
    plt.hist(Dt, 30)
    plt.ylabel('Cantidad de eventos', fontsize = 20)
    plt.xlabel('Intervalo entre mediciones [s]', fontsize = 20)
    plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
    plt.grid(axis = 'both', which = 'both', alpha = 0.8, linewidth = 2, linestyle = '--')
    plt.show()
    
    plt.savefig(path_analisis+'hist_intervalos_mediciones_MAG_{}'.format(shock_date))
    
    #zoom del hsitograma
    
    plt.figure(2, figsize = (30,20), tight_layout = True)
    plt.title('Intervalos temporales entre mediciones MAG - ampliación del histograma', fontsize = 30)   
    plt.hist(Dt, 30)
    plt.ylabel('Cantidad de eventos', fontsize = 20)
    plt.ylim(ymin = 0, ymax = 100)
    plt.xlabel('Intervalo entre mediciones [s]', fontsize = 20)
    plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
    plt.grid(axis = 'both', which = 'both', alpha = 0.8, linewidth = 2, linestyle = '--')
    plt.show()
    
    plt.savefig(path_analisis+'hist_zoom_intervalos_mediciones_MAG_{}'.format(shock_date))







