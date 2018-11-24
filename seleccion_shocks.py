# script para seleccionar shocks rapido

import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import cdflib
#%%
#para pasar de time unix a tiempo decimal o entero
def tfromtimeunix(timeunix):
    
    '''
    time unix es cant de segundos desde 1 enero 1970 00:00:00
    la resolucion del vector time_unix es de 4 s
    
    toordinal devuelve la cantidad de dias contando desde el 1/1/1 agregando un dia cada año bisiesto y estando 1 día cada siglo (excepto que el siglo sea multiplo de 400)
    como time unix no agrega segundo bisiestos, tengo que sacarme de encima esas correcciones
    
    
    minicodigo para chequear que estoy corrigiendo bien:

    year = 1969
    D = (dt.date(year,12,31)).toordinal() - int(year/4) + int(year/100) - int(year/400)  
    diference = D - 365*year
    print(diference)
    
    '''
    
    d0_tu = 1 + ( (dt.date(1969,12,31)).toordinal() - int(1969/4) + int(1960/100) - int(1960/400) ) #cant de dias desde 1/1/1 al 1970/1/1
    dec_day = d0_tu + timeunix/86400 #dia decimal desde 1/1/1 al momento de cada medicion (en un dia hay 86400)
        
    # days, hours, minutes, seconds = cant de dias, horas, minutos y segundos desde 1970/1/1 hasta el momento de cada medicion
    # dec_day, dec_hour, dec_min, dec_sec = idem pero fracciones decimales
    days = np.asarray([int(i) for i in dec_day])
    dec_hour = (dec_day - days)*86400/3600
    hours = np.asarray([int(i) for i in dec_hour])
    dec_min = (dec_hour - hours)*3600/60
    minutes = np.asarray([int(i) for i in dec_min])
    dec_sec = (dec_min - minutes)*60
    seconds = np.asarray([int(i) for i in dec_sec]) #me quedo con resolucion en seg
    
    
    #chequeo que sea el dia del shock
    ds = days[0] + int(1969/4) - int(1960/100) + int(1960/400)
    shock_day = dt.datetime.fromordinal(ds)
    print(shock_day)
    
    #por ahora voy a trabajar en hora decimal
    return dec_hour

#%%

#fecha del shock
shock_date = dt.date(2017,4,6) #* cambiar a mano

#Datos MAG
path_mag = r'C:\Users\sofia\Documents\Facultad\Tesis\Datos Maven\MAG/'
d, day_frac, Bx, By, Bz = np.loadtxt(path_mag+'2017/04/mvn_mag_l2_2017096ss1s_20170406_v01_r01.sts', skiprows = 155, usecols = (1,6,7,8,9),unpack = True)

#Datos momentos SWIA
path_swia_mom = r'C:\Users\sofia\Documents\Facultad\Tesis\Datos Maven\SWIA/momentos/'
data_swia_mom = cdflib.CDF(path_swia_mom+'2017/04/mvn_swi_l2_onboardsvymom_20170406_v01_r01.cdf')
data_swia_mom.cdf_info()


t_mag = (day_frac-d[5])*24
B = np.sqrt(Bx**2 + By**2 + Bz**2)

timeunix_swia_mom = data_swia_mom.varget('time_unix')
density_swia = data_swia_mom.varget('density')
velocity_swia = data_swia_mom.varget('velocity_mso')
#los guardo en otros array que se mantengan cuando cierre el archivo
tu_swia_mom = np.asarray(timeunix_swia_mom)
densidad_swia = np.asarray(density_swia)
velocidad_swia = np.asarray(velocity_swia)
data_swia_mom.close()
t_swia_mom = tfromtimeunix(tu_swia_mom[:-15]) #hora decimal (saco ultimas mediciones porque son del dia sig)
velocidad_swia_norm = np.sqrt(velocidad_swia[:-15,0]**2 + velocidad_swia[:-15,1]**2 + velocidad_swia[:-15,2]**2)



f1, (g1,g2,g3) = plt.subplots(3,1, sharex = True, figsize = (30,20)) #ojo con sharex y los distintos inst

f1.suptitle('Datos MAVEN {}'.format(shock_date), fontsize = 20)

g1.plot(t_mag, B, linewidth = 2, color = 'C0')
g1.set_ylabel('Módulo de campo magnético\n[nT]', fontsize = 20)
g1.axes.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
g1.axes.grid(axis = 'both', which = 'both', alpha = 0.8, linewidth = 2, linestyle = '--')

g2.plot(t_swia_mom, densidad_swia[:-15], linewidth = 2, color = 'C1')
g2.set_ylabel('$n_p$\n[$cm^{-3}$]', fontsize = 20)
g2.axes.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
g2.axes.grid(axis = 'both', which = 'both', alpha = 0.8, linewidth = 2, linestyle = '--')

g3.plot(t_swia_mom, velocidad_swia_norm, linewidth = 2, color = 'C2')
g3.set_xlabel('Tiempo\n[hora decimal]', fontsize = 20)
g3.set_ylabel('Módulo de velocidad MSO\n[km/s]', fontsize = 20)
g3.axes.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
g3.axes.grid(axis = 'both', which = 'both', alpha = 0.8, linewidth = 2, linestyle = '--')

