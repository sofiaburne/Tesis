# 0 uso modulo desde otro modulo
# 1 uso modulo y quiero que me haga plots y los guarde
MODO_swiaswea = 0

'''
This is a python script to read CDF files without needing to install the
CDF NASA library. You will need Python version 3, as well as the Numpy
library to use this module.
To install, open up your terminal/command prompt, and type::
    pip install cdflib
##########
CDF Class
##########
To begin accessing the data within a CDF file, first create a new CDF class.
This can be done with the following commands::
    import cdflib
    cdf_file = cdflib.CDF('/path/to/cdf_file.cdf')
Then, you can call various functions on the variable.  For example::
    x = cdf_file.varget("NameOfVariable", startrec = 0, endrec = 150)
This command will return all data inside of the variable "Variable1", from
records 0 to 150.
Sample use::
    import cdflib
    swea_cdf_file = cdflib.CDF('/path/to/swea_file.cdf')
    swea_cdf_file.cdf_info()
    x = swea_cdf_file.varget('NameOfVariable')
    swea_cdf_file.close()
    cdflib.cdfread.CDF.getVersion()
@author: Bryan Harter, Michael Liu
'''
#%%
import cdflib
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from mag import shock_date

path_analisis = r'C:\Users\sofia\Documents\Facultad\Tesis\Analisis\{}/'.format(shock_date)

#%%

#Funciones

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




'''
Los espectros de energia del swia y swea no se miden para niveles de
energia equiespaciados. Esta funcion es para quedarme con los niveles de energia
y correspondientes espectros entre un rango dado por[E_max eV, E_min eV]
donde se mide una cant de iones/electrones apreciable como para ver el shock, y
tomando niveles separados por un salto ~dE eV.
'''

def rango_energias_shock(E_min, E_max, dE, niveles, flujo):
    
    Emin = (np.abs(niveles-E_min)).argmin()
    Emax = (np.abs(niveles-E_max)).argmin()
    
    rango_indices = []
    rango_energias = []
    rango_flujos = []
    
    cursor = Emax
    while cursor <= Emin:
        rango_indices.append(cursor)
        rango_energias.append(niveles[cursor])
        rango_flujos.append(flujo[:,cursor])
        cursor = (np.abs(niveles - (niveles[cursor]-dE))).argmin()
    return np.asarray(rango_indices), np.asarray(rango_energias), np.asarray(rango_flujos).T

#%%#######################################################################################################
##########################################################################################################
##########################################################################################################
#%%

#Datos swea (cambiar con el shock de interes)
    
path_swea = r'C:\Users\sofia\Documents\Facultad\Tesis\Datos Maven\SWEA/'
data_swea = cdflib.CDF(path_swea+'2014/12/mvn_swe_l2_svyspec_20141225_v04_r01.cdf')
data_swea.cdf_info()

#%%

#variables swea

timeunix_swea = data_swea.varget('time_unix')
energyflux_swea = data_swea.varget('diff_en_fluxes')
energylevel_swea = data_swea.varget('energy')
counts_swea = data_swea.varget('counts')

#los guardo en otros array que se mantengan cuando cierre el archivo
tu_swea = np.asarray(timeunix_swea)
flujosenergia_swea = np.asarray(energyflux_swea)
nivelesenergia_swea = np.asarray(energylevel_swea)

data_swea.close()

t_swea = tfromtimeunix(tu_swea) #tomo t_swea como la hora decimal

#%%

if MODO_swiaswea == 1:
    
    '''
    plot de intervalo entre mediciones para checkear si hay agujeros donde la nave no haya medido
    (Si algun intervalo es >4s entonces ahi la nave no midio)
    '''
    
    Dt_swea = np.empty([len(t_swea)-1])     
    for i in range(len(t_swea)):
        Dt_swea[i-1] = t_swea[i]-t_swea[i-1]
    Dt_swea = Dt_swea*3600 #expreso los intervalos en segundos
    
    plt.figure(3, figsize = (30,20), tight_layout = True)
    plt.title('Intervalos temporales entre mediciones SWEA', fontsize = 30)
    plt.plot(t_swea[:-1], Dt_swea, 'o-', linewidth = 2)
    plt.ylabel('Intervalo entre mediciones [s]', fontsize = 20)
    plt.xlabel('Tiempo [hora decimal]', fontsize = 20)
    plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
    plt.grid(axis = 'both', which = 'both', alpha = 0.8, linewidth = 2, linestyle = '--')
    plt.show()
    
    plt.savefig(path_analisis+'intervalos_mediciones_SWEA_{}'.format(shock_date))
        
    
    #lo veo en forma de histograma para mostrar que la mayoria de las mediciones respetan la frecuencia de muestreo
    
    plt.figure(4, figsize = (30,20), tight_layout = True)
    plt.title('Intervalos temporales entre mediciones SWEA', fontsize = 30)   
    plt.hist(Dt_swea, 30)
    plt.ylabel('Cantidad de eventos', fontsize = 20)
    plt.xlabel('Intervalo entre mediciones [s]', fontsize = 20)
    plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
    plt.grid(axis = 'both', which = 'both', alpha = 0.8, linewidth = 2, linestyle = '--')
    plt.show()
    
    plt.savefig(path_analisis+'hist_intervalos_mediciones_SWEA_{}'.format(shock_date))
    
    #zoom del hsitograma
    
    plt.figure(5, figsize = (30,20), tight_layout = True)
    plt.title('Intervalos temporales entre mediciones SWEA - ampliación del histograma', fontsize = 30)   
    plt.hist(Dt_swea, 30)
    plt.ylabel('Cantidad de eventos', fontsize = 20)
    plt.ylim(ymin = 0, ymax = 100)
    plt.xlabel('Intervalo entre mediciones [s]', fontsize = 20)
    plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
    plt.grid(axis = 'both', which = 'both', alpha = 0.8, linewidth = 2, linestyle = '--')
    plt.show()
    
    plt.savefig(path_analisis+'hist_zoom_intervalos_mediciones_SWEA_{}'.format(shock_date))

#%%

#Datos swia (cambiar con el shock de interes)

#momentos
path_swia_mom = r'C:\Users\sofia\Documents\Facultad\Tesis\Datos Maven\SWIA/momentos/'
data_swia_mom = cdflib.CDF(path_swia_mom+'2014/12/mvn_swi_l2_onboardsvymom_20141225_v01_r00.cdf')
data_swia_mom.cdf_info()

#espectros
path_swia_spec =r'C:\Users\sofia\Documents\Facultad\Tesis\Datos Maven\SWIA/espectros/'
data_swia_spec = cdflib.CDF(path_swia_spec+'2014/12/mvn_swi_l2_onboardsvyspec_20141225_v01_r00.cdf')
data_swia_spec.cdf_info()



#%%

#variables swia

#momentos
timeunix_swia_mom = data_swia_mom.varget('time_unix')
density_swia = data_swia_mom.varget('density')
velocity_swia = data_swia_mom.varget('velocity_mso')
temperature_swia = data_swia_mom.varget('temperature_mso')

#espectros
timeunix_swia_spec = data_swia_spec.varget('time_unix')
spectra_diff_en_fluxes_swia = data_swia_spec.varget('spectra_diff_en_fluxes')
energy_spectra_swia = data_swia_spec.varget('energy_spectra')

#los guardo en otros array que se mantengan cuando cierre el archivo
tu_swia_mom = np.asarray(timeunix_swia_mom)
tu_swia_spec = np.asarray(timeunix_swia_spec)
densidad_swia = np.asarray(density_swia)
velocidad_swia = np.asarray(velocity_swia)
temperatura_swia = np.asarray(temperature_swia)
flujosenergia_swia = np.asarray(spectra_diff_en_fluxes_swia)
nivelesenergia_swia = np.asarray(energy_spectra_swia)

data_swia_mom.close()
data_swia_spec.close()

t_swia_mom = tfromtimeunix(tu_swia_mom[:-4]) #hora decimal (saco ultimas mediciones porque son del dia sig)
t_swia_spec = tfromtimeunix(tu_swia_spec[:-1]) #hora decimal

velocidad_swia_norm = np.sqrt(velocidad_swia[:-4,0]**2 + velocidad_swia[:-4,1]**2 + velocidad_swia[:-4,2]**2)
temperatura_swia_norm = np.sqrt(temperatura_swia[:-4,0]**2 + temperatura_swia[:-4,1]**2 + temperatura_swia[:-4,2]**2)

#%%

if MODO_swiaswea == 1:
    
    '''
    plot de intervalo entre mediciones para checkear si hay agujeros donde la nave no haya medido
    (Si algun intervalo es >4s entonces ahi la nave no midio)
    '''
    
    #para los momentos
    
    Dt_swia_mom = np.empty([len(t_swia_mom)-1])     
    for i in range(len(t_swia_mom)):
        Dt_swia_mom[i-1] = t_swia_mom[i]-t_swia_mom[i-1]
    Dt_swia_mom = Dt_swia_mom*3600 #expreso los intervalos en segundos
    
    plt.figure(6, figsize = (30,20), tight_layout = True)
    plt.title('Intervalos temporales entre mediciones de momentos SWIA', fontsize = 30)
    plt.plot(t_swia_mom[:-1], Dt_swia_mom, 'o-', linewidth = 2)
    plt.ylabel('Intervalo entre mediciones [s]', fontsize = 20)
    plt.xlabel('Tiempo [hora decimal]', fontsize = 20)
    plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
    plt.grid(axis = 'both', which = 'both', alpha = 0.8, linewidth = 2, linestyle = '--')
    plt.show()
    
    plt.savefig(path_analisis+'intervalos_mediciones_SWIAmom_{}'.format(shock_date))
        
    
    #lo veo en forma de histograma para mostrar que la mayoria de las mediciones respetan la frecuencia de muestreo
    
    plt.figure(7, figsize = (30,20), tight_layout = True)
    plt.title('Intervalos temporales entre mediciones de momentos SWIA', fontsize = 30)   
    plt.hist(Dt_swia_mom, 30)
    plt.ylabel('Cantidad de eventos', fontsize = 20)
    plt.xlabel('Intervalo entre mediciones [s]', fontsize = 20)
    plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
    plt.grid(axis = 'both', which = 'both', alpha = 0.8, linewidth = 2, linestyle = '--')
    plt.show()
    
    plt.savefig(path_analisis+'hist_intervalos_mediciones_SWIAmom_{}'.format(shock_date))
    
    #zoom del hsitograma
    
    plt.figure(8, figsize = (30,20), tight_layout = True)
    plt.title('Intervalos temporales entre mediciones de momentos SWIA - ampliación del histograma', fontsize = 30)   
    plt.hist(Dt_swia_mom, 30)
    plt.ylabel('Cantidad de eventos', fontsize = 20)
    plt.ylim(ymin = 0, ymax = 100)
    plt.xlabel('Intervalo entre mediciones [s]', fontsize = 20)
    plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
    plt.grid(axis = 'both', which = 'both', alpha = 0.8, linewidth = 2, linestyle = '--')
    plt.show()
    
    plt.savefig(path_analisis+'hist_zoom_intervalos_mediciones_SWIAmom_{}'.format(shock_date))
    
    
    #para los espectros
    
    Dt_swia_spec = np.empty([len(t_swia_spec)-1])     
    for i in range(len(t_swia_spec)):
        Dt_swia_spec[i-1] = t_swia_spec[i]-t_swia_spec[i-1]
    Dt_swia_spec = Dt_swia_spec*3600 #expreso los intervalos en segundos
    
    plt.figure(9, figsize = (30,20), tight_layout = True)
    plt.title('Intervalos temporales entre mediciones de espectros SWIA', fontsize = 30)
    plt.plot(t_swia_spec[:-1], Dt_swia_spec, 'o-', linewidth = 2)
    plt.ylabel('Intervalo entre mediciones [s]', fontsize = 20)
    plt.xlabel('Tiempo [hora decimal]', fontsize = 20)
    plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
    plt.grid(axis = 'both', which = 'both', alpha = 0.8, linewidth = 2, linestyle = '--')
    plt.show()
    
    plt.savefig(path_analisis+'intervalos_mediciones_SWIAspec_{}'.format(shock_date))
        
    
    #lo veo en forma de histograma para mostrar que la mayoria de las mediciones respetan la frecuencia de muestreo
    
    plt.figure(10, figsize = (30,20), tight_layout = True)
    plt.title('Intervalos temporales entre mediciones de espectros SWIA', fontsize = 30)   
    plt.hist(Dt_swia_spec, 30)
    plt.ylabel('Cantidad de eventos', fontsize = 20)
    plt.xlabel('Intervalo entre mediciones [s]', fontsize = 20)
    plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
    plt.grid(axis = 'both', which = 'both', alpha = 0.8, linewidth = 2, linestyle = '--')
    plt.show()
    
    plt.savefig(path_analisis+'hist_intervalos_mediciones_SWIAspec_{}'.format(shock_date))
    
    #zoom del hsitograma
    
    plt.figure(11, figsize = (30,20), tight_layout = True)
    plt.title('Intervalos temporales entre mediciones de espectros SWIA - ampliación del histograma', fontsize = 30)   
    plt.hist(Dt_swia_spec, 30)
    plt.ylabel('Cantidad de eventos', fontsize = 20)
    plt.ylim(ymin = 0, ymax = 100)
    plt.xlabel('Intervalo entre mediciones [s]', fontsize = 20)
    plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
    plt.grid(axis = 'both', which = 'both', alpha = 0.8, linewidth = 2, linestyle = '--')
    plt.show()
    
    plt.savefig(path_analisis+'hist_zoom_intervalos_mediciones_SWIAspec_{}'.format(shock_date))