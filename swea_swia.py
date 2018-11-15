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
import datetime as dt
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

#Datos swia (cambiar con el shock de interes)

#momentos
path_swia_mom = r'C:\Users\sofia\Documents\Facultad\Tesis\Datos Maven\SWIA/momentos/'
data_swia_mom = cdflib.CDF(path_swia_mom+'2014/mvn_swi_l2_onboardsvymom_20141225_v01_r00.cdf')
data_swia_mom.cdf_info()

#espectros
path_swia_spec =r'C:\Users\sofia\Documents\Facultad\Tesis\Datos Maven\SWIA/espectros/'
data_swia_spec = cdflib.CDF(path_swia_spec+'2014/mvn_swi_l2_onboardsvyspec_20141225_v01_r00.cdf')
data_swia_spec.cdf_info()

#%%

#variables swea

timeunix_swea = data_swea.varget('time_unix')
energyflux_swea = data_swea.varget('diff_en_fluxes')
energylevel_swea = data_swea.varget('energy')

#los guardo en otros array que se mantengan cuando cierre el archivo
tu_swea = np.asarray(timeunix_swea)
flujosenergia_swea = np.asarray(energyflux_swea)
nivelesenergia_swea = np.asarray(energylevel_swea)

data_swea.close()

t_swea = tfromtimeunix(tu_swea) #tomo t_swea como la hora decimal

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

t_swia_mom = tfromtimeunix(tu_swia_mom) #hora decimal
t_swia_spec = tfromtimeunix(tu_swia_spec) #hora decimal

velocidad_swia_norm = np.sqrt(velocidad_swia[:,0]**2 + velocidad_swia[:,1]**2 + velocidad_swia[:,2]**2)
temperatura_swia_norm = np.sqrt(temperatura_swia[:,0]**2 + temperatura_swia[:,1]**2 + temperatura_swia[:,2]**2)






