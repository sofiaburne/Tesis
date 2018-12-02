
from mag import shock_date
import delimitacion_shock as ds
from delimitacion_shock import v_nave, t_apo11, t_apo12, t_apo21, t_apo22, x, y, z, B, Bx, By, Bz, t_mag, t_swia_mom, densidad_swia, t_swea, nivelesenergia_swea, flujosenergia_swea, t_swia_spec, nivelesenergia_swia, flujosenergia_swia, i_u, f_u, i_d, f_d, Bu, Bd, norm_Bu, norm_Bd, std_Bu, std_Bd, std_norm_Bu, std_norm_Bd
import funciones_fit_bowshock as fbow
import funciones_coplanaridad as fcop


from importlib import reload
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

#importo Bx, By, Bz de baja frecuencia para ver el perfil de B en distintas frecuencias en simultaneo
path_mag_low = r'C:\Users\sofia\Documents\Facultad\Tesis\Datos Maven\MAG\baja_frec/'
d_low, day_frac_low, Bx_low, By_low, Bz_low, X_low, Y_low, Z_low = np.loadtxt(path_mag_low+'2014/12/mvn_mag_l2_2014359ss1s_20141225_v01_r01.sts', skiprows = 147, usecols = (1,6,7,8,9,11,12,13),unpack = True)

#en radios marcianos
x_low, y_low, z_low = X_low/3390, Y_low/3390, Z_low/3390
#vector de tiempos en hora decimal
t_mag_low = (day_frac_low-d_low[5])*24
#modulo de B en coordenadas MSO
B_low = np.sqrt(Bx_low**2 + By_low**2 + Bz_low**2)

#me quedo con los datos de la orbita
R_apo1_low, iapo1_low = ds.orbita(t_apo11,t_apo12,t_mag_low,x_low,y_low,z_low)
R_apo2_low, iapo2_low = ds.orbita(t_apo21,t_apo22,t_mag_low,x_low,y_low,z_low)
B_low = B_low[iapo1_low:iapo2_low]
t_mag_low = t_mag_low[iapo1_low:iapo2_low]


#carpeta para guardar resultados
path_analisis = r'C:\Users\sofia\Documents\Facultad\Tesis\Analisis/{}/'.format(shock_date)
if not os.path.exists(path_analisis):
    os.makedirs(path_analisis)


#%% FUNCIONES

def lim_int_ext(Bx, By, Bz, B, Bu, norm_Bu, std_Bu, std_norm_Bu, inicio, fin, factor_sigma):
    
    '''
    Busca los tiempos (y sus indices) en los que Bx, By, Bz y B superan el valor de Bu (o Bd) en
    factor_sigma veces su dispersion. Luego toma como limite a izquierda el menor de esos tiempos,
    y como limite a derecha el mayor.
    '''

    index_Bx = (np.abs(Bx[inicio:fin] - (Bu[0] + factor_sigma*std_Bu[0]))).argmin()
    index_By = (np.abs( By[inicio:fin] - (Bu[1] + factor_sigma*std_Bu[1]))).argmin()
    index_Bz = (np.abs(Bz[inicio:fin] - (Bu[2] + factor_sigma*std_Bu[2]))).argmin()
    index_B = (np.abs(B[inicio:fin] - (norm_Bu + factor_sigma*std_norm_Bu))).argmin()
    
    t_izq = np.min([t_mag[inicio+index_Bx], t_mag[inicio+index_By], t_mag[inicio+index_Bz], t_mag[inicio+index_B]])
    t_der = np.max([t_mag[inicio+index_Bx], t_mag[inicio+index_By], t_mag[inicio+index_Bz], t_mag[inicio+index_B]])
    index_izq = list(t_mag).index(t_izq)
    index_der = list(t_mag).index(t_der)
    
    return t_izq, t_der, index_izq, index_der




def chired(ydata, ymod, param_fit):
    
    '''
    Calcula el chi cuadrado reducido de un ajuste lineal respecto a los parámetros
    (ej: un ajuste lineal, que tiene param_fit = 2)
    '''
    sigma_sq = np.mean(ydata**2) - (np.mean(ydata))**2
    
    chi = np.sum(((ydata - ymod)**2)/sigma_sq)
    
    chi_red = chi/(len(ydata) - param_fit)
    
    return chi_red
 



def fit_ramp(t_mag, B, i1_ramp_eye, i2_ramp_eye, i1_over_eye, i2_over_eye):
    
    '''
    Para determinar el limite a izquierda del incio de la rampa hago un fit lineal entre
    todos los puntos de la primera mitad del rango elegido a ojo para el incio de la rampa
    y todos los puntos del rango para el inicio del overshoot. Para un dado punto de la región
    "inicio overshoot", barro todos los puntos de la primer mitad de "inicio rampa a ojo" y me
    quedo con el fit que mejor chi cuadrado tenga. Después, elijo el mejor entre todos los fits
    de cada punto de la región "inicio overshoot".
    
    Hago lo mismo para determinar el limite a derecha del inicio de la rampa, pero barriendo la
    segunda mitad de la región "inicio rampa a ojo".
    '''
    
    #t_mag = t_mag
    #B = B
    #i1_ramp_eye = i1_ramp_eye
    #i2_ramp_eye = i2_ramp_eye
    #i1_over_eye = i1_over
    #i2_over_eye = i2_over
    
    i_half = int(abs(i2_ramp_eye - i1_ramp_eye)/2) + i1_ramp_eye
    
    M = len(B[i1_ramp_eye:i_half+1])   
    print('# puntos inicio media rampa: ', M)
    K = len(B[i1_over_eye:i2_over_eye+1])
    print('# puntos fin rampa: ', K)
    
    index1 = np.empty([K])
    params1 = np.empty([K,2])
    err_fit1 = np.empty([K])
    
    index2 = np.empty([K])
    params2 = np.empty([K,2])
    err_fit2 = np.empty([K])
    
    #plt.plot(t_mag, B, linewidth = 2, marker = 'o', markersize = 5)
    ##inicio ramp
    #plt.axvline(x = t_mag[i1_ramp_eye], linewidth = 2, linestyle = '--', color = 'k')
    #plt.axvline(x = t_mag[i2_ramp_eye], linewidth = 2, linestyle = '--', color = 'k')
    ##inicio overshoot
    #plt.axvline(x = ti1_over, linewidth = 2, linestyle = '-', color = 'g')
    #plt.axvline(x = ti2_over, linewidth = 2, linestyle = '-', color = 'g')
    #plt.xlim(9.8194, 9.81957)
    
    for k in range(K):
        
        params1_m = np.empty([M,2])
        err_fit1_m = np.empty([M])
        
        params2_m = np.empty([M,2])
        err_fit2_m = np.empty([M])
        
        for m in range(M):
    
            #si tengo dos puntos para fitear no tiene sentido hacer fit lineal
            if len(B[i1_ramp_eye+m:i1_over_eye+k+1]) < 3:
                print(k,m)
                continue
            else:
                params1_m[m,:] = np.polyfit(t_mag[i1_ramp_eye+m:i1_over_eye+k+1], B[i1_ramp_eye+m:i1_over_eye+k+1], 1, cov = False)
                #defino error del fit a partir del chi cuadrado
                err_fit1_m[m] = chired(B[i1_ramp_eye+m:i1_over_eye+k+1], params1_m[m,0]*t_mag[i1_ramp_eye+m:i1_over_eye+k+1] + params1_m[m,1], 2)
                
                #plt.plot(t_mag[i1_ramp_eye+m:i1_over_eye+k+1], params1_m[m,0]*t_mag[i1_ramp_eye+m:i1_over_eye+k+1]+params1_m[m,1], linewidth = 2, color = 'C{}'.format(m))
                
                
        for m in range(M):
            
            if len(B[i_half+m:i1_over_eye+k+1]) < 3:
                continue
            else:
                params2_m[m,:] = np.polyfit(t_mag[i_half+m:i1_over_eye+k+1], B[i_half+m:i1_over_eye+k+1], 1, cov = False)
                #defino error del fit a partir del chi cuadrado
                err_fit2_m[m] = chired(B[i_half+m:i1_over_eye+k+1], params2_m[m,0]*t_mag[i_half+m:i1_over_eye+k+1] + params2_m[m,1], 2)
                
                #plt.plot(t_mag[i_half+m:i1_over_eye+k+1], params2_m[m,0]*t_mag[i_half+m:i1_over_eye+k+1]+params2_m[m,1], linewidth = 2, color = 'C{}'.format(m+3))
    
        
        #me quedo con el mejor fit para un dado k en cada mitad
       
        index1[k] = (np.abs(err_fit1_m - 1)).argmin()
        params1[k,:] = params1_m[int(index1[k]),:]
        err_fit1[k] = err_fit1_m[int(index1[k])]
        
        index2[k] = (np.abs(err_fit2_m - 1)).argmin()
        params2[k,:] = params2_m[int(index2[k]),:]
        err_fit2[k] = err_fit2_m[int(index2[k])]
    
    
    #me quedo con el mejor fit entre todos los k para cada mitad
    
    index_1 = (np.abs(err_fit1 - 1)).argmin()
    indicei1 =  int(index1[int(index_1)]) + i1_ramp_eye
    
    index_2 = (np.abs(err_fit2 - 1)).argmin()
    indicei2 =  int(index2[int(index_2)]) + i_half
    
    indicef1 = index_1 + i1_over_eye #indice del tiempo (perteneciente a la region inicio overshoot) con el que hice el fit para encontrar el mejor ti1_ramp
    indicef2 = index_2 + i1_over_eye #indice del tiempo (perteneciente a la region inicio overshoot) con el que hice el fit para encontrar el mejor ti2_ramp
        
    
    return indicei1, params1[index_1,:], err_fit1[index_1], indicei2, params2[index_2,:], err_fit2[index_2], indicef1, indicef2




def filter_data(data, fs_new = 8, fs_data = 32):
    
    '''
    Toma data y promedia cada j_av puntos tal que se pasa de tener
    una frecuencia de muestreo de fs_data mediciones por seg a fs_new
    mediciones por seg. Si el tamaño de data no es divisible en j_av
    puntos, entonces elimino los primeros m puntos necesarios para que
    lo sea.
    '''
  
    j_av = int(fs_data/fs_new)  #para tener frec de sampleo: fs_new Hz, tengo que promediar mis datos cada j_av puntos 
    
    if len(data)%j_av == 0:
        data_new = np.mean(data.reshape(-1, j_av), axis=1)
    
    else:
        m = len(data) - j_av*int(len(data)/j_av)
        data_new_ = data[m:]
        data_new = np.mean(data_new_.reshape(-1, j_av), axis=1)
    
    return data_new


#%% ploteo para elegir limites a ojo


# me genero un set de datos de 8 mediciones/s (una frec intermedia entre 32 y 1)
B_mid = filter_data(B)
t_mag_mid = filter_data(t_mag)



f0, ((p3,p1), (p5,p2), (p6,p4)) = plt.subplots(3,2, sharex = True) #ojo con sharex y los distintos inst

f0.suptitle('Datos MAVEN {}'.format(shock_date), fontsize = 20)
f0.tight_layout()

#B alta frec
p3.plot(t_mag, B, linewidth = 2, marker = 'o', markersize = 5)
#regiones up/down
p3.axvspan(xmin = t_mag[i_u], xmax = t_mag[f_u], facecolor = 'r', alpha = 0.15, label = 'Upstream')
p3.axvspan(xmin = t_mag[i_d], xmax = t_mag[f_d], facecolor = 'y', alpha = 0.15, label = 'Downstream')
#asintotas de Bu y Bd
p3.axhline(y = norm_Bu, linewidth = 2, color = 'r')
p3.axhline(y = norm_Bd, linewidth = 2, color = 'r')
p3.axhspan(ymin = norm_Bu - std_norm_Bu, ymax = norm_Bu + std_norm_Bu, facecolor = 'g', alpha = 0.4)
p3.axhspan(ymin = norm_Bd - std_norm_Bd, ymax = norm_Bd + std_norm_Bd, facecolor = 'g', alpha = 0.4)
p3.set_ylabel('$B$ [nT]\n32 Hz', fontsize = 20)
p3.axes.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
p3.axes.grid(axis = 'both', which = 'both', alpha = 0.8, linewidth = 2, linestyle = '--')
p3.legend(loc = 0, fontsize = 15)

#B media frec
p5.plot(t_mag_mid, B_mid, linewidth = 2, marker = 'o', markersize = 5)
#regiones up/down
p5.axvspan(xmin = t_mag[i_u], xmax = t_mag[f_u], facecolor = 'r', alpha = 0.15, label = 'Upstream')
p5.axvspan(xmin = t_mag[i_d], xmax = t_mag[f_d], facecolor = 'y', alpha = 0.15, label = 'Downstream')
#asintotas de Bu y Bd
p5.axhline(y = norm_Bu, linewidth = 2, color = 'r')
p5.axhline(y = norm_Bd, linewidth = 2, color = 'r')
p5.axhspan(ymin = norm_Bu - std_norm_Bu, ymax = norm_Bu + std_norm_Bu, facecolor = 'g', alpha = 0.4)
p5.axhspan(ymin = norm_Bd - std_norm_Bd, ymax = norm_Bd + std_norm_Bd, facecolor = 'g', alpha = 0.4)
p5.set_ylabel('$B$ [nT]\n8 Hz', fontsize = 20)
#p5.set_xlim(9.5,10.5)
#p5.set_ylim(0,50)
p5.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
p5.grid(axis = 'both', which = 'both', alpha = 0.8, linewidth = 2, linestyle = '--')
p5.legend(loc = 0, fontsize = 15)

#B baja frec
p6.plot(t_mag_low, B_low, linewidth = 2, marker = 'o', markersize = 5)
#regiones up/down
p6.axvspan(xmin = t_mag[i_u], xmax = t_mag[f_u], facecolor = 'r', alpha = 0.15, label = 'Upstream')
p6.axvspan(xmin = t_mag[i_d], xmax = t_mag[f_d], facecolor = 'y', alpha = 0.15, label = 'Downstream')
#asintotas de Bu y Bd
p6.axhline(y = norm_Bu, linewidth = 2, color = 'r')
p6.axhline(y = norm_Bd, linewidth = 2, color = 'r')
p6.set_ylabel('$B$ [nT]\n1 Hz', fontsize = 20)
p6.set_xlabel('Tiempo [hora decimal]', fontsize = 20)
p6.axes.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
p6.axes.grid(axis = 'both', which = 'both', alpha = 0.8, linewidth = 2, linestyle = '--')
p6.legend(loc = 0, fontsize = 15)


#espectros electrones
spec1 = p1.contourf(t_swea, nivelesenergia_swea, flujosenergia_swea.T, locator=ticker.LogLocator(), cmap='jet')
divider = make_axes_locatable(p1)
cax = divider.append_axes('top', size='5%', pad=0.3)
f0.colorbar(spec1, cax=cax, orientation='horizontal')
p1.axes.set_yscale('log')
#regiones up/down
p1.axvspan(xmin = t_mag[i_u], xmax = t_mag[f_u], facecolor = 'r', alpha = 0.15, label = 'Upstream')
p1.axvspan(xmin = t_mag[i_d], xmax = t_mag[f_d], facecolor = 'y', alpha = 0.15, label = 'Downstream')
p1.set_ylabel('Flujo electrones\n[$(cm^2 sr s)^{-1}$]', fontsize = 20)
p1.axes.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
p1.axes.grid(axis = 'both', which = 'major', alpha = 0.8, linewidth = 2, linestyle = '--')



#espectros iones
spec2 = p2.contourf(t_swia_spec, nivelesenergia_swia, flujosenergia_swia.T, locator=ticker.LogLocator(), cmap='jet')
divider = make_axes_locatable(p2)
cax = divider.append_axes('top', size='5%', pad=0.3)
f0.colorbar(spec2, cax=cax, orientation='horizontal')
p2.axes.set_yscale('log')
#regiones up/down
p2.axvspan(xmin = t_mag[i_u], xmax = t_mag[f_u], facecolor = 'r', alpha = 0.15, label = 'Upstream')
p2.axvspan(xmin = t_mag[i_d], xmax = t_mag[f_d], facecolor = 'y', alpha = 0.15, label = 'Downstream')
p2.set_ylabel('Flujo iones\n[$(cm^2 sr s)^{-1}$]', fontsize = 20)
p2.axes.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
p2.axes.grid(axis = 'both', which = 'major', alpha = 0.8, linewidth = 2, linestyle = '--')



#plot densidad swia
p4.plot(t_swia_mom, densidad_swia, linewidth = 2)
#regiones up/down
p4.axvspan(xmin = t_mag[i_u], xmax = t_mag[f_u], facecolor = 'r', alpha = 0.15, label = 'Upstream')
p4.axvspan(xmin = t_mag[i_d], xmax = t_mag[f_d], facecolor = 'y', alpha = 0.15, label = 'Downstream')
p4.set_ylabel('$n_p$\n[$cm^{-3}$]', fontsize = 20)
p4.set_xlabel('Tiempo [hora decimal]', fontsize = 20)
p4.axes.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
p4.axes.grid(axis = 'both', which = 'both', alpha = 0.8, linewidth = 2, linestyle = '--')


#%% limites a ojo


#FOOT      Solo marco el incio del foot, el final lo marco al delimitar el incio de la rampa

#Afuera de Ti1 seguro no estoy en el foot.
#Afuera de Ti2 seguro estoy adentro del foot.
#Entonces el foot empieza en algun lado en entre estos dos limites.
#Ojo con los paquetes de onditas sueltos

Ti1_foot_eye = 9.81436 #*
Ti2_foot_eye = 9.81596 #*

i1_foot_eye = (np.abs(t_mag - Ti1_foot_eye)).argmin()
i2_foot_eye = (np.abs(t_mag - Ti2_foot_eye)).argmin()


#RAMP      Solo marco el incio de la ramp, el final lo marco al delimitar el incio del overshoot

#tomar un intervalo con mas de dos puntos

Ti1_ramp_eye = 9.819427 #*
Ti2_ramp_eye = 9.819458 #*

i1_ramp_eye = (np.abs(t_mag - Ti1_ramp_eye)).argmin()
i2_ramp_eye = (np.abs(t_mag - Ti2_ramp_eye)).argmin()


#OVERSHOOT

#inicio

#cuando ya estoy por arriba de Bd

Ti1_over_eye = 9.81947 #*
Ti2_over_eye = 9.8195 #*
i1_over_eye = (np.abs(t_mag - Ti1_over_eye)).argmin()
i2_over_eye = (np.abs(t_mag - Ti2_over_eye)).argmin()

#fin

#cuando hay muchos puntos consecutivos por debajo de Bd ya esoty en el undershoot

Tf1_over_eye = 9.841666 #*
Tf2_over_eye = 9.84786 #*
f1_over_eye = (np.abs(t_mag - Tf1_over_eye)).argmin()
f2_over_eye = (np.abs(t_mag - Tf2_over_eye)).argmin()


#%% ploteo para ver si elegi bien los limites a ojo


f1, ((p3,p1), (p5,p2), (p6,p4)) = plt.subplots(3,2, sharex = True)

f1.suptitle('Datos MAVEN {}'.format(shock_date), fontsize = 20)
f1.tight_layout()

#B alta frec
p3.plot(t_mag, B, linewidth = 2, marker = 'o', markersize = 5)
#regiones up/down
p3.axvspan(xmin = t_mag[i_u], xmax = t_mag[f_u], facecolor = 'r', alpha = 0.15, label = 'Upstream')
p3.axvspan(xmin = t_mag[i_d], xmax = t_mag[f_d], facecolor = 'y', alpha = 0.15, label = 'Downstream')
#asintotas de Bu y Bd
p3.axhline(y = norm_Bu, linewidth = 2, color = 'r')
p3.axhline(y = norm_Bd, linewidth = 2, color = 'r')
p3.axhspan(ymin = norm_Bu - std_norm_Bu, ymax = norm_Bu + std_norm_Bu, facecolor = 'g', alpha = 0.4)
p3.axhspan(ymin = norm_Bd - std_norm_Bd, ymax = norm_Bd + std_norm_Bd, facecolor = 'g', alpha = 0.4)
#inicio foot
p3.axvline(x = Ti1_foot_eye, linewidth = 2, linestyle = '--', color = 'k')
p3.axvline(x = Ti2_foot_eye, linewidth = 2, linestyle = '--', color = 'k')
#inicio ramp
p3.axvline(x = Ti1_ramp_eye, linewidth = 2, linestyle = '--', color = 'k')
p3.axvline(x = Ti2_ramp_eye, linewidth = 2, linestyle = '--', color = 'k')
#inicio overshoot
p3.axvline(x = Ti1_over_eye, linewidth = 2, linestyle = '--', color = 'k')
p3.axvline(x = Ti2_over_eye, linewidth = 2, linestyle = '--', color = 'k')
#final overshoot
p3.axvline(x = Tf1_over_eye, linewidth = 2, linestyle = '--', color = 'k')
p3.axvline(x = Tf2_over_eye, linewidth = 2, linestyle = '--', color = 'k')
p3.set_ylabel('$B$ [nT]\n32 Hz', fontsize = 20)
p3.axes.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
p3.axes.grid(axis = 'both', which = 'both', alpha = 0.8, linewidth = 2, linestyle = '--')
p3.legend(loc = 0, fontsize = 15)

#B media frec
p5.plot(t_mag_mid, B_mid, linewidth = 2, marker = 'o', markersize = 5)
#regiones up/down
p5.axvspan(xmin = t_mag[i_u], xmax = t_mag[f_u], facecolor = 'r', alpha = 0.15, label = 'Upstream')
p5.axvspan(xmin = t_mag[i_d], xmax = t_mag[f_d], facecolor = 'y', alpha = 0.15, label = 'Downstream')
#asintotas de Bu y Bd
p5.axhline(y = norm_Bu, linewidth = 2, color = 'r')
p5.axhline(y = norm_Bd, linewidth = 2, color = 'r')
p5.axhspan(ymin = norm_Bu - std_norm_Bu, ymax = norm_Bu + std_norm_Bu, facecolor = 'g', alpha = 0.4)
p5.axhspan(ymin = norm_Bd - std_norm_Bd, ymax = norm_Bd + std_norm_Bd, facecolor = 'g', alpha = 0.4)
#inicio foot
p5.axvline(x = Ti1_foot_eye, linewidth = 2, linestyle = '--', color = 'k')
p5.axvline(x = Ti2_foot_eye, linewidth = 2, linestyle = '--', color = 'k')
#inicio ramp
p5.axvline(x = Ti1_ramp_eye, linewidth = 2, linestyle = '--', color = 'k')
p5.axvline(x = Ti2_ramp_eye, linewidth = 2, linestyle = '--', color = 'k')
#inicio overshoot
p5.axvline(x = Ti1_over_eye, linewidth = 2, linestyle = '--', color = 'k')
p5.axvline(x = Ti2_over_eye, linewidth = 2, linestyle = '--', color = 'k')
#final overshoot
p5.axvline(x = Tf1_over_eye, linewidth = 2, linestyle = '--', color = 'k')
p5.axvline(x = Tf2_over_eye, linewidth = 2, linestyle = '--', color = 'k')
p5.set_ylabel('$B$ [nT]\n8 Hz', fontsize = 20)
#p5.set_xlim(9.5,10.5)
#p5.set_ylim(0,50)
p5.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
p5.grid(axis = 'both', which = 'both', alpha = 0.8, linewidth = 2, linestyle = '--')
p5.legend(loc = 0, fontsize = 15)

#B baja frec
p6.plot(t_mag_low, B_low, linewidth = 2, marker = 'o', markersize = 5)
#regiones up/down
p6.axvspan(xmin = t_mag[i_u], xmax = t_mag[f_u], facecolor = 'r', alpha = 0.15, label = 'Upstream')
p6.axvspan(xmin = t_mag[i_d], xmax = t_mag[f_d], facecolor = 'y', alpha = 0.15, label = 'Downstream')
#inicio foot
p6.axvline(x = Ti1_foot_eye, linewidth = 2, linestyle = '--', color = 'k')
p6.axvline(x = Ti2_foot_eye, linewidth = 2, linestyle = '--', color = 'k')
#inicio ramp
p6.axvline(x = Ti1_ramp_eye, linewidth = 2, linestyle = '--', color = 'k')
p6.axvline(x = Ti2_ramp_eye, linewidth = 2, linestyle = '--', color = 'k')
#inicio overshoot
p6.axvline(x = Ti1_over_eye, linewidth = 2, linestyle = '--', color = 'k')
p6.axvline(x = Ti2_over_eye, linewidth = 2, linestyle = '--', color = 'k')
#final overshoot
p6.axvline(x = Tf1_over_eye, linewidth = 2, linestyle = '--', color = 'k')
p6.axvline(x = Tf2_over_eye, linewidth = 2, linestyle = '--', color = 'k')
#asintotas de Bu y Bd
p6.axhline(y = norm_Bu, linewidth = 2, color = 'r')
p6.axhline(y = norm_Bd, linewidth = 2, color = 'r')
p6.set_ylabel('$B$ [nT]\n1 Hz', fontsize = 20)
p6.set_xlabel('Tiempo [hora decimal]', fontsize = 20)
p6.axes.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
p6.axes.grid(axis = 'both', which = 'both', alpha = 0.8, linewidth = 2, linestyle = '--')
p6.legend(loc = 0, fontsize = 15)


#espectros electrones
spec1 = p1.contourf(t_swea, nivelesenergia_swea, flujosenergia_swea.T, locator=ticker.LogLocator(), cmap='jet')
divider = make_axes_locatable(p1)
cax = divider.append_axes('top', size='5%', pad=0.3)
f1.colorbar(spec1, cax=cax, orientation='horizontal')
p1.axes.set_yscale('log')
#regiones up/down
p1.axvspan(xmin = t_mag[i_u], xmax = t_mag[f_u], facecolor = 'r', alpha = 0.15, label = 'Upstream')
p1.axvspan(xmin = t_mag[i_d], xmax = t_mag[f_d], facecolor = 'y', alpha = 0.15, label = 'Downstream')
#inicio foot
p1.axvline(x = Ti1_foot_eye, linewidth = 2, linestyle = '--', color = 'k')
p1.axvline(x = Ti2_foot_eye, linewidth = 2, linestyle = '--', color = 'k')
#inicio ramp
p1.axvline(x = Ti1_ramp_eye, linewidth = 2, linestyle = '--', color = 'k')
p1.axvline(x = Ti2_ramp_eye, linewidth = 2, linestyle = '--', color = 'k')
#inicio overshoot
p1.axvline(x = Ti1_over_eye, linewidth = 2, linestyle = '--', color = 'k')
p1.axvline(x = Ti2_over_eye, linewidth = 2, linestyle = '--', color = 'k')
#final overshoot
p1.axvline(x = Tf1_over_eye, linewidth = 2, linestyle = '--', color = 'k')
p1.axvline(x = Tf2_over_eye, linewidth = 2, linestyle = '--', color = 'k')
p1.set_ylabel('Flujo electrones\n[$(cm^2 sr s)^{-1}$]', fontsize = 20)
p1.axes.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
p1.axes.grid(axis = 'both', which = 'major', alpha = 0.8, linewidth = 2, linestyle = '--')



#espectros iones
spec2 = p2.contourf(t_swia_spec, nivelesenergia_swia, flujosenergia_swia.T, locator=ticker.LogLocator(), cmap='jet')
divider = make_axes_locatable(p2)
cax = divider.append_axes('top', size='5%', pad=0.3)
f1.colorbar(spec2, cax=cax, orientation='horizontal')
p2.axes.set_yscale('log')
#regiones up/down
p2.axvspan(xmin = t_mag[i_u], xmax = t_mag[f_u], facecolor = 'r', alpha = 0.15, label = 'Upstream')
p2.axvspan(xmin = t_mag[i_d], xmax = t_mag[f_d], facecolor = 'y', alpha = 0.15, label = 'Downstream')
#inicio foot
p2.axvline(x = Ti1_foot_eye, linewidth = 2, linestyle = '--', color = 'k')
p2.axvline(x = Ti2_foot_eye, linewidth = 2, linestyle = '--', color = 'k')
#inicio ramp
p2.axvline(x = Ti1_ramp_eye, linewidth = 2, linestyle = '--', color = 'k')
p2.axvline(x = Ti2_ramp_eye, linewidth = 2, linestyle = '--', color = 'k')
#inicio overshoot
p2.axvline(x = Ti1_over_eye, linewidth = 2, linestyle = '--', color = 'k')
p2.axvline(x = Ti2_over_eye, linewidth = 2, linestyle = '--', color = 'k')
#final overshoot
p2.axvline(x = Tf1_over_eye, linewidth = 2, linestyle = '--', color = 'k')
p2.axvline(x = Tf2_over_eye, linewidth = 2, linestyle = '--', color = 'k')
p2.set_ylabel('Flujo iones\n[$(cm^2 sr s)^{-1}$]', fontsize = 20)
p2.axes.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
p2.axes.grid(axis = 'both', which = 'major', alpha = 0.8, linewidth = 2, linestyle = '--')



#plot densidad swia
p4.plot(t_swia_mom, densidad_swia, linewidth = 2)
#regiones up/down
p4.axvspan(xmin = t_mag[i_u], xmax = t_mag[f_u], facecolor = 'r', alpha = 0.15, label = 'Upstream')
p4.axvspan(xmin = t_mag[i_d], xmax = t_mag[f_d], facecolor = 'y', alpha = 0.15, label = 'Downstream')
#inicio foot
p4.axvline(x = Ti1_foot_eye, linewidth = 2, linestyle = '--', color = 'k')
p4.axvline(x = Ti2_foot_eye, linewidth = 2, linestyle = '--', color = 'k')
#inicio ramp
p4.axvline(x = Ti1_ramp_eye, linewidth = 2, linestyle = '--', color = 'k')
p4.axvline(x = Ti2_ramp_eye, linewidth = 2, linestyle = '--', color = 'k')
#inicio overshoot
p4.axvline(x = Ti1_over_eye, linewidth = 2, linestyle = '--', color = 'k')
p4.axvline(x = Ti2_over_eye, linewidth = 2, linestyle = '--', color = 'k')
#final overshoot
p4.axvline(x = Tf1_over_eye, linewidth = 2, linestyle = '--', color = 'k')
p4.axvline(x = Tf2_over_eye, linewidth = 2, linestyle = '--', color = 'k')
p4.set_ylabel('$n_p$\n[$cm^{-3}$]', fontsize = 20)
p4.set_xlabel('Tiempo [hora decimal]', fontsize = 20)
p4.axes.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
p4.axes.grid(axis = 'both', which = 'both', alpha = 0.8, linewidth = 2, linestyle = '--')


#%% automatizo delimitaciones


#FOOT inicio

#como en la region del foot hay muchas mediciones considero moverme 30 mediciones afuer del range a ojo
#como Bu varia poco, miro variaciones de 3 sigmas

ti1_foot, ti2_foot, i1_foot, i2_foot = lim_int_ext(Bx, By, Bz, B, Bu, norm_Bu, std_Bu, std_norm_Bu, i1_foot_eye, i2_foot_eye, 3)
i_foot = int(np.abs(i2_foot - i1_foot)/2) + i1_foot
ti_foot = t_mag[i_foot]


#OVERSHOOT

#inicio

#como hacia la rampa hay pocas mediciones, me muevo hasta 1 mediciones a la izq de el incio del overshoot a ojo
#como hacia el overshoot tengo mas mediciones, me muevo hasta 2 a la derecha del inicio del overshoot a ojo
#como Bd varia bastante, considero variaciones de 1 sigma

ti1_over, ti2_over, i1_over, i2_over = lim_int_ext(Bx, By, Bz, B, Bd, norm_Bd, std_Bd, std_norm_Bd, i1_over_eye, i2_over_eye, 1)
i_over = int(np.abs(i2_over - i1_over)/2) + i1_over
ti_over = t_mag[i_over]

#final

tf1_over, tf2_over, f1_over, f2_over = lim_int_ext(Bx, By, Bz, B, Bd, norm_Bd, std_Bd, std_norm_Bd, f1_over_eye, f2_over_eye, 1)
f_over = int(np.abs(f2_over - f1_over)/2) + f1_over
tf_over = t_mag[f_over]


#inicio RAMP

i1_ramp, i1_params, i1_chired, i2_ramp, i2_params, i2_chired, f1_rampfit, f2_rampfit = fit_ramp(t_mag, B, i1_ramp_eye, i2_ramp_eye, i1_over, i2_over)
ti1_ramp = t_mag[i1_ramp]
ti2_ramp = t_mag[i2_ramp]
i_ramp = int(np.abs(i2_ramp - i1_ramp)/2) + i1_ramp
ti_ramp = t_mag[i_ramp]


#%% ploteo todos los limites determinados (a ojo y automatizados)

plt.figure(2, tight_layout = True)
plt.title('Delimitacion de subestructuras', fontsize = 20)

plt.plot(t_mag, B, linewidth = 2, marker = 'o', markersize = 5)
#regiones up/down
plt.axvspan(xmin = t_mag[i_u], xmax = t_mag[f_u], facecolor = 'r', alpha = 0.15, label = 'Upstream')
plt.axvspan(xmin = t_mag[i_d], xmax = t_mag[f_d], facecolor = 'y', alpha = 0.15, label = 'Downstream')
#asintotas de Bu y Bd
plt.axhline(y = norm_Bu, linewidth = 2, color = 'r')
plt.axhline(y = norm_Bd, linewidth = 2, color = 'r')
plt.axhspan(ymin = norm_Bu - std_norm_Bu, ymax = norm_Bu + std_norm_Bu, facecolor = 'g', alpha = 0.1)
plt.axhspan(ymin = norm_Bd - std_norm_Bd, ymax = norm_Bd + std_norm_Bd, facecolor = 'g', alpha = 0.1)
#inicio foot
plt.axvline(x = Ti1_foot_eye, linewidth = 2, linestyle = '--', color = 'k')
plt.axvline(x = Ti2_foot_eye, linewidth = 2, linestyle = '--', color = 'k')
plt.axvline(x = ti1_foot, linewidth = 2, linestyle = '-', color = 'm')
plt.axvline(x = ti2_foot, linewidth = 2, linestyle = '-', color = 'm')
#inicio ramp
plt.axvline(x = Ti1_ramp_eye, linewidth = 2, linestyle = '--', color = 'k')
plt.axvline(x = Ti2_ramp_eye, linewidth = 2, linestyle = '--', color = 'k')
plt.axvline(x = ti1_ramp, linewidth = 2, linestyle = '-', color = 'c')
plt.axvline(x = ti2_ramp, linewidth = 2, linestyle = '-', color = 'c')
#ajustes ramp
plt.plot(t_mag[i1_ramp:f1_rampfit+1], i1_params[0]*t_mag[i1_ramp:f1_rampfit+1] + i1_params[1], linewidth = 2, color = 'y', linestyle = '-.')
plt.plot(t_mag[i2_ramp:f2_rampfit+1], i2_params[0]*t_mag[i2_ramp:f2_rampfit+1] + i2_params[1], linewidth = 2, color = 'r', linestyle = '-.')
#inicio overshoot
plt.axvline(x = Ti1_over_eye, linewidth = 2, linestyle = '--', color = 'k')
plt.axvline(x = Ti2_over_eye, linewidth = 2, linestyle = '--', color = 'k')
plt.axvline(x = ti1_over, linewidth = 2, linestyle = '-', color = 'g')
plt.axvline(x = ti2_over, linewidth = 2, linestyle = '-', color = 'g')
#final overshoot
plt.axvline(x = Tf1_over_eye, linewidth = 2, linestyle = '--', color = 'k')
plt.axvline(x = Tf2_over_eye, linewidth = 2, linestyle = '--', color = 'k')
plt.axvline(x = tf1_over, linewidth = 2, linestyle = '-', color = 'y')
plt.axvline(x = tf2_over, linewidth = 2, linestyle = '-', color = 'y')

plt.xlabel('Tiempo [hora decimal]', fontsize = 20)
plt.ylabel(r'$|\vec{B}|$ [nT]', fontsize = 20)
plt.xlim(9.5,10.5)
plt.ylim(0,50)
plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
plt.grid(axis = 'both', which = 'both', alpha = 0.8, linewidth = 2, linestyle = '--')
plt.legend(loc = 0, fontsize = 15)


#%%

#centro del shock (centro de la rampa)
tc = abs(ti_over - ti_ramp)/2 + np.min(ti_ramp,ti_over)
C = (np.abs(t_mag-tc)).argmin()
#posicion de la nave en el centro del shock
Rc = np.array([x[C], y[C], z[C]])


#ancho temporal del shock en seg (inicio foot : final ramp)
ancho_shock_temp = 3600*abs(ti_foot - ti_over)
#ancho espacial del shock en km
ancho_shock = ancho_shock_temp*np.array([abs(v_nave[0]), abs(v_nave[1]), abs(v_nave[2])])
norm_ancho_shock = np.linalg.norm(ancho_shock)


#normal del shock reescalando el fit macro del bowshock
L = fbow.L_fit(Rc) #redefino L para que el fit contenga el centro de mi shock y calculo normal del fit
N = fbow.norm_fit_MGS(Rc[0], Rc[1], Rc[2], L)
#angulo entre campo upstream y normal del fit
theta_N = fcop.alpha(Bu,N)
#angulo entre posicion de la nave en el centro del shock y normal del fit
theta_NRc = fcop.alpha(Rc,N)
















