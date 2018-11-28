
from mag import shock_date
from delimitacion_shock import iapo1, iapo2, B, Bx, By, Bz, t_mag, i_u, f_u, i_d, f_d, Bu, Bd, norm_Bu, norm_Bd, std_Bu, std_Bd, std_norm_Bu, std_norm_Bd

from importlib import reload
import numpy as np
import matplotlib.pyplot as plt
import os

#importo Bx, By, Bz de baja frecuencia para ver el perfil de B en distintas frecuencias en simultaneo
path_mag_low = r'C:\Users\sofia\Documents\Facultad\Tesis\Datos Maven\MAG/baja_frec/'
d_low, day_frac_low, Bx_low, By_low, Bz_low = np.loadtxt(path_mag_low+'mvn_mag_l2_2014359ss1s_20141225_v01_r01.sts', skiprows = 147, usecols = (1,6,7,8,9),unpack = True)

#vector de tiempos en hora decimal
t_mag_low = (day_frac_low-d_low[5])*24
#modulo de B en coordenadas MSO
B_low = np.sqrt(Bx_low**2 + By_low**2 + Bz_low**2)

#me quedo con los datos de la orbita
B_low = B_low[iapo1:iapo2]
t_mag_low[iapo1:iapo2]


#carpeta para guardar resultados
path_analisis = r'C:\Users\sofia\Documents\Facultad\Tesis\Analisis/{}/'.format(shock_date)
if not os.path.exists(path_analisis):
    os.makedirs(path_analisis)


#%% FUNCIONES

def lim_int_ext(Bx, By, Bz, B, Bu, norm_Bu, std_Bu, std_norm_Bu, inicio, fin, factor_sigma):
    
    '''
    Busca los tiempos en los que Bx,By,Bz y B superan el valor de Bu (o Bd) en 3 veces
    su dispersion. Luego toma como limite a izquierda el menor de esos tiempos, y como
    limite a derecha el mayor.
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




def fit_ramp(t_mag, B, i1_ramp_eye, i2_ramp_eye, f_over):
    
    '''
    Calcula fit lineal de los datos de la rampa, tomando como inicio
    un punto entre los limites de inicio de rampa marcados a ojo y 
    como fin f_over. Va cambiando el punto de incio moviendose de a
    1 punto, empezando por i1_ramp_eye y terminando en i2_ramp_eye.
    
    Calcula el error del fit con cada punto de inicio y despues devuelve
    el indice y los parametros y cov del fit para el mejor fit.    
    '''
    
    M = len(B[i1_ramp_eye:i2_ramp_eye])
    params = np.empty([M,2])
    cov = np.empty([2,2,M])
    err_fit = np.empty([M])
    
    for m in range(M):
        
        if len(B[i1_ramp_eye+m:f_over]) < 3: #si tengo dos puntos para fitear no tiene sentido hacer fit lineal
            break
        
        else:
            params[m,:], cov[:,:,m] = np.polyfit(t_mag[i1_ramp_eye+m:f_over], B[i1_ramp_eye+m:f_over], 1, cov = True)
            
            #errores relativos de los parametros
            err_pendiente = cov[0,0,m]/params[m,0]
            err_ordenada = cov[1,1,m]/params[m,1]
            
            #defino error del fit como el menor entre sus dos err rel
            err_fit[m] = np.min([err_pendiente, err_ordenada])
    
    #me quedo con el fit con el menor error
    index = list(err_fit).index(min(err_fit))
    indice =  index + i1_ramp_eye
    
    return indice, params[index,:], cov[:,:,index]


#%% ploteo para elegir limites a ojo


plt.figure(0, tight_layout = True)
plt.suptitle('Delimitacion de subestructuras', fontsize = 20)

plt.subplot(11)
plt.plot(t_mag, B, linewidth = 2, marker = 'o', markersize = 3)
#regiones up/down
plt.axvspan(xmin = t_mag[i_u], xmax = t_mag[f_u], facecolor = 'r', alpha = 0.15, label = 'Upstream')
plt.axvspan(xmin = t_mag[i_d], xmax = t_mag[f_d], facecolor = 'y', alpha = 0.15, label = 'Downstream')
#asintotas de Bu y Bd
plt.axhline(y = norm_Bu, linewidth = 2, color = 'r')
plt.axhline(y = norm_Bd, linewidth = 2, color = 'r')

plt.ylabel(r'$|\vec{B}|$ [nT]', fontsize = 20)
plt.xlim(9.5,10.5)
plt.ylim(0,50)
plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
plt.grid(axis = 'both', which = 'both', alpha = 0.8, linewidth = 2, linestyle = '--')
plt.legend(loc = 0, fontsize = 15)


plt.subplot(21)
plt.plot(t_mag_low, B_low, linewidth = 2, marker = 'o', markersize = 3)
#regiones up/down
plt.axvspan(xmin = t_mag[i_u], xmax = t_mag[f_u], facecolor = 'r', alpha = 0.15, label = 'Upstream')
plt.axvspan(xmin = t_mag[i_d], xmax = t_mag[f_d], facecolor = 'y', alpha = 0.15, label = 'Downstream')

plt.xlabel('Tiempo [hora decimal]', fontsize = 20)
plt.ylabel(r'$|\vec{B}|$ [nT]', fontsize = 20)
plt.xlim(9.5,10.5)
plt.ylim(0,50)
plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
plt.grid(axis = 'both', which = 'both', alpha = 0.8, linewidth = 2, linestyle = '--')
plt.legend(loc = 0, fontsize = 15)


#%% limites a ojo


#FOOT      Solo marco el incio del foot, el final lo marco al delimitar el incio de la rampa

'''
Afuera de Ti1 seguro no estoy en el foot.
Afuera de Ti2 seguro estoy adentro del foot.
Entonces el foot empieza en algun lado en entre estos dos limites.
'''

Ti1_foot_eye = 9.81375 #*
Ti2_foot_eye = 9.81375 #*

i1_foot_eye = (np.abs(t_mag - Ti1_foot_eye)).argmin()
i2_foot_eye = (np.abs(t_mag - Ti2_foot_eye)).argmin()


#RAMP      Solo marco el incio de la ramp, el final lo marco al delimitar el incio del overshoot

Ti1_ramp_eye = 9.81375 #*
Ti2_ramp_eye = 9.81375 #*

i1_ramp_eye = (np.abs(t_mag - Ti1_ramp_eye)).argmin()
i2_ramp_eye = (np.abs(t_mag - Ti2_ramp_eye)).argmin()


#OVERSHOOT

Ti1_over_eye = 9.81375 #*
Ti2_over_eye = 9.81375 #*
Tf1_over_eye = 9.81375 #*
Tf2_over_eye = 9.81375 #*

i1_over_eye = (np.abs(t_mag - Ti1_over_eye)).argmin()
i2_over_eye = (np.abs(t_mag - Ti2_over_eye)).argmin()
f1_over_eye = (np.abs(t_mag - Ti1_over_eye)).argmin()
f2_over_eye = (np.abs(t_mag - Ti2_over_eye)).argmin()


#%%

plt.figure(1, tight_layout = True)
plt.suptitle('Delimitacion de subestructuras - limites a ojo', fontsize = 20)

plt.subplot(11)

plt.plot(t_mag, B, linewidth = 2, marker = 'o', markersize = 3)
#regiones up/down
plt.axvspan(xmin = t_mag[i_u], xmax = t_mag[f_u], facecolor = 'r', alpha = 0.15, label = 'Upstream')
plt.axvspan(xmin = t_mag[i_d], xmax = t_mag[f_d], facecolor = 'y', alpha = 0.15, label = 'Downstream')
#asintotas de Bu y Bd
plt.axhline(y = norm_Bu, linewidth = 2, color = 'r')
plt.axhline(y = norm_Bd, linewidth = 2, color = 'r')
#inicio foot
plt.axvline(x = Ti1_foot_eye, linewidth = 2, linestyle = '--', color = 'k')
plt.axvline(x = Ti2_foot_eye, linewidth = 2, linestyle = '--', color = 'k')
#inicio ramp
plt.axvline(x = Ti1_ramp_eye, linewidth = 2, linestyle = '--', color = 'k')
plt.axvline(x = Ti2_ramp_eye, linewidth = 2, linestyle = '--', color = 'k')
#inicio overshoot
plt.axvline(x = Ti1_over_eye, linewidth = 2, linestyle = '--', color = 'k')
plt.axvline(x = Ti2_over_eye, linewidth = 2, linestyle = '--', color = 'k')
#final overshoot
plt.axvline(x = Tf1_over_eye, linewidth = 2, linestyle = '--', color = 'k')
plt.axvline(x = Tf2_over_eye, linewidth = 2, linestyle = '--', color = 'k')

plt.ylabel(r'$|\vec{B}|$ [nT]', fontsize = 20)
plt.xlim(9.5,10.5)
plt.ylim(0,50)
plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
plt.grid(axis = 'both', which = 'both', alpha = 0.8, linewidth = 2, linestyle = '--')
plt.legend(loc = 0, fontsize = 15)


plt.subplot(21)

plt.plot(t_mag_low, B_low, linewidth = 2, marker = 'o', markersize = 3)
#regiones up/down
plt.axvspan(xmin = t_mag[i_u], xmax = t_mag[f_u], facecolor = 'r', alpha = 0.15, label = 'Upstream')
plt.axvspan(xmin = t_mag[i_d], xmax = t_mag[f_d], facecolor = 'y', alpha = 0.15, label = 'Downstream')
#inicio foot
plt.axvline(x = Ti1_foot_eye, linewidth = 2, linestyle = '--', color = 'k')
plt.axvline(x = Ti2_foot_eye, linewidth = 2, linestyle = '--', color = 'k')
#inicio ramp
plt.axvline(x = Ti1_ramp_eye, linewidth = 2, linestyle = '--', color = 'k')
plt.axvline(x = Ti2_ramp_eye, linewidth = 2, linestyle = '--', color = 'k')
#inicio overshoot
plt.axvline(x = Ti1_over_eye, linewidth = 2, linestyle = '--', color = 'k')
plt.axvline(x = Ti2_over_eye, linewidth = 2, linestyle = '--', color = 'k')
#final overshoot
plt.axvline(x = Tf1_over_eye, linewidth = 2, linestyle = '--', color = 'k')
plt.axvline(x = Tf2_over_eye, linewidth = 2, linestyle = '--', color = 'k')

plt.xlabel('Tiempo [hora decimal]', fontsize = 20)
plt.ylabel(r'$|\vec{B}|$ [nT]', fontsize = 20)
plt.xlim(9.5,10.5)
plt.ylim(0,50)
plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
plt.grid(axis = 'both', which = 'both', alpha = 0.8, linewidth = 2, linestyle = '--')
plt.legend(loc = 0, fontsize = 15)


#%% automatizo delimitaciones


#FOOT inicio
ti1_foot, ti2_foot, i1_foot, i1_foot = lim_int_ext(Bx, By, Bz, B, Bu, norm_Bu, std_Bu, std_norm_Bu, i1_foot_eye-70, i2_foot_eye+70, 3)
i_foot = (np.abs(t_mag - (np.abs(ti2_foot - ti1_foot)/2))).argmin()


#OVERSHOOT

#inicio
ti1_over, ti2_over, i1_over, i2_over = lim_int_ext(Bx, By, Bz, B, Bd, norm_Bd, std_Bd, std_norm_Bd, i1_over_eye-30, i2_over_eye+30, 3)
i_over = (np.abs(t_mag - (np.abs(ti2_over - ti1_over)/2))).argmin()

#final
tf1_over, tf2_over, f1_over, f2_over = lim_int_ext(Bx, By, Bz, B, Bd, norm_Bd, std_Bd, std_norm_Bd, f1_over_eye-30, f2_over_eye+30, 3)
f_over = (np.abs(t_mag - (np.abs(tf2_over - tf1_over)/2))).argmin()


#inicio RAMP
i1_ramp, i1_params, i1_cov = fit_ramp(t_mag, B, i1_ramp_eye, i2_ramp_eye, f1_over)
i2_ramp, i2_params, i2_cov = fit_ramp(t_mag, B, i1_ramp_eye, i2_ramp_eye, f2_over)

ti1_ramp = t_mag[i1_ramp]
ti2_ramp = t_mag[i2_ramp]


#%%

plt.figure(1, tight_layout = True)
plt.title('Delimitacion de subestructuras', fontsize = 20)

plt.plot(t_mag, B, linewidth = 2)
#regiones up/down
plt.axvspan(xmin = t_mag[i_u], xmax = t_mag[f_u], facecolor = 'r', alpha = 0.15, label = 'Upstream')
plt.axvspan(xmin = t_mag[i_d], xmax = t_mag[f_d], facecolor = 'y', alpha = 0.15, label = 'Downstream')
#asintotas de Bu y Bd
plt.axhline(y = norm_Bu, linewidth = 2, color = 'r')
plt.axhline(y = norm_Bd, linewidth = 2, color = 'r')
#inicio foot
plt.axvline(x = Ti1_foot_eye, linewidth = 2, linestyle = '--', color = 'k')
plt.axvline(x = Ti2_foot_eye, linewidth = 2, linestyle = '--', color = 'k')
plt.axvline(x = ti1_foot, linewidth = 2, linestyle = '-', color = 'm')
plt.axvline(x = ti2_foot, linewidth = 2, linestyle = '-', color = 'm')
#inicio ramp
plt.axvline(x = Ti1_ramp_eye, linewidth = 2, linestyle = '--', color = 'k')
plt.axvline(x = Ti2_ramp_eye, linewidth = 2, linestyle = '--', color = 'k')
plt.axvline(x = ti1_ramp, linewidth = 2, linestyle = '-', color = 'm')
plt.axvline(x = ti2_ramp, linewidth = 2, linestyle = '-', color = 'm')
#inicio overshoot
plt.axvline(x = Ti1_over_eye, linewidth = 2, linestyle = '--', color = 'k')
plt.axvline(x = Ti2_over_eye, linewidth = 2, linestyle = '--', color = 'k')
plt.axvline(x = ti1_over, linewidth = 2, linestyle = '-', color = 'm')
plt.axvline(x = ti2_over, linewidth = 2, linestyle = '-', color = 'm')
#final overshoot
plt.axvline(x = Tf1_over_eye, linewidth = 2, linestyle = '--', color = 'k')
plt.axvline(x = Tf2_over_eye, linewidth = 2, linestyle = '--', color = 'k')
plt.axvline(x = tf1_over, linewidth = 2, linestyle = '-', color = 'm')
plt.axvline(x = tf2_over, linewidth = 2, linestyle = '-', color = 'm')

plt.xlabel('Tiempo [hora decimal]', fontsize = 20)
plt.ylabel(r'$|\vec{B}|$ [nT]', fontsize = 20)
plt.xlim(9.5,10.5)
plt.ylim(0,50)
plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
plt.grid(axis = 'both', which = 'both', alpha = 0.8, linewidth = 2, linestyle = '--')
plt.legend(loc = 0, fontsize = 15)

