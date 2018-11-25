from mag import B, Bx, By, Bz, t_mag, shock_date

from importlib import reload
import numpy as np
import matplolib.pyplot as plt
import os

path_analisis = r'C:\Users\sofia\Documents\Facultad\Tesis\Analisis/{}/'.format(shock_date)
if not os.path.exists(path_analisis):
    os.makedirs(path_analisis)

#%%

#me quedo con los datos en la orbita del shock
    

#estos valores los saco de 'caracteristicas_generales_shock_{}.txt'
Ti_orbit = 8.82096825600047     #*
Tf_orbit = 13.401525767999374   #*

#busco valor mas cercano porque ahora t_mag es de alta frec
i_orbit = (np.abs(t_mag - Ti_orbit)).argmin()
f_orbit = (np.abs(t_mag - Tf_orbit)).argmin()

B = B[i_orbit:f_orbit]
Bx = Bx[i_orbit:f_orbit]
By = By[i_orbit:f_orbit]
Bz = Bz[i_orbit:f_orbit]
t_mag = t_mag[i_orbit:f_orbit]

#%%


#FOOT

#primero marco limites a ojo que delimiten donde buscar el foot
i_foot_eye = (np.abs(t_mag - 9.75)).argmin()
f_foot_eye = (np.abs(t_mag - 9.8)).argmin()


#inicio foot, delimitacion mas fina

#me fijo cuando Bx, By, Bz y B superan el valor de Bu en 3 veces su desv estandar
ifoot_Bx = (np.abs( Bx[i_foot_eye:f_foot_eye] - (Bu[0] + 3*std_Bu[0]) )).argmin()
ifoot_By = (np.abs( By[i_foot_eye:f_foot_eye] - (Bu[1] + 3*std_Bu[1]) )).argmin()
ifoot_Bz = (np.abs( Bz[i_foot_eye:f_foot_eye] - (Bu[2] + 3*std_Bu[2]) )).argmin()
ifoot_B = (np.abs( B[i_foot_eye:f_foot_eye] - (norm_Bu + 3*std_norm_Bu) )).argmin()

#tomo como limite externo del inicio del foot al menor tiempo
#en el que Bx By Bz o B cumplen la condicion de arriba
#y el limite interno del inicio del foot como el mayor de ellos
ti_foot_ext = np.min([t_mag[i_foot_eye+ifoot_Bx], t_mag[i_foot_eye+ifoot_By], t_mag[i_foot_eye+ifoot_Bz], t_mag[i_foot_eye+ifoot_B]])
ti_foot_int = np.max([t_mag[i_foot_eye+ifoot_Bx], t_mag[i_foot_eye+ifoot_By], t_mag[i_foot_eye+ifoot_Bz], t_mag[i_foot_eye+ifoot_B]])

#%%

#RAMP

#primero marco limites a ojo que delimiten donde buscar la ramp
i_ramp_eye = (np.abs(t_mag - 9.75)).argmin() 
f_ramp_eye = (np.abs(t_mag - 9.8)).argmin() 


#fin de la rampa, delimitacion mas fina

#veo cuando B supera a Bd en 3 sigmas en el entorno del fin de la rampa
framp_Bx = (np.abs( Bx[f_ramp_eye-5:f_ramp_eye+5] - (Bd[0] + 3*std_Bd[0]) )).argmin()
framp_By = (np.abs( By[f_ramp_eye-5:f_ramp_eye+5] - (Bd[1] + 3*std_Bd[1]) )).argmin()
framp_Bz = (np.abs( Bz[f_ramp_eye-5:f_ramp_eye+5] - (Bd[2] + 3*std_Bd[2]) )).argmin()
framp_B = (np.abs( B[f_ramp_eye-5:f_ramp_eye+5] - (norm_Bd + 3*std_norm_Bd) )).argmin()

#saco lim ext e int
tf_ramp_int = np.min([t_mag[f_ramp_eye-5+framp_Bx], t_mag[f_ramp_eye-5+framp_By], t_mag[f_ramp_eye-5+framp_Bz], t_mag[f_ramp_eye-5+framp_B]])
tf_ramp_ext = np.max([t_mag[f_ramp_eye-5+framp_Bx], t_mag[f_ramp_eye-5+framp_By], t_mag[f_ramp_eye-5+framp_Bz], t_mag[f_ramp_eye-5+framp_B]])
f_ramp_int = list(t_mag).index(tf_ramp_int)
f_ramp_ext = list(t_mag).index(tf_ramp_ext)

#fit lineal del intervalo tomando lim ext e int, me quedo con el que tiene menor cov
#   !!!  no entiendo para que te sirve el fit si los limites igual van a ser los mismos
params_int, cov_int = np.polyfit(t_mag[i_ramp_eye:f_ramp_int], B[i_ramp_eye:f_ramp_int], 1, cov = True)
params_ext, cov_ext = np.polyfit(t_mag[i_ramp_eye:f_ramp_ext], B[i_ramp_eye:f_ramp_ext], 1, cov = True)

#errores relativos de los parametros del ajuste
err_m_int = cov_int[0,0]/params_int[0]
err_b_int = cov_int[1,1]/params_int[1]
err_m_ext = cov_ext[0,0]/params_ext[0]
err_b_ext = cov_ext[1,1]/params_ext[1]


#me quedo con el fit de menor error

if err_m_int > err_b_int:
    err_fit_int = err_m_int
else:
    err_fit_int = err_b_int

if err_m_ext > err_b_ext:
    err_fit_ext = err_m_ext
else:
    err_fit_ext = err_b_ext

if err_fit_int > err_fit_ext:
    best_params = params_ext
    best_cov = cov_ext
else:
    best_params = params_int
    best_cov = cov_int   


#%%

#OVERSHOOT

#primero marco limites a ojo que delimiten donde buscar el overshoot
i_over_eye = (np.abs(t_mag - 9.75)).argmin() 
f_over_eye = (np.abs(t_mag - 9.8)).argmin() 


#inicio del overshoot, delimitacion mas fina

#veo cuando B supera a Bd en 3 sigmas en el entorno del inicio del overshoot
iover_Bx = (np.abs( Bx[i_over_eye-5:i_over_eye+5] - (Bd[0] + 3*std_Bd[0]) )).argmin()
iover_By = (np.abs( By[i_over_eye-5:i_over_eye+5] - (Bd[1] + 3*std_Bd[1]) )).argmin()
iover_Bz = (np.abs( Bz[i_over_eye-5:i_over_eye+5] - (Bd[2] + 3*std_Bd[2]) )).argmin()
iover_B = (np.abs( B[i_over_eye-5:i_over_eye+5] - (norm_Bd + 3*std_norm_Bd) )).argmin()




#fin del overshoot, delimitacion mas fina

#veo cuando B supera a Bd en 3 sigmas en el entorno del fin del overshoot
fover_Bx = (np.abs( Bx[f_over_eye-5:f_over_eye+5] - (Bd[0] + 3*std_Bd[0]) )).argmin()
fover_By = (np.abs( By[f_over_eye-5:f_over_eye+5] - (Bd[1] + 3*std_Bd[1]) )).argmin()
fover_Bz = (np.abs( Bz[f_over_eye-5:f_over_eye+5] - (Bd[2] + 3*std_Bd[2]) )).argmin()
fover_B = (np.abs( B[f_over_eye-5:f_over_eye+5] - (norm_Bd + 3*std_norm_Bd) )).argmin()





#%%

plt.figure(0, tight_layout = True)
plt.title('Delimitacion de subestructuras', fontsize = 20)

plt.plot(t_mag, B, linewidth = 2)
#regiones up/down
plt.axvspan(xmin = t_mag[i_u], xmax = t_mag[f_u], facecolor = 'r', alpha = 0.15, label = 'Upstream')
plt.axvspan(xmin = t_mag[i_d], xmax = t_mag[f_d], facecolor = 'y', alpha = 0.15, label = 'Downstream')
#asintotas de Bu y Bd
plt.axhline(y = np.linalg.norm(Bu), linewidth = 2, color = 'r')
plt.axhline(y = np.linalg.norm(Bd), linewidth = 2, color = 'r')
#incio foot
plt.axvline(x = t_ifoot_ext, linewidth = 2, linestyle = '--', color = 'k')
plt.axvline(x = t_ifoot_int, linewidth = 2, linestyle = '--', color = 'k')

plt.xlabel('Tiempo [hora decimal]', fontsize = 20)
plt.ylabel(r'$|\vec{B}|$ [nT]', fontsize = 20)
plt.xlim(8.5,14)
plt.ylim(0,40)
plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
plt.grid(axis = 'both', which = 'both', alpha = 0.8, linewidth = 2, linestyle = '--')
plt.legend(loc = 0, fontsize = 15)

