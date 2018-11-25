#from mag import B, Bx, By, Bz, t_mag, shock_date
#
#from importlib import reload
#import numpy as np
#import matplolib.pyplot as plt
#import os
#
#path_analisis = r'C:\Users\sofia\Documents\Facultad\Tesis\Analisis/{}/'.format(shock_date)
#if not os.path.exists(path_analisis):
#    os.makedirs(path_analisis)

##%%
#
##me quedo con los datos en la orbita del shock
#    
#
##estos valores los saco de 'caracteristicas_generales_shock_{}.txt'
#Ti_orbit = 8.82096825600047     #*
#Tf_orbit = 13.401525767999374   #*
#
##busco valor mas cercano porque ahora t_mag es de alta frec
#i_orbit = (np.abs(t_mag - Ti_orbit)).argmin()
#f_orbit = (np.abs(t_mag - Tf_orbit)).argmin()
#
#B = B[i_orbit:f_orbit]
#Bx = Bx[i_orbit:f_orbit]
#By = By[i_orbit:f_orbit]
#Bz = Bz[i_orbit:f_orbit]
#t_mag = t_mag[i_orbit:f_orbit]
#
#
##importo Bu, Bd, sus modulos y desviaciones estandar
#Bu = 
#Bd = 
#norm_Bu = 
#norm_Bd = 
#std_Bu = 
#std_Bd = 
#std_norm_Bu = 
#std_norm_Bd = 
#


#%%

'''
Busca los tiempos en los que Bx,By,Bz y B superan el valor de Bu (o Bd) en 3 veces
su dispersion. Luego toma como limite a izquierda el menor de esos tiempos, y como
limite a derecha el mayor.
'''

def lim_int_ext(Bx, By, Bz, B, Bu, norm_Bu, std_Bu, std_norm_Bu, inicio, fin):

    index_Bx = (np.abs(Bx[inicio:fin] - (Bu[0] + 3*std_Bu[0]))).argmin()
    index_By = (np.abs( By[inicio:fin] - (Bu[1] + 3*std_Bu[1]))).argmin()
    index_Bz = (np.abs(Bz[inicio:fin] - (Bu[2] + 3*std_Bu[2]))).argmin()
    index_B = (np.abs(B[inicio:fin] - (norm_Bu + 3*std_norm_Bu))).argmin()
    
    t_izq = np.min([t_mag[inicio+index_Bx], t_mag[inicio+index_By], t_mag[inicio+index_Bz], t_mag[inicio+index_B]])
    t_der = np.max([t_mag[inicio+index_Bx], t_mag[inicio+index_By], t_mag[inicio+index_Bz], t_mag[inicio+index_B]])
    index_izq = list(t_mag).index(t_izq)
    index_der = list(t_mag).index(t_der)
    
    return t_izq, t_der, index_izq, index_der

#%%

#FOOT

#primero marco limites a ojo que delimiten donde buscar el foot
Ti_foot_eye = 9.75 #*
Tf_foot_eye = 9.8 #*
i_foot_eye = (np.abs(t_mag - Ti_foot_eye)).argmin()
f_foot_eye = (np.abs(t_mag - Tf_foot_eye)).argmin()


#inicio foot, delimitacion mas fina
ti_foot_izq, ti_foot_der, i_foot_izq, i_foot_der = lim_int_ext(Bx, By, Bz, B, Bu, norm_Bu, std_Bu, std_norm_Bu, i_foot_eye-100, i_foot_eye+100)


#%%

#RAMP

#primero marco limites a ojo que delimiten donde buscar la ramp
Ti_ramp_eye = 9.75 #*
Tf_ramp_eye = 9.8 #*
i_ramp_eye = (np.abs(t_mag - Ti_ramp_eye)).argmin() 
f_ramp_eye = (np.abs(t_mag - Tf_ramp_eye)).argmin() 


#fin de la rampa, delimitacion mas fina

tf_ramp_izq, tf_ramp_der, f_ramp_izq, f_ramp_der = lim_int_ext(Bx, By, Bz, B, Bd, norm_Bd, std_Bd, std_norm_Bd, f_ramp_eye-100, f_ramp_eye+100)

#fit lineal del intervalo tomando lim izq y der, me quedo con el que tiene menor cov
#   !!!  no entiendo para que te sirve el fit si los limites igual van a ser los mismos
params_der, cov_der = np.polyfit(t_mag[i_ramp_eye:f_ramp_der], B[i_ramp_eye:f_ramp_der], 1, cov = True)
params_izq, cov_izq = np.polyfit(t_mag[i_ramp_eye:f_ramp_izq], B[i_ramp_eye:f_ramp_izq], 1, cov = True)

#errores relativos de los parametros del ajuste
err_m_der = cov_der[0,0]/params_der[0]
err_b_der = cov_der[1,1]/params_der[1]
err_m_izq = cov_izq[0,0]/params_izq[0]
err_b_izq = cov_izq[1,1]/params_izq[1]


#me quedo con el fit de menor error
if err_m_der > err_b_der:
    err_fit_der = err_m_der
else:
    err_fit_der = err_b_der

if err_m_izq > err_b_izq:
    err_fit_izq = err_m_izq
else:
    err_fit_izq = err_b_izq

if err_fit_der > err_fit_izq:
    print('mejor fit con lim izq')
    best_params = params_izq
    best_cov = cov_izq
else:
    print('mejor fit con lim der')    
    best_params = params_der
    best_cov = cov_der


#%%

#OVERSHOOT

#primero marco limites a ojo que delimiten donde buscar el overshoot
Ti_over_eye = 9.75 #*
Tf_over_eye = 9.8 #*
i_over_eye = (np.abs(t_mag - Ti_over_eye)).argmin() 
f_over_eye = (np.abs(t_mag - Tf_over_eye)).argmin() 


#inicio del overshoot, delimitacion mas fina
ti_over_izq, ti_over_der, i_over_izq, i_over_der = lim_int_ext(Bx, By, Bz, B, Bd, norm_Bd, std_Bd, std_norm_Bd, i_over_eye-100, i_over_eye+100)


#fin del overshoot, delimitacion mas fina
tf_over_izq, tf_over_der, f_over_izq, f_over_der = lim_int_ext(Bx, By, Bz, B, Bd, norm_Bd, std_Bd, std_norm_Bd, f_over_eye-100, f_over_eye+100)


#%%

plt.figure(0, tight_layout = True)
plt.title('Delimitacion de subestructuras', fontsize = 20)

plt.plot(t_mag, B, linewidth = 2)
#regiones up/down
plt.axvspan(xmin = t_mag[i_u], xmax = t_mag[f_u], facecolor = 'r', alpha = 0.15, label = 'Upstream')
plt.axvspan(xmin = t_mag[i_d], xmax = t_mag[f_d], facecolor = 'y', alpha = 0.15, label = 'Downstream')
#asintotas de Bu y Bd
plt.axhline(y = norm_Bu, linewidth = 2, color = 'r')
plt.axhline(y = norm_Bd, linewidth = 2, color = 'r')
#delimitacion foot
plt.axvline(x = ti_foot_izq, linewidth = 1, linestyle = '--', color = 'g')
plt.axvline(x = ti_foot_der, linewidth = 1, linestyle = '--', color = 'g')
plt.axvline(x = Tf_foot_eye, linewidth = 1, linestyle = '--', color = 'g')
#delimitacion ramp
plt.axvline(x = Ti_ramp_eye, linewidth = 1, linestyle = '--', color = 'k')
plt.axvline(x = tf_ramp_izq, linewidth = 1, linestyle = '--', color = 'k')
plt.axvline(x = tf_ramp_der, linewidth = 1, linestyle = '--', color = 'k')
#delimitacion overshoot
plt.axvline(x = ti_over_izq, linewidth = 1, linestyle = '--', color = 'm')
plt.axvline(x = ti_over_der, linewidth = 1, linestyle = '--', color = 'm')
plt.axvline(x = tf_over_izq, linewidth = 1, linestyle = '--', color = 'm')
plt.axvline(x = tf_over_der, linewidth = 1, linestyle = '--', color = 'm')

plt.xlabel('Tiempo [hora decimal]', fontsize = 20)
plt.ylabel(r'$|\vec{B}|$ [nT]', fontsize = 20)
plt.xlim(8.5,14)
plt.ylim(0,40)
plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
plt.grid(axis = 'both', which = 'both', alpha = 0.8, linewidth = 2, linestyle = '--')
plt.legend(loc = 0, fontsize = 15)
