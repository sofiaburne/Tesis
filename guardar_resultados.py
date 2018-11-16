# GUARDO RESULTADOS DE analisisMHD.py

import numpy as np
import analisisMHD

#%%

# parametros que cambio a mano, delimitacion del shock

datos1 = np.array([])

#limites t_mag apoapsides
datos1[0,0] = t_apo11
datos1[0,1] = t_apo12
datos1[0,2] = t_apo21
datos1[0,3] = t_apo22

#limites t_mag regiones up/downstream
datos1[1,0] = t_id
datos1[1,1] = t_fd
datos1[1,2] = t_iu

#limites t_mag ancho temporal del shock
datos1[2,0] = t_ancho_temp1
datos1[2,1] = t_ancho_temp2
#t_mag centro del shock
datos1[3,0] = tc

#limites t_mag extremos para encontrar regiones up/downstream
datos1[4,0] = lim_t1u
datos1[4,1] = lim_t2u
datos1[4,2] = lim_t1d
datos1[4,3] = lim_t2d

np.savetxt(path_analisis+'parametros_shock_amano_{}'.format(shock_date), datos1, delimiter = '\t',
           header = '\n'.join(['{}'.format(date),'limites apoapsides',
                                                 'limites regiones up/dowstream',
                                                 'limites ancho temporal shock',
                                                 'tiempo centro shock',
                                                 'extremos regiones up/downstream']))


# caracteristicas generales del shock

datos2 = np.array([])

#posisicon de la nave en el centro del shock
datos2[0,:] = Rc

#vel de la nave (xyz) y su norma
datos2[1,:] = v_nave
datos2[1,3] = norm_v_nave

#ancho temporal del shock
datos2[2,0] = ancho_shock_temp

#ancho espacial del shock y su modulo
datos2[3,:] = ancho_shock
datos2[3,3] = norm_ancho_shock

np.savetxt(path_analisis+'caracteristicas_generales_shock_{}'.format(shock_date), datos2, delimiter = '\t',
           header = '\n'.join(['{}'.format(date),'(x,y,z) nave en el centro del shock',
                                                 'vel nave (x,y,z) y su norma',
                                                 'ancho temporal shock',
                                                 'ancho espacial shock (x,y,z) y su norma']))


# coplanaridad para un sample

datos3 = np.array([])

#Bu y su devstd
datos3[0,:] = Bu
datos3[1,:] = std_Bu
#Bd y su desvstd
datos3[2,:] = Bd
datos3[3,:] = std_Bd
#Vu y su desvstd
datos3[4,:] = Vu
#datos3[5,:] = std_Vu
#Vd y su desvstd
datos3[6,:] = Vd
#datos3[7,:] = std_Vd

#normales nB, nBuV, nBdV, nBduV, nV
datos3[8,:] = nB 
datos3[9,:] = nBuV
datos3[10,:] = nBdV
datos3[11,:] = nBduV
datos3[12,:] = nV

#angulos entre normales y Bu
datos3[13,0] = thetaB 
datos3[13,1] = thetaBuV
datos3[13,2] = thetaBdV
datos3[13,3] = thetaBduV
datos3[13,4] = thetaV

#angulos entre normales y Rc
datos3[14,0] = thetaB_Rc 
datos3[14,1] = thetaBuV_Rc
datos3[14,2] = thetaBdV_Rc
datos3[14,3] = thetaBduV_Rc
datos3[14,4] = thetaV_Rc

np.savetxt(path_analisis+'complanaridad_1sample_shock_{}'.format(shock_date), datos3, delimiter = '\t',
           header = '\n'.join(['{}'.format(date),'Bu',
                                                 'desvstd Bu',
                                                 'Bd',
                                                 'desvstd Bd',
                                                 'Vu',
                                                 'desvstd Vu',
                                                 'Vd',
                                                 'desvstd Vd',
                                                 'nB',
                                                 'nBuV',
                                                 'nBdV',
                                                 'nBduV',
                                                 'nV',
                                                 'angulos entre normales y Bu',
                                                 'angulos entre normales y Rc']))


# bootstrap coplanaridad para un sample

datos4 = np.array([])

#cantidad de samples
datos4[0,0] = Ns

#Bu medio entre todos los samples y su desvstd
datos4[1,:] = av_Bu_boot
datos4[2,:] = std_Bu_boot
#Bd medio entre todos los samples y su desvstd
datos4[3,:] = av_Bd_boot
datos4[4,:] = std_Bd_boot
#Vu medio entre todos los samples y su desvstd
datos4[5,:] = av_Vu_boot
datos4[6,:] = std_Vu_boot
#Vd medio entre todos los samples y su desvstd
datos4[7,:] = av_Vd_boot
datos4[8,:] = std_Vd_boot

#nB media entre todos los samples y sudesvstd
datos4[9,:] = av_nB_boot
datos4[10,:] = std_nB_boot
#nBuV media entre todos los samples y sudesvstd
datos4[11,:] = av_nBuV_boot
datos4[12,:] = std_nBuV_boot
#nBdV media entre todos los samples y sudesvstd
datos4[13,:] = av_nBdV_boot
datos4[14,:] = std_nBdV_boot
#nBduV media entre todos los samples y sudesvstd
datos4[15,:] = av_nBduV_boot
datos4[16,:] = std_nBduV_boot
#nV media entre todos los samples y sudesvstd
datos4[17,:] = av_nV_boot
datos4[18,:] = std_nV_boot

#angulo entre nB y Bu medio entre todos los samples y sudesvstd
datos4[19,0] = av_thetaB_boot
datos4[19,1] = std_thetaB_boot
#angulo entre nBuV y Bu medio entre todos los samples y sudesvstd
datos4[20,0] = av_thetaBuV_boot
datos4[20,1] = std_thetaBuV_boot
#angulo entre nBdV y Bu medio entre todos los samples y sudesvstd
datos4[21,0] = av_thetaBdV_boot
datos4[21,1] = std_thetaBdV_boot
#angulo entre nBduV y Bu media entre todos los samples y sudesvstd
datos4[22,0] = av_thetaBduV_boot
datos4[22,1] = std_thetaBduV_boot
#angulo entre nV y Bu media entre todos los samples y sudesvstd
datos4[23,0] = av_thetaV_boot
datos4[23,1] = std_thetaV_boot

#angulo entre nB y Rc medio entre todos los samples y sudesvstd
datos4[24,0] = av_thetaB_Rc_boot
datos4[24,1] = std_thetaB_Rc_boot
#angulo entre nBuV y Rc medio entre todos los samples y sudesvstd
datos4[25,0] = av_thetaBuV_Rc_boot
datos4[25,1] = std_thetaBuV_Rc_boot
#angulo entre nBdV y Rc medio entre todos los samples y sudesvstd
datos4[26,0] = av_thetaBdV_Rc_boot
datos4[26,1] = std_thetaBdV_Rc_boot
#angulo entre nBduV y Rc media entre todos los samples y sudesvstd
datos4[27,0] = av_thetaBduV_Rc_boot
datos4[27,1] = std_thetaBduV_Rc_boot
#angulo entre nV y Rc media entre todos los samples y sudesvstd
datos4[28,0] = av_thetaV_Rc_boot
datos4[28,1] = std_thetaV_Rc_boot

np.savetxt(path_analisis+'complanaridad_boot_shock_{}'.format(shock_date), datos4, delimiter = '\t',
           header = '\n'.join(['{}'.format(date),'cantidad de samples',
                                                 'Bu medio',
                                                 'desvstd Bu',
                                                 'Bd medio',
                                                 'desvstd Bd',
                                                 'Vu medio',
                                                 'desvstd Vu',
                                                 'Vd medio',
                                                 'desvstd Vd',
                                                 'normal media nB',
                                                 'desvstd nB',
                                                 'normal media nBuV',
                                                 'desvstd nBuV',
                                                 'normal media nBdV',
                                                 'desvstd nBdV',
                                                 'normal media nBduV',
                                                 'desvstd nBduV',
                                                 'normal media nV',
                                                 'desvstd nV',
                                                 'angulo medio entre Bu y nB y su desvstd',
                                                 'angulo medio entre Bu y nBuV y su desvstd',
                                                 'angulo medio entre Bu y nBdV y su desvstd',
                                                 'angulo medio entre Bu y nBduV y su desvstd',
                                                 'angulo medio entre Bu y nV y su desvstd',
                                                 'angulo medio entre Rc y nB y su desvstd',
                                                 'angulo medio entre Rc y nBuV y su desvstd',
                                                 'angulo medio entre Rc y nBdV y su desvstd',
                                                 'angulo medio entre Rc y nBduV y su desvstd',
                                                 'angulo medio entre Rc y nV y su desvstd']))


# coplanaridad variando intervalos up/downstream
    
datos5 = np.array([])

#cantidad de intervalos para Bu/Bd y Vu/Vd que puedo tomar 
datos5[0,0] = Lu
datos5[0,1] = Ld
datos5[0,2] = Luv
datos5[0,3] = Ldv

#Bu medio entre todos los samples y su desvstd, Bd medio y su desvstd
datos5[1,:] = av_Bu_s
datos5[2,:] = std_Bu_s
datos5[3,:] = av_Bd_s
datos5[4,:] = std_Bd_s
#Vu medio entre todos los samples y su desvstd, Vd medio y su desvstd
datos5[5,:] = av_Vu_s
datos5[6,:] = std_Vu_s
datos5[7,:] = av_Vd_s
datos5[8,:] = std_Vd_s

#variando ambos intervalos

#nB medio entre todo los samples y su desvstd
datos5[9,:] = av_nB_s2
datos5[10,:] = std_nB_s2
#nBuV medio entre todo los samples y su desvstd
datos5[11,:] = av_nBuV_s2
datos5[12,:] = std_nBuV_s2
#nBdV medio entre todo los samples y su desvstd
datos5[13,:] = av_nBdV_s2
datos5[14,:] = std_nBdV_s2
#nBduV medio entre todo los samples y su desvstd
datos5[15,:] = av_nBduV_s2
datos5[16,:] = std_nBduV_s2
#nV medio entre todo los samples y su desvstd
datos5[17,:] = av_nV_s2
datos5[18,:] = std_nV_s2

#angulo entre Bu y nB medio entre todo los samples y su desvstd
datos5[19,0] = av_thetaB_s2
datos5[19,1] = std_thetaB_s2
#angulo entre Bu y nBuV medio entre todo los samples y su desvstd
datos5[20,0] = av_thetaBuV_s2
datos5[20,1] = std_thetaBuV_s2
#angulo entre Bu y nBdV medio entre todo los samples y su desvstd
datos5[21,0] = av_thetaBdV_s2
datos5[21,1] = std_thetaBdV_s2
#angulo entre Bu y nBduV medio entre todo los samples y su desvstd
datos5[22,0] = av_thetaBduV_s2
datos5[22,1] = std_thetaBduV_s2
#angulo entre Bu y nV medio entre todo los samples y su desvstd
datos5[23,0] = av_thetaV_s2
datos5[23,1] = std_thetaV_s2

#conos de error
datos5[24,0] = cono_err_nB
datos5[24,1] = cono_err_nBuV
datos5[24,2] = cono_err_nBdV
datos5[24,3] = cono_err_nBduV
datos5[24,4] = cono_err_nV

#variando intervalo upstream

#nB medio entre todo los samples y su desvstd
datos5[25,:] = av_nB_su
datos5[26,:] = std_nB_su
#nBuV medio entre todo los samples y su desvstd
datos5[27,:] = av_nBuV_su
datos5[28,:] = std_nBuV_su
#nBdV medio entre todo los samples y su desvstd
datos5[29,:] = av_nBdV_su
datos5[30,:] = std_nBdV_su
#nBduV medio entre todo los samples y su desvstd
datos5[31,:] = av_nBduV_su
datos5[32,:] = std_nBduV_su
#nV medio entre todo los samples y su desvstd
datos5[33,:] = av_nV_su
datos5[34,:] = std_nV_su

#angulo entre Bu y nB medio entre todo los samples y su desvstd
datos5[35,0] = av_thetaB_su
datos5[35,1] = std_thetaB_su
#angulo entre Bu y nBuV medio entre todo los samples y su desvstd
datos5[36,0] = av_thetaBuV_su
datos5[36,1] = std_thetaBuV_su
#angulo entre Bu y nBdV medio entre todo los samples y su desvstd
datos5[37,0] = av_thetaBdV_su
datos5[37,1] = std_thetaBdV_su
#angulo entre Bu y nBduV medio entre todo los samples y su desvstd
datos5[38,0] = av_thetaBduV_su
datos5[38,1] = std_thetaBduV_su
#angulo entre Bu y nV medio entre todo los samples y su desvstd
datos5[39,0] = av_thetaV_su
datos5[39,1] = std_thetaV_su

#variando intervalo downstream

#nB medio entre todo los samples y su desvstd
datos5[40,:] = av_nB_sd
datos5[41,:] = std_nB_sd
#nBuV medio entre todo los samples y su desvstd
datos5[42,:] = av_nBuV_sd
datos5[43,:] = std_nBuV_sd
#nBdV medio entre todo los samples y su desvstd
datos5[44,:] = av_nBdV_sd
datos5[45,:] = std_nBdV_sd
#nBduV medio entre todo los samples y su desvstd
datos5[46,:] = av_nBduV_sd
datos5[47,:] = std_nBduV_sd
#nV medio entre todo los samples y su desvstd
datos5[48,:] = av_nV_sd
datos5[49,:] = std_nV_sd

#angulo entre Bu y nB medio entre todo los samples y su desvstd
datos5[50,0] = av_thetaB_sd
datos5[50,1] = std_thetaB_sd
#angulo entre Bu y nBuV medio entre todo los samples y su desvstd
datos5[51,0] = av_thetaBuV_sd
datos5[51,1] = std_thetaBuV_sd
#angulo entre Bu y nBdV medio entre todo los samples y su desvstd
datos5[52,0] = av_thetaBdV_sd
datos5[52,1] = std_thetaBdV_sd
#angulo entre Bu y nBduV medio entre todo los samples y su desvstd
datos5[53,0] = av_thetaBduV_sd
datos5[53,1] = std_thetaBduV_sd
#angulo entre Bu y nV medio entre todo los samples y su desvstd
datos5[54,0] = av_thetaV_sd
datos5[54,1] = std_thetaV_sd

np.savetxt(path_analisis+'complanaridad_variacion_intervalos_shock_{}'.format(shock_date), datos5, delimiter = '\t',
           header = '\n'.join(['{}'.format(date),'cant samples Bu Bd Vu Vd',
                                                 'Bu medio', 'std Bu', 'Bd medio', 'std Bd',
                                                 'Vu medio', 'std Vu', 'Vd medio', 'std Vd',
                                                 'VARIANDO UP/DOWN',
                                                 'nB medio', 'std nB',
                                                 'nBuV medio', 'std nBuV',
                                                 'nBdV medio', 'std nBdV',
                                                 'nBduV medio', 'std nBduV',
                                                 'nV medio', 'std nV',
                                                 'angulo Bu y nB medio std',
                                                 'angulo Bu y nBuV medio std',
                                                 'angulo Bu y nBdV medio std',
                                                 'angulo Bu y nduV medio std',
                                                 'angulo Bu y nV medio std'
                                                 'cono error nB nBuV nBdV nBduV nV',
                                                 'VARIANDO UP',
                                                 'nB medio', 'std nB',
                                                 'nBuV medio', 'std nBuV',
                                                 'nBdV medio', 'std nBdV',
                                                 'nBduV medio', 'std nBduV',
                                                 'nV medio', 'std nV',
                                                 'angulo Bu y nB medio std',
                                                 'angulo Bu y nBuV medio std',
                                                 'angulo Bu y nBdV medio std',
                                                 'angulo Bu y nduV medio std',
                                                 'angulo Bu y nV medio std'
                                                 'VARIANDO DOWN',
                                                 'nB medio', 'std nB',
                                                 'nBuV medio', 'std nBuV',
                                                 'nBdV medio', 'std nBdV',
                                                 'nBduV medio', 'std nBduV',
                                                 'nV medio', 'std nV',
                                                 'angulo Bu y nB medio std',
                                                 'angulo Bu y nBuV medio std',
                                                 'angulo Bu y nBdV medio std',
                                                 'angulo Bu y nduV medio std',
                                                 'angulo Bu y nV medio std']))


#testeo hipotesis MHD

datos6 = np.array([])

#normal de referencia
datos6[0,:] = norm

#Tu Td Pu Pd
datos6[1,0] = Tu
datos6[1,1] = Td
datos6[1,2] = Pu
datos6[1,3] = Pd

#numeros de Mach
datos6[2,0] = M_A
datos6[2,1] = M_cs
datos6[2,2] = M_f
datos6[2,3] = M_c

#conservaciones
datos6[3,0] = cons_masa
datos6[3,1] = cons_impul_t
datos6[3,2] = cons_energ
datos6[3,3] = cons_Bn
datos6[3,4] = cons_Et

#hipotesis teo coplanaridad
datos6[4,0] = hipt_copl_B

np.savetxt(path_analisis+'hipotesis_MHD_shock_{}'.format(shock_date), datos6, delimiter = '\t',
           header = '\n'.join(['{}'.format(date),'normal de ref usada en calculos',
                                                 'Tu Td Pu Pd',
                                                 'M_Alfv M_sonico M_rapido M_critico',
                                                 'conservaciones: masa, impulso tang, energia, Bn, campo elect tang',
                                                 'hipotesis teo coplanaridad']))


#comparacion con fir bowshock macro

datos7 = np.array([])

#normal del fit
datos7[0,:] = N
#angulo entre Bu y normal fit
datos7[1,0] = theta_N
#angulo entre Rc y normal fit
datos7[2,0] = theta_NRc

#angulos entre normal fit y normales coplanares
datos7[3,0] = angulo_normsB
datos7[3,1] = angulo_normsBuV
datos7[3,2] = angulo_normsBdV
datos7[3,3] = angulo_normsBduV
datos7[3,4] = angulo_normsV

#diferencia entre angulo Bu norm fit y Bu norm coplanares
datos7[4,0] = dif_thetasB
datos7[4,1] = dif_thetasBuV
datos7[4,2] = dif_thetasBdV
datos7[4,3] = dif_thetasBduV
datos7[4,4] = dif_thetasV

np.savetxt(path_analisis+'comparacion_fit_shock_{}'.format(shock_date), datos7, delimiter = '\t',
           header = '\n'.join(['{}'.format(date),'normal del fit',
                                                 'angulo Bu y norm fit',
                                                 'angulo Rc y norm fit',
                                                 'angulo entre norm fit y nB, nBuV, nBdV, nBduV, nV',
                                                 'diferencia entre angulo Bu y norm fit y Bu y nB, nBuV, nBdV, nBduV, nVd']))
