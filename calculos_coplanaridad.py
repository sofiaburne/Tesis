
from mag import shock_date
from delimitacion_shock import B, Bx, By, Bz, t_mag
from delimitacion_shock import t_swea, flujosenergia_swea, nivelesenergia_swea
from delimitacion_shock import t_swia_mom, t_swia_spec, densidad_swia, velocidad_swia, velocidad_swia_norm, temperatura_swia, temperatura_swia_norm, flujosenergia_swia, nivelesenergia_swia
from delimitacion_shock import B1, B2, V1, V2, Bd, Bu, Vd, Vu, i_u, f_u, i_d, f_d, lim_t1u, lim_t2u, lim_t1d, lim_t2d, i_u5, f_u5, i_d5, f_d5, iu_v5, fu_v5, id_v5, fd_v5, vel, v_nave
from analisis_subestructuras import N, theta_N, Rc
from testeo_hipotesisMHD import M_A, M_c
import funciones_coplanaridad as fcop


from importlib import reload
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import statistics as st
import os


path_analisis = r'C:\Users\sofia\Documents\Facultad\Tesis\Analisis/{}/'.format(shock_date)
if not os.path.exists(path_analisis):
    os.makedirs(path_analisis)


#%%---------------------------- COPLANARIDAD PARA UN INTERVALO UP/DOWN DADO ----------------------------
#me falta propagar errores normal y angulos &&


#hay 5 normales que se pueden calcular a partir de coplanaridad
nB, nBuV, nBdV, nBduV, nV = fcop.norm_coplanar(Bd,Bu,Vd,Vu)


#flags: los angulos con Bu cercanos a 0 y 90 son problematicos (uso normal del fit como ref)
#puedo que esto lo tenga que reescribir (por si estan rotados los vectores en 180) &&
if theta_N in range(85,95):
    print('nB no es buena', 'nV es buena')
elif (theta_N in range(0,5)) and (M_A > M_c):
    print('nB no es buena', 'nV es buena')

if nB[0] <= 0.007: #pongo este valor de ref porque con nB_x ~0.0058 pasaron cosas raras (shock 2016-03-19)
    print('nB_x muy chica')


#angulos con campo upstream
thetaB = fcop.alpha(Bu,nB)
thetaBuV = fcop.alpha(Bu,nBuV)
thetaBdV = fcop.alpha(Bu,nBdV)
thetaBduV = fcop.alpha(Bu,nBduV)
thetaV = fcop.alpha(Bu,nV)

#angulos con vector posicion de la nave en centro del shock    
thetaB_Rc = fcop.alpha(nB,Rc)
thetaBuV_Rc = fcop.alpha(nBuV,Rc)
thetaBdV_Rc = fcop.alpha(nBdV,Rc)
thetaBduV_Rc = fcop.alpha(nBduV,Rc)
thetaV_Rc = fcop.alpha(nV,Rc)

#angulos entre Vu y normales
thetaB_Vu = fcop.alpha(Vu,nB)
thetaBuV_Vu = fcop.alpha(Vu,nBuV)
thetaBdV_Vu = fcop.alpha(Vu,nBdV)
thetaBduV_Vu = fcop.alpha(Vu,nBduV)
thetaV_Vu = fcop.alpha(Vu,nV)

#angulos entre vel nave y normales
thetaB_vnave = fcop.alpha(v_nave,nB)
thetaBuV_vnave = fcop.alpha(v_nave,nBuV)
thetaBdV_vnave = fcop.alpha(v_nave,nBdV)
thetaBduV_vnave = fcop.alpha(v_nave,nBduV)
thetaV_vnave = fcop.alpha(v_nave,nV)

#%% analisis de Bn

Bn_B = np.dot(np.array([Bx,By,Bz]), nB)
Bn_BuV = np.dot(np.array([Bx,By,Bz]), nBuV)
Bn_BdV = np.dot(np.array([Bx,By,Bz]), nBdV)
Bn_BduV = np.dot(np.array([Bx,By,Bz]), nBduV)
Bn_V = np.dot(np.array([Bx,By,Bz]), nV)


plt.figure(100, figsize = (30,20))
plt.suptitle(r'Componente normal del campo magnético', fontsize = 30)

plot0 = plt.subplot(511)
plt.title('n1', fontsize = 20)
plt.plot(t_mag, Bn_B, linewidth = 3, color = 'C0')
plt.ylabel(r'$B_n$ [nT]', fontsize = 20)
plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
plt.grid(which = 'both', axis = 'both', linewidth = 2, linestyle = '--', alpha = 0.8)

plt.subplot(512, sharex = plot0)
plt.title('n2', fontsize = 20)
plt.plot(t_mag, Bn_BuV, linewidth = 3, color = 'C1')
plt.ylabel(r'$B_n$ [nT]', fontsize = 20)
plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
plt.grid(which = 'both', axis = 'both', linewidth = 2, linestyle = '--', alpha = 0.8)

plt.subplot(513, sharex = plot0)
plt.title('n3', fontsize = 20)
plt.plot(t_mag, Bn_BdV, linewidth = 3, color = 'C1')
plt.ylabel(r'$B_n$ [nT]', fontsize = 20)
plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
plt.grid(which = 'both', axis = 'both', linewidth = 2, linestyle = '--', alpha = 0.8)

plt.subplot(514, sharex = plot0)
plt.title('n4', fontsize = 20)
plt.plot(t_mag, Bn_BduV, linewidth = 3, color = 'C1')
plt.ylabel(r'$B_n$ [nT]', fontsize = 20)
plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
plt.grid(which = 'both', axis = 'both', linewidth = 2, linestyle = '--', alpha = 0.8)

plt.subplot(515, sharex = plot0)
plt.title('n5', fontsize = 20)
plt.plot(t_mag, Bn_V, linewidth = 3, color = 'C1')
plt.ylabel(r'$B_n$ [nT]', fontsize = 20)
plt.xlabel(r'Tiempo [hora decimal]', fontsize = 20)
plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
plt.grid(which = 'both', axis = 'both', linewidth = 2, linestyle = '--', alpha = 0.8)


#%%

#analizo que tan estables son las normales repitiendo los calculos con una y otra mitad de los intervalos

half_u = int(np.abs(f_u - i_u)/2) + i_u
half_d = int(np.abs(f_d - i_d)/2) + i_d

half1_Bu = np.mean(B1[i_u:half_u,:], axis = 0)
half1_Bd = np.mean(B2[i_d:half_d,:], axis = 0)
std_half1_Bu = np.array([st.stdev(B1[i_u:half_u,0]), st.stdev(B1[i_u:half_u,1]), st.stdev(B1[i_u:half_u,2])])
std_half1_Bd = np.array([st.stdev(B2[i_d:half_d,0]), st.stdev(B2[i_d:half_d,1]), st.stdev(B2[i_d:half_d,2])])

half1_Vu = np.mean(V1[i_u:half_u,:], axis = 0)
half1_Vd = np.mean(V2[i_d:half_d,:], axis = 0)
std_half1_Vu = np.array([st.stdev(V1[i_u:half_u,0]), st.stdev(V1[i_u:half_u,1]), st.stdev(V1[i_u:half_u,2])])
std_half1_Vd = np.array([st.stdev(V2[i_d:half_d,0]), st.stdev(V2[i_d:half_d,1]), st.stdev(V2[i_d:half_d,2])])

half2_Bu = np.mean(B1[half_u:f_u,:], axis = 0)
half2_Bd = np.mean(B2[half_d:f_d,:], axis = 0)
std_half2_Bu = np.array([st.stdev(B1[half_u:f_u,0]), st.stdev(B1[half_u:f_u,1]), st.stdev(B1[half_u:f_u,2])])
std_half2_Bd = np.array([st.stdev(B2[half_d:f_d,0]), st.stdev(B2[half_d:f_d,1]), st.stdev(B2[half_d:f_d,2])])

half2_Vu = np.mean(V1[half_u:f_u,:], axis = 0)
half2_Vd = np.mean(V2[half_d:f_d,:], axis = 0)
std_half2_Vu = np.array([st.stdev(V1[half_u:f_u,0]), st.stdev(V1[half_u:f_u,1]), st.stdev(V1[half_u:f_u,2])])
std_half2_Vd = np.array([st.stdev(V2[half_d:f_d,0]), st.stdev(V2[half_d:f_d,1]), st.stdev(V2[half_d:f_d,2])])


half1_nB, half1_nBuV, half1_nBdV, half1_nBduV, half1_nV = fcop.norm_coplanar(half1_Bd,half1_Bu,half1_Vd,half1_Vu)
half2_nB, half2_nBuV, half2_nBdV, half2_nBduV, half2_nV = fcop.norm_coplanar(half2_Bd,half2_Bu,half2_Vd,half2_Vu)


#compara las normales de cada mitad con la del fit

ang_N_half1_nB = fcop.alpha(half1_nB,N)
ang_N_half1_nBuV = fcop.alpha(half1_nBuV,N)
ang_N_half1_nBdV = fcop.alpha(half1_nBdV,N)
ang_N_half1_nBduV = fcop.alpha(half1_nBduV,N)
ang_N_half1_nV = fcop.alpha(half1_nV,N)

ang_N_half2_nB = fcop.alpha(half2_nB,N)
ang_N_half2_nBuV = fcop.alpha(half2_nBuV,N)
ang_N_half2_nBdV = fcop.alpha(half2_nBdV,N)
ang_N_half2_nBduV = fcop.alpha(half2_nBduV,N)
ang_N_half2_nV = fcop.alpha(half2_nV,N)


if ang_N_half1_nB > 2*ang_N_half2_nB:
    print('mejor nB en segunda mitad')
elif ang_N_half2_nB > 2*ang_N_half1_nB:
    print('mejor nB en primera mitad')

if ang_N_half1_nBuV > 2*ang_N_half2_nBuV:
    print('mejor nBuV en segunda mitad')
elif ang_N_half2_nBuV > 2*ang_N_half1_nBuV:
    print('mejor nBuV en primera mitad')

if ang_N_half1_nBdV > 2*ang_N_half2_nBdV:
    print('mejor nBdV en segunda mitad')
elif ang_N_half2_nBdV > 2*ang_N_half1_nBdV:
    print('mejor nBdV en primera mitad')

if ang_N_half1_nBduV > 2*ang_N_half2_nBduV:
    print('mejor nBduV en segunda mitad')
elif ang_N_half2_nBduV > 2*ang_N_half1_nBduV:
    print('mejor nBduV en primera mitad')

if ang_N_half1_nV > 2*ang_N_half2_nV:
    print('mejor nV en segunda mitad')
elif ang_N_half2_nV > 2*ang_N_half1_nV:
    print('mejor nV en primera mitad')



#%%#####################################################################################################
########################################################################################################
########################################################################################################
########################################################################################################
#%% ----------------------- COPLANARIDAD BOOTSTRAP PARA UN INTERVALO UP/DOWN DADO ----------------------

#calculos con campo magnetico
Ns = 1000
Bu_boot, Bd_boot, Vu_boot, Vd_boot, nB_boot, nBuV_boot, nBdV_boot, nBduV_boot, nV_boot = fcop.copl_boot(B1,B2,V1,V2,Ns)


#angulo entre normales y Bu (para cada bootstrap sample)

thetaB_boot = np.empty([Ns, 1])
thetaBuV_boot = np.empty([Ns, 1])
thetaBdV_boot = np.empty([Ns, 1])
thetaBduV_boot = np.empty([Ns, 1])
thetaV_boot = np.empty([Ns, 1])

for i in range(Ns):
    thetaB_boot[i] = fcop.alpha(Bu_boot[i,:],nB_boot[i,:])
    thetaBuV_boot[i] = fcop.alpha(Bu_boot[i,:],nBuV_boot[i,:])
    thetaBdV_boot[i] = fcop.alpha(Bu_boot[i,:],nBdV_boot[i,:])
    thetaBduV_boot[i] = fcop.alpha(Bu_boot[i,:],nBduV_boot[i,:])
    thetaV_boot[i] = fcop.alpha(Bu_boot[i,:],nV_boot[i,:])    


#angulo entre normales y Rc (para cada bootstrap sample)
    
thetaB_Rc_boot = np.empty([Ns, 1])
thetaBuV_Rc_boot = np.empty([Ns, 1])
thetaBdV_Rc_boot = np.empty([Ns, 1])
thetaBduV_Rc_boot = np.empty([Ns, 1])
thetaV_Rc_boot = np.empty([Ns, 1])

for i in range(len(nB_boot[:,0])):
    thetaB_Rc_boot[i] = fcop.alpha(nB_boot[i,:],Rc)
    thetaBuV_Rc_boot[i] = fcop.alpha(nBuV_boot[i,:],Rc)
    thetaBdV_Rc_boot[i] = fcop.alpha(nBdV_boot[i,:],Rc)
    thetaBduV_Rc_boot[i] = fcop.alpha(nBduV_boot[i,:],Rc)
    thetaV_Rc_boot[i] = fcop.alpha(nV_boot[i,:],Rc)


#valores medios y desviaciones estandar
    
av_Bu_boot = np.mean(Bu_boot, axis = 0)
std_Bu_boot = np.array([st.stdev(Bu_boot[:,0]), st.stdev(Bu_boot[:,1]), st.stdev(Bu_boot[:,2])])
av_Bd_boot = np.mean(Bd_boot, axis = 0)
std_Bd_boot = np.array([st.stdev(Bd_boot[:,0]), st.stdev(Bd_boot[:,1]), st.stdev(Bd_boot[:,2])])
av_Vu_boot = np.mean(Vu_boot, axis = 0)
std_Vu_boot = np.array([st.stdev(Vu_boot[:,0]), st.stdev(Vu_boot[:,1]), st.stdev(Vu_boot[:,2])])
av_Vd_boot = np.mean(Vd_boot, axis = 0)
std_Vd_boot = np.array([st.stdev(Vd_boot[:,0]), st.stdev(Vd_boot[:,1]), st.stdev(Vd_boot[:,2])])

av_nB_boot = np.mean(nB_boot, axis = 0)
std_nB_boot = np.array([st.stdev(nB_boot[:,0]), st.stdev(nB_boot[:,1]), st.stdev(nB_boot[:,2])])
av_nBuV_boot = np.mean(nBuV_boot, axis = 0)
std_nBuV_boot = np.array([st.stdev(nBuV_boot[:,0]), st.stdev(nBuV_boot[:,1]), st.stdev(nBuV_boot[:,2])])
av_nBdV_boot = np.mean(nBdV_boot, axis = 0)
std_nBdV_boot = np.array([st.stdev(nBdV_boot[:,0]), st.stdev(nBdV_boot[:,1]), st.stdev(nBdV_boot[:,2])])
av_nBduV_boot = np.mean(nBduV_boot, axis = 0)
std_nBduV_boot = np.array([st.stdev(nBduV_boot[:,0]), st.stdev(nBduV_boot[:,1]), st.stdev(nBduV_boot[:,2])])
av_nV_boot = np.mean(nV_boot, axis = 0)
std_nV_boot = np.array([st.stdev(nV_boot[:,0]), st.stdev(nV_boot[:,1]), st.stdev(nV_boot[:,2])])

av_thetaB_boot = np.mean(thetaB_boot)
std_thetaB_boot = st.stdev(thetaB_boot[:,0])
av_thetaBuV_boot = np.mean(thetaBuV_boot)
std_thetaBuV_boot = st.stdev(thetaBuV_boot[:,0])
av_thetaBdV_boot = np.mean(thetaBdV_boot)
std_thetaBdV_boot = st.stdev(thetaBdV_boot[:,0])
av_thetaBduV_boot = np.mean(thetaBduV_boot)
std_thetaBduV_boot = st.stdev(thetaBduV_boot[:,0])
av_thetaV_boot = np.mean(thetaV_boot)
std_thetaV_boot = st.stdev(thetaV_boot[:,0])

av_thetaB_Rc_boot = np.mean(thetaB_Rc_boot)
std_thetaB_Rc_boot = st.stdev(thetaB_Rc_boot[:,0])
av_thetaBuV_Rc_boot = np.mean(thetaBuV_Rc_boot)
std_thetaBuV_Rc_boot = st.stdev(thetaBuV_Rc_boot[:,0])
av_thetaBdV_Rc_boot = np.mean(thetaBdV_Rc_boot)
std_thetaBdV_Rc_boot = st.stdev(thetaBdV_Rc_boot[:,0])
av_thetaBduV_Rc_boot = np.mean(thetaBduV_Rc_boot)
std_thetaBduV_Rc_boot = st.stdev(thetaBduV_Rc_boot[:,0])
av_thetaV_Rc_boot = np.mean(thetaV_Rc_boot)
std_thetaV_Rc_boot = st.stdev(thetaV_Rc_boot[:,0])

#%% 

#histogramas para las componentes de las normales

#hist_nB_x, bins_nB_x = np.histogram(nB_boot[:,0], 70)
#hist_nB_y, bins_nB_y = np.histogram(nB_boot[:,1], 70)
#hist_nB_z, bins_nB_z = np.histogram(nB_boot[:,2], 70)
#
#hist_nBuV_x, bins_nBuV_x = np.histogram(nBuV_boot[:,0], 70)
#hist_nBuV_y, bins_nBuV_y = np.histogram(nBuV_boot[:,1], 70)
#hist_nBuV_z, bins_nBuV_z = np.histogram(nBuV_boot[:,2], 70)
#
#hist_nBdV_x, bins_nBdV_x = np.histogram(nBdV_boot[:,0], 70)
#hist_nBdV_y, bins_nBdV_y = np.histogram(nBdV_boot[:,1], 70)
#hist_nBdV_z, bins_nBdV_z = np.histogram(nBdV_boot[:,2], 70)
#
#hist_nBduV_x, bins_nBduV_x = np.histogram(nBduV_boot[:,0], 70)
#hist_nBduV_y, bins_nBduV_y = np.histogram(nBduV_boot[:,1], 70)
#hist_nBduV_z, bins_nBduV_z = np.histogram(nBduV_boot[:,2], 70)
#
#hist_nV_x, bins_nV_x = np.histogram(nV_boot[:,0], 70)
#hist_nV_y, bins_nV_y = np.histogram(nV_boot[:,1], 70)
#hist_nV_z, bins_nV_z = np.histogram(nV_boot[:,2], 70)


plt.figure(4, figsize = (30,20))
plt.suptitle(r'Histogramas normal bootstrap - $n_1 = \frac{(B_d \times B_u) \times \Delta B}{|(B_d \times B_u) \times \Delta B|}$', fontsize = 30)

pB = plt.subplot(131)
#plt.plot(bins_nB_x[:-1], hist_nB_x, linewidth = 3, color = 'C0')
plt.hist(nB_boot[:,0], bins = 70, color = 'C0')
plt.axvline(x = av_nB_boot[0], linewidth = 3, label = 'nx medio', color = 'C1')
plt.xlabel(r'$n_x$', fontsize = 20)
plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
plt.legend(loc = 0, fontsize = 20)
plt.grid(which = 'both', axis = 'both', linewidth = 2, linestyle = '--', alpha = 0.8)

plt.subplot(132, sharey = pB)
#plt.plot(bins_nB_y[:-1], hist_nB_y, linewidth = 3, color = 'C0')
plt.hist(nB_boot[:,1], bins = 70, color = 'C0')
plt.axvline(x = av_nB_boot[1], linewidth = 3, label = 'ny medio', color = 'C1')
plt.xlabel(r'$n_y$', fontsize = 20)
plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
plt.legend(loc = 0, fontsize = 20)
plt.grid(which = 'both', axis = 'both', linewidth = 2, linestyle = '--', alpha = 0.8)

plt.subplot(133, sharey = pB)
#plt.plot(bins_nB_z[:-1], hist_nB_z, linewidth = 3, color = 'C0')
plt.hist(nB_boot[:,2], bins = 70, color = 'C0')
plt.axvline(x = av_nB_boot[2], linewidth = 3, label = 'nz medio', color = 'C1')
plt.xlabel(r'$n_z$', fontsize = 20)
plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
plt.legend(loc = 0, fontsize = 20)
plt.grid(which = 'both', axis = 'both', linewidth = 2, linestyle = '--', alpha = 0.8)

#plt.savefig(path_analisis+'hist_normalB_copl_boot_{}'.format(shock_date))


plt.figure(5, figsize = (30,20))
plt.suptitle(r'Histogramas normal bootstrap - $n_2 = \frac{(B_u \times \Delta V) \times \Delta B}{|(B_u \times \Delta V) \times \Delta B|}$', fontsize = 30)

pBuV = plt.subplot(131)
#plt.plot(bins_nBuV_x[:-1], hist_nBuV_x, linewidth = 3, color = 'C2')
plt.hist(nBuV_boot[:,0], bins = 70, color = 'C2')
plt.axvline(x = av_nBuV_boot[0], linewidth = 3, label = 'nx medio', color = 'C3')
plt.xlabel(r'$n_x$', fontsize = 20)
plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
plt.legend(loc = 0, fontsize = 20)
plt.grid(which = 'both', axis = 'both', linewidth = 2, linestyle = '--', alpha = 0.8)

plt.subplot(132, sharey = pBuV)
#plt.plot(bins_nBuV_y[:-1], hist_nBuV_y, linewidth = 3, color = 'C2')
plt.hist(nBuV_boot[:,1], bins = 70, color = 'C2')
plt.axvline(x = av_nBuV_boot[1], linewidth = 3, label = 'ny medio', color = 'C3')
plt.xlabel(r'$n_y$', fontsize = 20)
plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
plt.legend(loc = 0, fontsize = 20)
plt.grid(which = 'both', axis = 'both', linewidth = 2, linestyle = '--', alpha = 0.8)

plt.subplot(133, sharey = pBuV)
#plt.plot(bins_nBuV_z[:-1], hist_nBuV_z, linewidth = 3, color = 'C2')
plt.hist(nBuV_boot[:,2], bins = 70, color = 'C2')
plt.axvline(x = av_nBuV_boot[2], linewidth = 3, label = 'nz medio', color = 'C3')
plt.xlabel(r'$n_z$', fontsize = 20)
plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
plt.legend(loc = 0, fontsize = 20)
plt.grid(which = 'both', axis = 'both', linewidth = 2, linestyle = '--', alpha = 0.8)

#plt.savefig(path_analisis+'hist_normalBuV_copl_boot_{}'.format(shock_date))


plt.figure(6, figsize = (30,20))
plt.suptitle(r'Histogramas normal bootstrap - $n_3 = \frac{(B_d \times \Delta V) \times \Delta B}{|(B_d \times \Delta V) \times \Delta B|}$', fontsize = 30)

pBdV = plt.subplot(131)
#plt.plot(bins_nBdV_x[:-1], hist_nBdV_x, linewidth = 3, color = 'C4')
plt.hist(nBdV_boot[:,0], bins = 70, color = 'C4')
plt.axvline(x = av_nBdV_boot[0], linewidth = 3, label = 'nx medio', color = 'C5')
plt.xlabel(r'$n_x$', fontsize = 20)
plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
plt.legend(loc = 0, fontsize = 20)
plt.grid(which = 'both', axis = 'both', linewidth = 2, linestyle = '--', alpha = 0.8)

plt.subplot(132, sharey = pBdV)
#plt.plot(bins_nBdV_y[:-1], hist_nBdV_y, linewidth = 3, color = 'C4')
plt.hist(nBdV_boot[:,1], bins = 70, color = 'C4')
plt.axvline(x = av_nBdV_boot[1], linewidth = 3, label = 'ny medio', color = 'C5')
plt.xlabel(r'$n_y$', fontsize = 20)
plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
plt.legend(loc = 0, fontsize = 20)
plt.grid(which = 'both', axis = 'both', linewidth = 2, linestyle = '--', alpha = 0.8)

plt.subplot(133, sharey = pBdV)
#plt.plot(bins_nBdV_z[:-1], hist_nBdV_z, linewidth = 3, color = 'C4')
plt.hist(nBdV_boot[:,2], bins = 70, color = 'C4')
plt.axvline(x = av_nBdV_boot[2], linewidth = 3, label = 'nz medio', color = 'C5')
plt.xlabel(r'$n_z$', fontsize = 20)
plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
plt.legend(loc = 0, fontsize = 20)
plt.grid(which = 'both', axis = 'both', linewidth = 2, linestyle = '--', alpha = 0.8)

#plt.savefig(path_analisis+'hist_normalBdV_copl_boot_{}'.format(shock_date))


plt.figure(7, figsize = (30,20))
plt.suptitle(r'Histogramas normal bootstrap - $n_4 = \frac{(\Delta B \times \Delta V) \times \Delta B}{|(\Delta B \times \Delta V) \times \Delta B|}$', fontsize = 30)

pBduV = plt.subplot(131)
#plt.plot(bins_nBduV_x[:-1], hist_nBduV_x, linewidth = 3, color = 'C6')
plt.hist(nBduV_boot[:,0], bins = 70, color = 'C6')
plt.axvline(x = av_nBduV_boot[0], linewidth = 3, label = 'nx medio', color = 'C7')
plt.xlabel(r'$n_x$', fontsize = 20)
plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
plt.legend(loc = 0, fontsize = 20)
plt.grid(which = 'both', axis = 'both', linewidth = 2, linestyle = '--', alpha = 0.8)

plt.subplot(132, sharey = pBduV)
#plt.plot(bins_nBduV_y[:-1], hist_nBduV_y, linewidth = 3, color = 'C6')
plt.hist(nBduV_boot[:,1], bins = 70, color = 'C6')
plt.axvline(x = av_nBduV_boot[1], linewidth = 3, label = 'ny medio', color = 'C7')
plt.xlabel(r'$n_y$', fontsize = 20)
plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
plt.legend(loc = 0, fontsize = 20)
plt.grid(which = 'both', axis = 'both', linewidth = 2, linestyle = '--', alpha = 0.8)

plt.subplot(133, sharey = pBduV)
#plt.plot(bins_nBduV_z[:-1], hist_nBduV_z, linewidth = 3, color = 'C6')
plt.hist(nBduV_boot[:,2], bins = 70, color = 'C6')
plt.axvline(x = av_nBduV_boot[2], linewidth = 3, label = 'nz medio', color = 'C7')
plt.xlabel(r'$n_z$', fontsize = 20)
plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
plt.legend(loc = 0, fontsize = 20)
plt.grid(which = 'both', axis = 'both', linewidth = 2, linestyle = '--', alpha = 0.8)

#plt.savefig(path_analisis+'hist_normalBduV_copl_boot_{}'.format(shock_date))


plt.figure(8, figsize = (30,20))
plt.suptitle(r'Histogramas normal bootstrap - $n_5 = \frac{V_d - V_u}{|V_d - V_u|}$', fontsize = 30)

pV = plt.subplot(131)
#plt.plot(bins_nV_x[:-1], hist_nV_x, linewidth = 3, color = 'C8')
plt.hist(nV_boot[:,0], bins = 70, color = 'C8')
plt.axvline(x = av_nV_boot[0], linewidth = 3, label = 'nx medio', color = 'C9')
plt.xlabel(r'$n_x$', fontsize = 20)
plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
plt.legend(loc = 0, fontsize = 20)
plt.grid(which = 'both', axis = 'both', linewidth = 2, linestyle = '--', alpha = 0.8)

plt.subplot(132, sharey = pV)
#plt.plot(bins_nV_y[:-1], hist_nV_y, linewidth = 3, color = 'C8')
plt.hist(nV_boot[:,1], bins = 70, color = 'C8')
plt.axvline(x = av_nV_boot[1], linewidth = 3, label = 'ny medio', color = 'C9')
plt.xlabel(r'$n_y$', fontsize = 20)
plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
plt.legend(loc = 0, fontsize = 20)
plt.grid(which = 'both', axis = 'both', linewidth = 2, linestyle = '--', alpha = 0.8)

plt.subplot(133, sharey = pV)
#plt.plot(bins_nV_z[:-1], hist_nV_z, linewidth = 3, color = 'C8')
plt.hist(nV_boot[:,2], bins = 70, color = 'C8')
plt.axvline(x = av_nV_boot[2], linewidth = 3, label = 'nz medio', color = 'C9')
plt.xlabel(r'$n_z$', fontsize = 20)
plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
plt.legend(loc = 0, fontsize = 20)
plt.grid(which = 'both', axis = 'both', linewidth = 2, linestyle = '--', alpha = 0.8)

#plt.savefig(path_analisis+'hist_normalV_copl_boot_{}'.format(shock_date))



#histograma angulo normales y Bu
    
#hist_thetaB, bins_thetaB = np.histogram(thetaB_boot, 100)
#hist_thetaBuV, bins_thetaBuV = np.histogram(thetaBuV_boot, 100)
#hist_thetaBdV, bins_thetaBdV = np.histogram(thetaBdV_boot, 100)
#hist_thetaBduV, bins_thetaBduV = np.histogram(thetaBduV_boot, 100)
#hist_thetaV, bins_thetaV = np.histogram(thetaV_boot, 100)


plt.figure(9, figsize = (30,30))
plt.suptitle(r'Histograma $\theta_{Bn}$ upstream', fontsize = 30)

graph = plt.subplot(151)
plt.title(r'$n_1$', fontsize = 25)
#plt.plot(bins_thetaB[:-1], hist_thetaB, linewidth = 3, color = 'C0')
plt.hist(thetaB_boot[:,0], bins = 70, color = 'C0')
plt.axvline(x = av_thetaB_boot, linewidth = 3, label = r'$\theta_{Bn}$ medio', color = 'C1')
plt.xlabel(r'$\theta_{Bn}$ [grados]', fontsize = 20)
plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
plt.legend(loc = 0, fontsize = 20)
plt.grid(which = 'both', axis = 'both', linewidth = 2, linestyle = '--', alpha = 0.8)

plt.subplot(152, sharey = graph)
plt.title(r'$n_2$', fontsize = 25)
#plt.plot(bins_thetaBuV[:-1], hist_thetaBuV, linewidth = 3, color = 'C2')
plt.hist(thetaBuV_boot[:,0], bins = 70, color = 'C2')
plt.axvline(x = av_thetaBuV_boot, linewidth = 3, label = r'$\theta_{Bn}$ medio', color = 'C3')
plt.xlabel(r'$\theta_{Bn}$ [grados]', fontsize = 20)
plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
plt.legend(loc = 0, fontsize = 20)
plt.grid(which = 'both', axis = 'both', linewidth = 2, linestyle = '--', alpha = 0.8)

plt.subplot(153, sharey = graph)
plt.title(r'$n_3$', fontsize = 25)
#plt.plot(bins_thetaBdV[:-1], hist_thetaBdV, linewidth = 3, color = 'C4')
plt.hist(thetaBdV_boot[:,0], bins = 70, color = 'C4')
plt.axvline(x = av_thetaBdV_boot, linewidth = 3, label = r'$\theta_{Bn}$ medio', color = 'C5')
plt.xlabel(r'$\theta_{Bn}$ [grados]', fontsize = 20)
plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
plt.legend(loc = 0, fontsize = 20)
plt.grid(which = 'both', axis = 'both', linewidth = 2, linestyle = '--', alpha = 0.8)

plt.subplot(154, sharey = graph)
plt.title(r'$n_4$', fontsize = 25)
#plt.plot(bins_thetaBduV[:-1], hist_thetaBduV, linewidth = 3, color = 'C6')
plt.hist(thetaBuV_boot[:,0], bins = 70, color = 'C6')
plt.axvline(x = av_thetaBduV_boot, linewidth = 3, label = r'$\theta_{Bn}$ medio', color = 'C7')
plt.xlabel(r'$\theta_{Bn}$ [grados]', fontsize = 20)
plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
plt.legend(loc = 0, fontsize = 20)
plt.grid(which = 'both', axis = 'both', linewidth = 2, linestyle = '--', alpha = 0.8)

plt.subplot(155, sharey = graph)
plt.title(r'$n_5$', fontsize = 25)
#plt.plot(bins_thetaV[:-1], hist_thetaV, linewidth = 3, color = 'C8')
plt.hist(thetaV_boot[:,0], bins = 70, color = 'C8')
plt.axvline(x = av_thetaV_boot, linewidth = 3, label = r'$\theta_{Bn}$ medio', color = 'C9')
plt.xlabel(r'$\theta_{Bn}$ [grados]', fontsize = 20)
plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
plt.legend(loc = 0, fontsize = 20)
plt.grid(which = 'both', axis = 'both', linewidth = 2, linestyle = '--', alpha = 0.8)

#plt.savefig(path_analisis+'hist_thetaBun_copl_boot_{}'.format(shock_date))

#%%#####################################################################################################
########################################################################################################
########################################################################################################
########################################################################################################
#%%------------------- COPLANARIDAD CON VARIACION DE INTERVALOS UP/DOWNSTREAM --------------------------


#stack de matrices de [Bx,By,Bz] y [vel x,vel y,vel z] up/downstream para cada intervalo seleccionado

Ba = fcop.intervalo(Bx,By,Bz,lim_t1u,lim_t2u,t_mag,i_u5,f_u5)
Bb = fcop.intervalo(Bx,By,Bz,lim_t1d,lim_t2d,t_mag,i_d5,f_d5)
Va = fcop.intervalo(vel[:,0],vel[:,1],vel[:,2],lim_t1u,lim_t2u,t_swia_mom,iu_v5,fu_v5)
Vb = fcop.intervalo(vel[:,0],vel[:,1],vel[:,2],lim_t1d,lim_t2d,t_swia_mom,id_v5,fd_v5)

Lu = len(Ba[0,0,:])
Ld = len(Bb[0,0,:])
Luv = len(Va[0,0,:])
Ldv = len(Vb[0,0,:])


#campos upstream para cada sample y sus std

Bu_s = np.empty([Lu,3])
err_Bu_s = np.empty([Lu,3])
Bd_s = np.empty([Ld,3])
err_Bd_s = np.empty([Ld,3])
Vu_s = np.empty([Luv,3])
err_Vu_s = np.empty([Luv,3])
Vd_s = np.empty([Ldv,3])
err_Vd_s = np.empty([Ldv,3])

for i in range(Lu):
    Bu_s[i,:] = np.mean(Ba[:,:,i], axis = 0)
    err_Bu_s[i,:] = np.array([st.stdev(Ba[:,0,i]), st.stdev(Ba[:,1,i]), st.stdev(Ba[:,2,i])])
for i in range(Ld):
    Bd_s[i,:] = np.mean(Bb[:,:,i], axis = 0)
    err_Bu_s[i,:] = np.array([st.stdev(Bb[:,0,i]), st.stdev(Bb[:,1,i]), st.stdev(Bb[:,2,i])])
for i in range(Luv):
    Vu_s[i,:] = np.mean(Va[:,:,i], axis = 0)
    #err_Vu_s[i,:] = np.array([st.stdev(Va[:,0,i]), st.stdev(Va[:,1,i]), st.stdev(Va[:,2,i])])
for i in range(Ldv):
    Vd_s[i,:] = np.mean(Vb[:,:,i], axis = 0)
    #err_Vd_s[i,:] = np.array([st.stdev(Vb[:,0,i]), st.stdev(Vb[:,1,i]), st.stdev(Vb[:,2,i])])


#promedios de Bu, Bd, Vu y Vd entre todos los samples y sus std
    
av_Bu_s = np.mean(Bu_s, axis = 0)
std_Bu_s = np.array([st.stdev(Bu_s[:,0]), st.stdev(Bu_s[:,1]), st.stdev(Bu_s[:,2])])
av_Bd_s = np.mean(Bd_s, axis = 0)
std_Bd_s = np.array([st.stdev(Bd_s[:,0]), st.stdev(Bd_s[:,1]), st.stdev(Bd_s[:,2])])
av_Vu_s = np.mean(Vu_s, axis = 0)
std_Vu_s = np.array([st.stdev(Vu_s[:,0]), st.stdev(Vu_s[:,1]), st.stdev(Vu_s[:,2])])
av_Vd_s = np.mean(Vd_s, axis = 0)
std_Vd_s = np.array([st.stdev(Vd_s[:,0]), st.stdev(Vd_s[:,1]), st.stdev(Vd_s[:,2])])



#calculo normales y angulos variando ambos intervalos

#puedo hacer tantos como el menor de los samples
L_B = np.min([Lu, Ld]) 
L_BV = np.min([Lu, Ld, Luv, Ldv])

nB_s2 = np.empty([L_B,3])
thetaB_s2 = np.empty(L_B)
nBuV_s2 = np.empty([L_BV,3])
thetaBuV_s2 = np.empty(L_BV)
nBdV_s2 = np.empty([L_BV,3])
thetaBdV_s2 = np.empty(L_BV)
nBduV_s2 = np.empty([L_BV,3])
thetaBduV_s2 = np.empty(L_BV)
nV_s2 = np.empty([L_BV,3])
thetaV_s2 = np.empty(L_BV)

for i in range(L_B):    
    nB_s2[i,:], _, _, _, _ = fcop.norm_coplanar(Bd_s[i,:],Bu_s[i,:],Vd_s[i,:],Vu_s[i,:])
    thetaB_s2[i] = fcop.alpha(Bu_s[i,:],nB_s2[i,:])
for i in range(L_BV):    
    _, nBuV_s2[i,:], nBdV_s2[i,:], nBduV_s2[i,:], nV_s2[i,:] = fcop.norm_coplanar(Bd_s[i,:],Bu_s[i,:],Vd_s[i,:],Vu_s[i,:])
    thetaBuV_s2[i] = fcop.alpha(Bu_s[i,:],nBuV_s2[i,:])
    thetaBdV_s2[i] = fcop.alpha(Bu_s[i,:],nBdV_s2[i,:])
    thetaBduV_s2[i] = fcop.alpha(Bu_s[i,:],nBduV_s2[i,:])
    thetaV_s2[i] = fcop.alpha(Bu_s[i,:],nV_s2[i,:])
    

#promedio  de normales y angulos entre todos los que obtuve para cada par de intervalos upstream y downstream
av_nB_s2 = np.mean(nB_s2, axis = 0)
std_nB_s2 = np.array([st.stdev(nB_s2[:,0]), st.stdev(nB_s2[:,1]), st.stdev(nB_s2[:,2])])
av_nBuV_s2 = np.mean(nBuV_s2, axis = 0)
std_nBuV_s2 = np.array([st.stdev(nBuV_s2[:,0]), st.stdev(nBuV_s2[:,1]), st.stdev(nBuV_s2[:,2])])
av_nBdV_s2 = np.mean(nBdV_s2, axis = 0)
std_nBdV_s2 = np.array([st.stdev(nBdV_s2[:,0]), st.stdev(nBdV_s2[:,1]), st.stdev(nBdV_s2[:,2])])
av_nBduV_s2 = np.mean(nBduV_s2, axis = 0)
std_nBduV_s2 = np.array([st.stdev(nBduV_s2[:,0]), st.stdev(nBduV_s2[:,1]), st.stdev(nBduV_s2[:,2])])
av_nV_s2 = np.mean(nV_s2, axis = 0)
std_nV_s2 = np.array([st.stdev(nV_s2[:,0]), st.stdev(nV_s2[:,1]), st.stdev(nV_s2[:,2])])

av_thetaB_s2 = np.mean(thetaB_s2)
std_thetaB_s2 = st.stdev(thetaB_s2)
av_thetaBuV_s2 = np.mean(thetaBuV_s2)
std_thetaBuV_s2 = st.stdev(thetaBuV_s2)
av_thetaBdV_s2 = np.mean(thetaBdV_s2)
std_thetaBdV_s2 = st.stdev(thetaBdV_s2)
av_thetaBduV_s2 = np.mean(thetaBduV_s2)
std_thetaBduV_s2 = st.stdev(thetaBduV_s2)
av_thetaV_s2 = np.mean(thetaV_s2)
std_thetaV_s2 = st.stdev(thetaV_s2)


 
#calculo normales y angulos variando solo intervalo upstream

nB_su = np.empty([Lu,3])
thetaB_su = np.empty(Lu)
nBuV_su = np.empty([Luv,3])
thetaBuV_su = np.empty(Luv)
nBdV_su = np.empty([Luv,3])
thetaBdV_su = np.empty(Luv)
nBduV_su = np.empty([Luv,3])
thetaBduV_su = np.empty(Luv)
nV_su = np.empty([Luv,3])
thetaV_su = np.empty(Luv)

for i in range(Lu):    
    nB_su[i,:], _, _, _, _ = fcop.norm_coplanar(Bd,Bu_s[i,:],Vd,Vu_s[i,:])
    thetaB_su[i] = fcop.alpha(Bu_s[i,:],nB_su[i,:])
for i in range(Luv):    
    _, nBuV_su[i,:], nBdV_su[i,:], nBduV_su[i,:], nV_su[i,:] = fcop.norm_coplanar(Bd,Bu_s[i,:],Vd,Vu_s[i,:])
    thetaBuV_su[i] = fcop.alpha(Bu_s[i,:],nBuV_su[i,:])
    thetaBdV_su[i] = fcop.alpha(Bu_s[i,:],nBdV_su[i,:])
    thetaBduV_su[i] = fcop.alpha(Bu_s[i,:],nBduV_su[i,:])
    thetaV_su[i] = fcop.alpha(Bu_s[i,:],nV_su[i,:])
    

#promedio  de normales y angulos entre todos los que obtuve para cada intervalo upstream
av_nB_su = np.mean(nB_su, axis = 0)
std_nB_su = np.array([st.stdev(nB_su[:,0]), st.stdev(nB_su[:,1]), st.stdev(nB_su[:,2])])
av_nBuV_su = np.mean(nBuV_su, axis = 0)
std_nBuV_su = np.array([st.stdev(nBuV_su[:,0]), st.stdev(nBuV_su[:,1]), st.stdev(nBuV_su[:,2])])
av_nBdV_su = np.mean(nBdV_su, axis = 0)
std_nBdV_su = np.array([st.stdev(nBdV_su[:,0]), st.stdev(nBdV_su[:,1]), st.stdev(nBdV_su[:,2])])
av_nBduV_su = np.mean(nBduV_su, axis = 0)
std_nBduV_su = np.array([st.stdev(nBduV_su[:,0]), st.stdev(nBduV_su[:,1]), st.stdev(nBduV_su[:,2])])
av_nV_su = np.mean(nV_su, axis = 0)
std_nV_su = np.array([st.stdev(nV_su[:,0]), st.stdev(nV_su[:,1]), st.stdev(nV_su[:,2])])

av_thetaB_su = np.mean(thetaB_su)
std_thetaB_su = st.stdev(thetaB_su)
av_thetaBuV_su = np.mean(thetaBuV_su)
std_thetaBuV_su = st.stdev(thetaBuV_su)
av_thetaBdV_su = np.mean(thetaBdV_su)
std_thetaBdV_su = st.stdev(thetaBdV_su)
av_thetaBduV_su = np.mean(thetaBduV_su)
std_thetaBduV_su = st.stdev(thetaBduV_su)
av_thetaV_su = np.mean(thetaV_su)
std_thetaV_su = st.stdev(thetaV_su)



#calculo normales y angulos variando solo intervalo downstream

nB_sd = np.empty([Ld,3])
thetaB_sd = np.empty(Ld)
nBuV_sd = np.empty([Ldv,3])
thetaBuV_sd = np.empty(Ldv)
nBdV_sd = np.empty([Ldv,3])
thetaBdV_sd = np.empty(Ldv)
nBduV_sd = np.empty([Ldv,3])
thetaBduV_sd = np.empty(Ldv)
nV_sd = np.empty([Ldv,3])
thetaV_sd = np.empty(Ldv)

for i in range(Ld):    
    nB_sd[i,:], _, _, _, _ = fcop.norm_coplanar(Bd_s[i,:],Bu,Vd_s[i,:],Vu)
    thetaB_sd[i] = fcop.alpha(Bu,nB_sd[i,:])
for i in range(Ldv):    
    _, nBuV_sd[i,:], nBdV_sd[i,:], nBduV_sd[i,:], nV_sd[i,:] = fcop.norm_coplanar(Bd_s[i,:],Bu,Vd_s[i,:],Vu)
    thetaBuV_sd[i] = fcop.alpha(Bu,nBuV_sd[i,:])
    thetaBdV_sd[i] = fcop.alpha(Bu,nBdV_sd[i,:])
    thetaBduV_sd[i] = fcop.alpha(Bu,nBduV_sd[i,:])
    thetaV_sd[i] = fcop.alpha(Bu,nV_sd[i,:])
    

#promedio  de normales y angulos entre todos los que obtuve para cada intervalo downstream
av_nB_sd = np.mean(nB_sd, axis = 0)
std_nB_sd = np.array([st.stdev(nB_sd[:,0]), st.stdev(nB_sd[:,1]), st.stdev(nB_sd[:,2])])
av_nBuV_sd = np.mean(nBuV_sd, axis = 0)
std_nBuV_sd = np.array([st.stdev(nBuV_sd[:,0]), st.stdev(nBuV_sd[:,1]), st.stdev(nBuV_sd[:,2])])
av_nBdV_sd = np.mean(nBdV_sd, axis = 0)
std_nBdV_sd = np.array([st.stdev(nBdV_sd[:,0]), st.stdev(nBdV_sd[:,1]), st.stdev(nBdV_sd[:,2])])
av_nBduV_sd = np.mean(nBduV_sd, axis = 0)
std_nBduV_sd = np.array([st.stdev(nBduV_sd[:,0]), st.stdev(nBduV_sd[:,1]), st.stdev(nBduV_sd[:,2])])
av_nV_sd = np.mean(nV_sd, axis = 0)
std_nV_sd = np.array([st.stdev(nV_sd[:,0]), st.stdev(nV_sd[:,1]), st.stdev(nV_sd[:,2])])

av_thetaB_sd = np.mean(thetaB_sd)
std_thetaB_sd = st.stdev(thetaB_sd)
av_thetaBuV_sd = np.mean(thetaBuV_sd)
std_thetaBuV_sd = st.stdev(thetaBuV_sd)
av_thetaBdV_sd = np.mean(thetaBdV_sd)
std_thetaBdV_sd = st.stdev(thetaBdV_sd)
av_thetaBduV_sd = np.mean(thetaBduV_sd)
std_thetaBduV_sd = st.stdev(thetaBduV_sd)
av_thetaV_sd = np.mean(thetaV_sd)
std_thetaV_sd = st.stdev(thetaV_sd)

#%%

'''
cono de error de las normales coplanares calculadas para la para
la primer eleccion de intervalos up/downstream, a partir de desv
std por variacion de ambos intervalos a la vez
'''

err_perp_nB = std_nB_s2 - (np.dot(std_nB_s2, nB))*nB
cono_err_nB = fcop.alpha(nB, nB + err_perp_nB)

err_perp_nBuV = std_nBuV_s2 - (np.dot(std_nBuV_s2, nBuV))*nBuV
cono_err_nBuV = fcop.alpha(nBuV, nBuV + err_perp_nBuV)

err_perp_nBdV = std_nBdV_s2 - (np.dot(std_nBdV_s2, nBdV))*nBdV
cono_err_nBdV = fcop.alpha(nBdV, nBdV + err_perp_nBdV)

err_perp_nBduV = std_nBduV_s2 - (np.dot(std_nBduV_s2, nBduV))*nBduV
cono_err_nBduV = fcop.alpha(nBduV, nBduV + err_perp_nBduV)

err_perp_nV = std_nV_s2 - (np.dot(std_nV_s2, nV))*nV
cono_err_nV = fcop.alpha(nV, nV + err_perp_nV)

#%% 

#plots de campos Bu y Bd al variar intervalos

plt.figure(10, figsize = (30,20))
plt.suptitle('B ante variación de intervalos upstream y downstream', fontsize = 25)

plt.subplot(211)
plt.plot(Bu_s[:,0], 'o')
plt.plot(Bu_s[:,1], 'o')
plt.plot(Bu_s[:,2], 'o')
plt.axhline(y = av_Bu_s[0], linewidth = 3, label = 'Bux medio', color = 'b')
plt.axhline(y = av_Bu_s[1], linewidth = 3, label = 'Buy medio', color = 'r')
plt.axhline(y = av_Bu_s[2], linewidth = 3, label = 'Buz medio', color = 'g')
plt.ylabel(r'$B_u$ [nT]', fontsize = 20)
plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
plt.legend(loc = 0, fontsize = 20)
plt.grid(which = 'both', axis = 'both', linewidth = 2, linestyle = '--', alpha = 0.8)

plt.subplot(212)
plt.plot(Bd_s[:,0], 'o')
plt.plot(Bd_s[:,1], 'o')
plt.plot(Bd_s[:,2], 'o')
plt.axhline(y = av_Bd_s[0], linewidth = 3, label = 'Bdx medio', color = 'b')
plt.axhline(y = av_Bd_s[1], linewidth = 3, label = 'Bdy medio', color = 'r')
plt.axhline(y = av_Bd_s[2], linewidth = 3, label = 'Bdz medio', color = 'g')
plt.ylabel(r'$B_d$ [nT]', fontsize = 20)
plt.xlabel(r'Realizaciones', fontsize = 20)
plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
plt.legend(loc = 0, fontsize = 20)
plt.grid(which = 'both', axis = 'both', linewidth = 2, linestyle = '--', alpha = 0.8)

#plt.savefig(path_analisis+'BuBd_coplanarity_variacion_up_down{}'.format(shock_date))


#plots de campos Vu y Vd al variar intervalos

plt.figure(11, figsize = (30,20))
plt.suptitle('V ante variación de intervalos upstream y downstream', fontsize = 25)

plt.subplot(211)
plt.plot(Vu_s[:,0], 'o')
plt.plot(Vu_s[:,1], 'o')
plt.plot(Vu_s[:,2], 'o')
plt.axhline(y = av_Vu_s[0], linewidth = 3, label = 'Vux medio', color = 'b')
plt.axhline(y = av_Vu_s[1], linewidth = 3, label = 'Vuy medio', color = 'r')
plt.axhline(y = av_Vu_s[2], linewidth = 3, label = 'Vuz medio', color = 'g')
plt.ylabel(r'$V_u$ [km/s]', fontsize = 20)
plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
plt.legend(loc = 0, fontsize = 20)
plt.grid(which = 'both', axis = 'both', linewidth = 2, linestyle = '--', alpha = 0.8)

plt.subplot(212)
plt.plot(Vd_s[:,0], 'o')
plt.plot(Vd_s[:,1], 'o')
plt.plot(Vd_s[:,2], 'o')
plt.axhline(y = av_Vd_s[0], linewidth = 3, label = 'Vdx medio', color = 'b')
plt.axhline(y = av_Vd_s[1], linewidth = 3, label = 'Vdy medio', color = 'r')
plt.axhline(y = av_Vd_s[2], linewidth = 3, label = 'Vdz medio', color = 'g')
plt.ylabel(r'$V_d$ [km/s]', fontsize = 20)
plt.xlabel(r'Realizaciones', fontsize = 20)
plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
plt.legend(loc = 0, fontsize = 20)
plt.grid(which = 'both', axis = 'both', linewidth = 2, linestyle = '--', alpha = 0.8)

#plt.savefig(path_analisis+'VuVd_coplanarity_variacion_up_down{}'.format(shock_date))



#plots de las componentes de n al variar los intervalos up/downstream

plt.figure(12, figsize = (30,20))
plt.suptitle(r'$n_1 = \frac{(B_d \times B_u) \times \Delta B}{|(B_d \times B_u) \times \Delta B|}$', fontsize = 30)

p = plt.subplot(131)
plt.title('Variación de intervalos upstream y downstream', fontsize = 20)
plt.plot(nB_s2[:,0], 'o')
plt.plot(nB_s2[:,1], 'o')
plt.plot(nB_s2[:,2], 'o')
plt.axhline(y = av_nB_s2[0], linewidth = 3, label = 'nx medio', color = 'b')
plt.axhline(y = av_nB_s2[1], linewidth = 3, label = 'ny medio', color = 'r')
plt.axhline(y = av_nB_s2[2], linewidth = 3, label = 'nz medio', color = 'g')
plt.xlabel(r'Realizaciones', fontsize = 20)
plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
plt.legend(loc = 0, fontsize = 20)
plt.grid(which = 'both', axis = 'both', linewidth = 2, linestyle = '--', alpha = 0.8)

plt.subplot(132, sharey = p)
plt.title('Variación de intervalo upstream', fontsize = 20)
plt.plot(nB_su[:,0], 'o')
plt.plot(nB_su[:,1], 'o')
plt.plot(nB_su[:,2], 'o')
plt.axhline(y = av_nB_su[0], linewidth = 3, label = 'nx medio', color = 'b')
plt.axhline(y = av_nB_su[1], linewidth = 3, label = 'ny medio', color = 'r')
plt.axhline(y = av_nB_su[2], linewidth = 3, label = 'nz medio', color = 'g')
plt.xlabel(r'Realizaciones', fontsize = 20)
plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
plt.legend(loc = 0, fontsize = 20)
plt.grid(which = 'both', axis = 'both', linewidth = 2, linestyle = '--', alpha = 0.8)

plt.subplot(133, sharey = p)
plt.title('Variación de intervalo downstream', fontsize = 20)
plt.plot(nB_sd[:,0], 'o')
plt.plot(nB_sd[:,1], 'o')
plt.plot(nB_sd[:,2], 'o')
plt.axhline(y = av_nB_sd[0], linewidth = 3, label = 'nx medio', color = 'b')
plt.axhline(y = av_nB_sd[1], linewidth = 3, label = 'ny medio', color = 'r')
plt.axhline(y = av_nB_sd[2], linewidth = 3, label = 'nz medio', color = 'g')
plt.xlabel(r'Realizaciones', fontsize = 20)
plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
plt.legend(loc = 0, fontsize = 20)
plt.grid(which = 'both', axis = 'both', linewidth = 2, linestyle = '--', alpha = 0.8)

#plt.savefig(path_analisis+'nB_copl_variacion_up_down{}'.format(shock_date))


plt.figure(13, figsize = (30,20))
plt.suptitle(r'$n_2 = \frac{(B_u \times \Delta V) \times \Delta B}{|(B_u \times \Delta V) \times \Delta B|}$', fontsize = 30)

p = plt.subplot(131)
plt.title('Variación de intervalos upstream y downstream', fontsize = 20)
plt.plot(nBuV_s2[:,0], 'o')
plt.plot(nBuV_s2[:,1], 'o')
plt.plot(nBuV_s2[:,2], 'o')
plt.axhline(y = av_nBuV_s2[0], linewidth = 3, label = 'nx medio', color = 'b')
plt.axhline(y = av_nBuV_s2[1], linewidth = 3, label = 'ny medio', color = 'r')
plt.axhline(y = av_nBuV_s2[2], linewidth = 3, label = 'nz medio', color = 'g')
plt.xlabel(r'Realizaciones', fontsize = 20)
plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
plt.legend(loc = 0, fontsize = 20)
plt.grid(which = 'both', axis = 'both', linewidth = 2, linestyle = '--', alpha = 0.8)

plt.subplot(132, sharey = p)
plt.title('Variación de intervalo upstream', fontsize = 20)
plt.plot(nBuV_su[:,0], 'o')
plt.plot(nBuV_su[:,1], 'o')
plt.plot(nBuV_su[:,2], 'o')
plt.axhline(y = av_nBuV_su[0], linewidth = 3, label = 'nx medio', color = 'b')
plt.axhline(y = av_nBuV_su[1], linewidth = 3, label = 'ny medio', color = 'r')
plt.axhline(y = av_nBuV_su[2], linewidth = 3, label = 'nz medio', color = 'g')
plt.xlabel(r'Realizaciones', fontsize = 20)
plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
plt.legend(loc = 0, fontsize = 20)
plt.grid(which = 'both', axis = 'both', linewidth = 2, linestyle = '--', alpha = 0.8)

plt.subplot(133, sharey = p)
plt.title('Variación de intervalo downstream', fontsize = 20)
plt.plot(nBuV_sd[:,0], 'o')
plt.plot(nBuV_sd[:,1], 'o')
plt.plot(nBuV_sd[:,2], 'o')
plt.axhline(y = av_nBuV_sd[0], linewidth = 3, label = 'nx medio', color = 'b')
plt.axhline(y = av_nBuV_sd[1], linewidth = 3, label = 'ny medio', color = 'r')
plt.axhline(y = av_nBuV_sd[2], linewidth = 3, label = 'nz medio', color = 'g')
plt.xlabel(r'Realizaciones', fontsize = 20)
plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
plt.legend(loc = 0, fontsize = 20)
plt.grid(which = 'both', axis = 'both', linewidth = 2, linestyle = '--', alpha = 0.8)

#plt.savefig(path_analisis+'nBuv_copl_variacion_up_down{}'.format(shock_date))


plt.figure(14, figsize = (30,20))
plt.suptitle(r'$n_3 = \frac{(B_d \times \Delta V) \times \Delta B}{|(B_d \times \Delta V) \times \Delta B|}$', fontsize = 30)

p = plt.subplot(131)
plt.title('Variación de intervalos upstream y downstream', fontsize = 20)
plt.plot(nBdV_s2[:,0], 'o')
plt.plot(nBdV_s2[:,1], 'o')
plt.plot(nBdV_s2[:,2], 'o')
plt.axhline(y = av_nBdV_s2[0], linewidth = 3, label = 'nx medio', color = 'b')
plt.axhline(y = av_nBdV_s2[1], linewidth = 3, label = 'ny medio', color = 'r')
plt.axhline(y = av_nBdV_s2[2], linewidth = 3, label = 'nz medio', color = 'g')
plt.xlabel(r'Realizaciones', fontsize = 20)
plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
plt.legend(loc = 0, fontsize = 20)
plt.grid(which = 'both', axis = 'both', linewidth = 2, linestyle = '--', alpha = 0.8)

plt.subplot(132, sharey = p)
plt.title('Variación de intervalo upstream', fontsize = 20)
plt.plot(nBdV_su[:,0], 'o')
plt.plot(nBdV_su[:,1], 'o')
plt.plot(nBdV_su[:,2], 'o')
plt.axhline(y = av_nBdV_su[0], linewidth = 3, label = 'nx medio', color = 'b')
plt.axhline(y = av_nBdV_su[1], linewidth = 3, label = 'ny medio', color = 'r')
plt.axhline(y = av_nBdV_su[2], linewidth = 3, label = 'nz medio', color = 'g')
plt.xlabel(r'Realizaciones', fontsize = 20)
plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
plt.legend(loc = 0, fontsize = 20)
plt.grid(which = 'both', axis = 'both', linewidth = 2, linestyle = '--', alpha = 0.8)

plt.subplot(133, sharey = p)
plt.title('Variación de intervalo downstream', fontsize = 20)
plt.plot(nBdV_sd[:,0], 'o')
plt.plot(nBdV_sd[:,1], 'o')
plt.plot(nBdV_sd[:,2], 'o')
plt.axhline(y = av_nBdV_sd[0], linewidth = 3, label = 'nx medio', color = 'b')
plt.axhline(y = av_nBdV_sd[1], linewidth = 3, label = 'ny medio', color = 'r')
plt.axhline(y = av_nBdV_sd[2], linewidth = 3, label = 'nz medio', color = 'g')
plt.xlabel(r'Realizaciones', fontsize = 20)
plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
plt.legend(loc = 0, fontsize = 20)
plt.grid(which = 'both', axis = 'both', linewidth = 2, linestyle = '--', alpha = 0.8)

#plt.savefig(path_analisis+'nBdV_copl_variacion_up_down{}'.format(shock_date))


plt.figure(15, figsize = (30,20))
plt.suptitle(r'$n_4 = \frac{(\Delta B \times \Delta V) \times \Delta B}{|(\Delta B \times \Delta V) \times \Delta B|}$', fontsize = 30)

p = plt.subplot(131)
plt.title('Variación de intervalos upstream y downstream', fontsize = 20)
plt.plot(nBduV_s2[:,0], 'o')
plt.plot(nBduV_s2[:,1], 'o')
plt.plot(nBduV_s2[:,2], 'o')
plt.axhline(y = av_nBduV_s2[0], linewidth = 3, label = 'nx medio', color = 'b')
plt.axhline(y = av_nBduV_s2[1], linewidth = 3, label = 'ny medio', color = 'r')
plt.axhline(y = av_nBduV_s2[2], linewidth = 3, label = 'nz medio', color = 'g')
plt.xlabel(r'Realizaciones', fontsize = 20)
plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
plt.legend(loc = 0, fontsize = 20)
plt.grid(which = 'both', axis = 'both', linewidth = 2, linestyle = '--', alpha = 0.8)

plt.subplot(132, sharey = p)
plt.title('Variación de intervalo upstream', fontsize = 20)
plt.plot(nBduV_su[:,0], 'o')
plt.plot(nBduV_su[:,1], 'o')
plt.plot(nBduV_su[:,2], 'o')
plt.axhline(y = av_nBduV_su[0], linewidth = 3, label = 'nx medio', color = 'b')
plt.axhline(y = av_nBduV_su[1], linewidth = 3, label = 'ny medio', color = 'r')
plt.axhline(y = av_nBduV_su[2], linewidth = 3, label = 'nz medio', color = 'g')
plt.xlabel(r'Realizaciones', fontsize = 20)
plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
plt.legend(loc = 0, fontsize = 20)
plt.grid(which = 'both', axis = 'both', linewidth = 2, linestyle = '--', alpha = 0.8)

plt.subplot(133, sharey = p)
plt.title('Variación de intervalo downstream', fontsize = 20)
plt.plot(nBduV_sd[:,0], 'o')
plt.plot(nBduV_sd[:,1], 'o')
plt.plot(nBduV_sd[:,2], 'o')
plt.axhline(y = av_nBduV_sd[0], linewidth = 3, label = 'nx medio', color = 'b')
plt.axhline(y = av_nBduV_sd[1], linewidth = 3, label = 'ny medio', color = 'r')
plt.axhline(y = av_nBduV_sd[2], linewidth = 3, label = 'nz medio', color = 'g')
plt.xlabel(r'Realizaciones', fontsize = 20)
plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
plt.legend(loc = 0, fontsize = 20)
plt.grid(which = 'both', axis = 'both', linewidth = 2, linestyle = '--', alpha = 0.8)

#plt.savefig(path_analisis+'nBduV_copl_variacion_up_down{}'.format(shock_date))


plt.figure(16, figsize = (30,20))
plt.suptitle(r'$n_5 = \frac{V_d - V_u}{|V_d - V_u|}$', fontsize = 30)

p = plt.subplot(131)
plt.title('Variación de intervalos upstream y downstream', fontsize = 20)
plt.plot(nV_s2[:,0], 'o')
plt.plot(nV_s2[:,1], 'o')
plt.plot(nV_s2[:,2], 'o')
plt.axhline(y = av_nV_s2[0], linewidth = 3, label = 'nx medio', color = 'b')
plt.axhline(y = av_nV_s2[1], linewidth = 3, label = 'ny medio', color = 'r')
plt.axhline(y = av_nV_s2[2], linewidth = 3, label = 'nz medio', color = 'g')
plt.xlabel(r'Realizaciones', fontsize = 20)
plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
plt.legend(loc = 0, fontsize = 20)
plt.grid(which = 'both', axis = 'both', linewidth = 2, linestyle = '--', alpha = 0.8)

plt.subplot(132, sharey = p)
plt.title('Variación de intervalo upstream', fontsize = 20)
plt.plot(nV_su[:,0], 'o')
plt.plot(nV_su[:,1], 'o')
plt.plot(nV_su[:,2], 'o')
plt.axhline(y = av_nV_su[0], linewidth = 3, label = 'nx medio', color = 'b')
plt.axhline(y = av_nV_su[1], linewidth = 3, label = 'ny medio', color = 'r')
plt.axhline(y = av_nV_su[2], linewidth = 3, label = 'nz medio', color = 'g')
plt.xlabel(r'Realizaciones', fontsize = 20)
plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
plt.legend(loc = 0, fontsize = 20)
plt.grid(which = 'both', axis = 'both', linewidth = 2, linestyle = '--', alpha = 0.8)

plt.subplot(133, sharey = p)
plt.title('Variación de intervalo downstream', fontsize = 20)
plt.plot(nV_sd[:,0], 'o')
plt.plot(nV_sd[:,1], 'o')
plt.plot(nV_sd[:,2], 'o')
plt.axhline(y = av_nV_sd[0], linewidth = 3, label = 'nx medio', color = 'b')
plt.axhline(y = av_nV_sd[1], linewidth = 3, label = 'ny medio', color = 'r')
plt.axhline(y = av_nV_sd[2], linewidth = 3, label = 'nz medio', color = 'g')
plt.xlabel(r'Realizaciones', fontsize = 20)
plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
plt.legend(loc = 0, fontsize = 20)
plt.grid(which = 'both', axis = 'both', linewidth = 2, linestyle = '--', alpha = 0.8)

#plt.savefig(path_analisis+'nV_copl_variacion_up_down{}'.format(shock_date))



#plots de theta_Bu_n al variar los intervalos up/downstream

plt.figure(17, figsize = (30,20))
plt.suptitle(r'$n_1$', fontsize = 30)

p = plt.subplot(131)
plt.title('Variación de intervalos upstream y downstream', fontsize = 20)
plt.plot(thetaB_s2, 'o')
plt.axhline(y = av_thetaB_s2, linewidth = 3, label = r'$\theta_{Bun}$ medio', color = 'b')
plt.ylabel(r'[grados]', fontsize = 20)
plt.xlabel(r'Realizaciones', fontsize = 20)
plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
plt.legend(loc = 0, fontsize = 20)
plt.grid(which = 'both', axis = 'both', linewidth = 2, linestyle = '--', alpha = 0.8)

plt.subplot(132, sharey = p)
plt.title('Variación de intervalo upstream', fontsize = 20)
plt.plot(thetaB_su, 'o')
plt.axhline(y = av_thetaB_su, linewidth = 3, label = r'$\theta_{Bun}$ medio', color = 'b')
plt.xlabel(r'Realizaciones', fontsize = 20)
plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
plt.legend(loc = 0, fontsize = 20)
plt.grid(which = 'both', axis = 'both', linewidth = 2, linestyle = '--', alpha = 0.8)

plt.subplot(133, sharey = p)
plt.title('Variación de intervalo downstream', fontsize = 20)
plt.plot(thetaB_sd, 'o')
plt.axhline(y = av_thetaB_sd, linewidth = 3, label = r'$\theta_{Bun}$ medio', color = 'b')
plt.xlabel(r'Realizaciones', fontsize = 20)
plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
plt.legend(loc = 0, fontsize = 20)
plt.grid(which = 'both', axis = 'both', linewidth = 2, linestyle = '--', alpha = 0.8)

#plt.savefig(path_analisis+'thetaB_copl_variacion_up_down{}'.format(shock_date))


plt.figure(18, figsize = (30,20))
plt.suptitle(r'$n_2$', fontsize = 30)

p = plt.subplot(131)
plt.title('Variación de intervalos upstream y downstream', fontsize = 20)
plt.plot(thetaBuV_s2, 'o')
plt.axhline(y = av_thetaBuV_s2, linewidth = 3, label = r'$\theta_{Bun}$ medio', color = 'b')
plt.ylabel(r'[grados]', fontsize = 20)
plt.xlabel(r'Realizaciones', fontsize = 20)
plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
plt.legend(loc = 0, fontsize = 20)
plt.grid(which = 'both', axis = 'both', linewidth = 2, linestyle = '--', alpha = 0.8)

plt.subplot(132, sharey = p)
plt.title('Variación de intervalo upstream', fontsize = 20)
plt.plot(thetaBuV_su, 'o')
plt.axhline(y = av_thetaBuV_su, linewidth = 3, label = r'$\theta_{Bun}$ medio', color = 'b')
plt.xlabel(r'Realizaciones', fontsize = 20)
plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
plt.legend(loc = 0, fontsize = 20)
plt.grid(which = 'both', axis = 'both', linewidth = 2, linestyle = '--', alpha = 0.8)

plt.subplot(133, sharey = p)
plt.title('Variación de intervalo downstream', fontsize = 20)
plt.plot(thetaBuV_sd, 'o')
plt.axhline(y = av_thetaBuV_sd, linewidth = 3, label = r'$\theta_{Bun}$ medio', color = 'b')
plt.xlabel(r'Realizaciones', fontsize = 20)
plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
plt.legend(loc = 0, fontsize = 20)
plt.grid(which = 'both', axis = 'both', linewidth = 2, linestyle = '--', alpha = 0.8)

#plt.savefig(path_analisis+'thetaBuV_copl_variacion_up_down{}'.format(shock_date))


plt.figure(19, figsize = (30,20))
plt.suptitle(r'$n_3$', fontsize = 30)

p = plt.subplot(131)
plt.title('Variación de intervalos upstream y downstream', fontsize = 20)
plt.plot(thetaBdV_s2, 'o')
plt.axhline(y = av_thetaBdV_s2, linewidth = 3, label = r'$\theta_{Bun}$ medio', color = 'b')
plt.ylabel(r'[grados]', fontsize = 20)
plt.xlabel(r'Realizaciones', fontsize = 20)
plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
plt.legend(loc = 0, fontsize = 20)
plt.grid(which = 'both', axis = 'both', linewidth = 2, linestyle = '--', alpha = 0.8)

plt.subplot(132, sharey = p)
plt.title('Variación de intervalo upstream', fontsize = 20)
plt.plot(thetaBdV_su, 'o')
plt.axhline(y = av_thetaBdV_su, linewidth = 3, label = r'$\theta_{Bun}$ medio', color = 'b')
plt.xlabel(r'Realizaciones', fontsize = 20)
plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
plt.legend(loc = 0, fontsize = 20)
plt.grid(which = 'both', axis = 'both', linewidth = 2, linestyle = '--', alpha = 0.8)

plt.subplot(133, sharey = p)
plt.title('Variación de intervalo downstream', fontsize = 20)
plt.plot(thetaBdV_sd, 'o')
plt.axhline(y = av_thetaBdV_sd, linewidth = 3, label = r'$\theta_{Bun}$ medio', color = 'b')
plt.xlabel(r'Realizaciones', fontsize = 20)
plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
plt.legend(loc = 0, fontsize = 20)
plt.grid(which = 'both', axis = 'both', linewidth = 2, linestyle = '--', alpha = 0.8)

#plt.savefig(path_analisis+'thetaBdV_copl_variacion_up_down{}'.format(shock_date))


plt.figure(20, figsize = (30,20))
plt.suptitle(r'$n_4$', fontsize = 30)

p = plt.subplot(131)
plt.title('Variación de intervalos upstream y downstream', fontsize = 20)
plt.plot(thetaBduV_s2, 'o')
plt.axhline(y = av_thetaBduV_s2, linewidth = 3, label = r'$\theta_{Bun}$ medio', color = 'b')
plt.ylabel(r'[grados]', fontsize = 20)
plt.xlabel(r'Realizaciones', fontsize = 20)
plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
plt.legend(loc = 0, fontsize = 20)
plt.grid(which = 'both', axis = 'both', linewidth = 2, linestyle = '--', alpha = 0.8)

plt.subplot(132, sharey = p)
plt.title('Variación de intervalo upstream', fontsize = 20)
plt.plot(thetaBduV_su, 'o')
plt.axhline(y = av_thetaBduV_su, linewidth = 3, label = r'$\theta_{Bun}$ medio', color = 'b')
plt.xlabel(r'Realizaciones', fontsize = 20)
plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
plt.legend(loc = 0, fontsize = 20)
plt.grid(which = 'both', axis = 'both', linewidth = 2, linestyle = '--', alpha = 0.8)

plt.subplot(133, sharey = p)
plt.title('Variación de intervalo downstream', fontsize = 20)
plt.plot(thetaBduV_sd, 'o')
plt.axhline(y = av_thetaBduV_sd, linewidth = 3, label = r'$\theta_{Bun}$ medio', color = 'b')
plt.xlabel(r'Realizaciones', fontsize = 20)
plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
plt.legend(loc = 0, fontsize = 20)
plt.grid(which = 'both', axis = 'both', linewidth = 2, linestyle = '--', alpha = 0.8)

#plt.savefig(path_analisis+'thetaBduV_copl_variacion_up_down{}'.format(shock_date))


plt.figure(21, figsize = (30,20))
plt.suptitle(r'$n_5$', fontsize = 30)

p = plt.subplot(131)
plt.title('Variación de intervalos upstream y downstream', fontsize = 20)
plt.plot(thetaV_s2, 'o')
plt.axhline(y = av_thetaV_s2, linewidth = 3, label = r'$\theta_{Bun}$ medio', color = 'b')
plt.ylabel(r'[grados]', fontsize = 20)
plt.xlabel(r'Realizaciones', fontsize = 20)
plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
plt.legend(loc = 0, fontsize = 20)
plt.grid(which = 'both', axis = 'both', linewidth = 2, linestyle = '--', alpha = 0.8)

plt.subplot(132, sharey = p)
plt.title('Variación de intervalo upstream', fontsize = 20)
plt.plot(thetaV_su, 'o')
plt.axhline(y = av_thetaV_su, linewidth = 3, label = r'$\theta_{Bun}$ medio', color = 'b')
plt.xlabel(r'Realizaciones', fontsize = 20)
plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
plt.legend(loc = 0, fontsize = 20)
plt.grid(which = 'both', axis = 'both', linewidth = 2, linestyle = '--', alpha = 0.8)

plt.subplot(133, sharey = p)
plt.title('Variación de intervalo downstream', fontsize = 20)
plt.plot(thetaV_sd, 'o')
plt.axhline(y = av_thetaV_sd, linewidth = 3, label = r'$\theta_{Bun}$ medio', color = 'b')
plt.xlabel(r'Realizaciones', fontsize = 20)
plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
plt.legend(loc = 0, fontsize = 20)
plt.grid(which = 'both', axis = 'both', linewidth = 2, linestyle = '--', alpha = 0.8)

#plt.savefig(path_analisis+'thetaV_copl_variacion_up_down{}'.format(shock_date))


#%%------------------------------- GUARDO RESULTADOS ------------------------------


# coplanaridad para un sample

datos3 = np.zeros([7,5])

#normales nB, nBuV, nBdV, nBduV, nV
datos3[0,0:3] = nB 
datos3[1,0:3] = nBuV
datos3[2,0:3] = nBdV
datos3[3,0:3] = nBduV
datos3[4,0:3] = nV

#angulos entre normales y Bu
datos3[5,0] = thetaB 
datos3[5,1] = thetaBuV
datos3[5,2] = thetaBdV
datos3[5,3] = thetaBduV
datos3[5,4] = thetaV

#angulos entre normales y Rc
datos3[6,0] = thetaB_Rc 
datos3[6,1] = thetaBuV_Rc
datos3[6,2] = thetaBdV_Rc
datos3[6,3] = thetaBduV_Rc
datos3[6,4] = thetaV_Rc

#angulos entre normales y Vu
datos3[7,0] = thetaB_Vu
datos3[7,1] = thetaBuV_Vu
datos3[7,2] = thetaBdV_Vu
datos3[7,3] = thetaBduV_Vu
datos3[7,4] = thetaV_Vu

#angulos entre normales y vel nave
datos3[8,0] = thetaB_vnave
datos3[8,1] = thetaBuV_vnave
datos3[8,2] = thetaBdV_vnave
datos3[8,3] = thetaBduV_vnave
datos3[8,4] = thetaV_vnave

#np.savetxt(path_analisis+'complanaridad_1sample_shock_{}'.format(shock_date), datos3, delimiter = '\t',
#           header = '\n'.join(['{}'.format(shock_date),'nB',
#                                                 'nBuV',
#                                                 'nBdV',
#                                                 'nBduV',
#                                                 'nV',
#                                                 'angulos entre normales y Bu [grados]',
#                                                 'angulos entre normales y Rc [grados]',
#                                                 'angulos entre normales y Vu',
#                                                 'angulos entre normales y v nave']))


# bootstrap coplanaridad para un sample

datos4 = np.zeros([29,3])

#cantidad de samples
datos4[0,0] = Ns

#Bu medio entre todos los samples y su desvstd
datos4[1,0:3] = av_Bu_boot
datos4[2,0:3] = std_Bu_boot
#Bd medio entre todos los samples y su desvstd
datos4[3,0:3] = av_Bd_boot
datos4[4,0:3] = std_Bd_boot
#Vu medio entre todos los samples y su desvstd
datos4[5,0:3] = av_Vu_boot
datos4[6,0:3] = std_Vu_boot
#Vd medio entre todos los samples y su desvstd
datos4[7,0:3] = av_Vd_boot
datos4[8,0:3] = std_Vd_boot

#nB media entre todos los samples y sudesvstd
datos4[9,0:3] = av_nB_boot
datos4[10,0:3] = std_nB_boot
#nBuV media entre todos los samples y sudesvstd
datos4[11,0:3] = av_nBuV_boot
datos4[12,0:3] = std_nBuV_boot
#nBdV media entre todos los samples y sudesvstd
datos4[13,0:3] = av_nBdV_boot
datos4[14,0:3] = std_nBdV_boot
#nBduV media entre todos los samples y sudesvstd
datos4[15,0:3] = av_nBduV_boot
datos4[16,0:3] = std_nBduV_boot
#nV media entre todos los samples y sudesvstd
datos4[17,0:3] = av_nV_boot
datos4[18,0:3] = std_nV_boot

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

#np.savetxt(path_analisis+'complanaridad_boot_shock_{}'.format(shock_date), datos4, delimiter = '\t',
#           header = '\n'.join(['{}'.format(shock_date),'cantidad de samples',
#                                                 'Bu medio [nT]',
#                                                 'desvstd Bu [nT]',
#                                                 'Bd medio [nT]',
#                                                 'desvstd Bd [nT]',
#                                                 'Vu medio [km/s]',
#                                                 'desvstd Vu [km/s]',
#                                                 'Vd medio [km/s]',
#                                                 'desvstd Vd [km/s]',
#                                                 'normal media nB',
#                                                 'desvstd nB',
#                                                 'normal media nBuV',
#                                                 'desvstd nBuV',
#                                                 'normal media nBdV',
#                                                 'desvstd nBdV',
#                                                 'normal media nBduV',
#                                                 'desvstd nBduV',
#                                                 'normal media nV',
#                                                 'desvstd nV',
#                                                 'angulo medio entre Bu y nB y su desvstd [grados]',
#                                                 'angulo medio entre Bu y nBuV y su desvstd [grados]',
#                                                 'angulo medio entre Bu y nBdV y su desvstd [grados]',
#                                                 'angulo medio entre Bu y nBduV y su desvstd [grados]',
#                                                 'angulo medio entre Bu y nV y su desvstd [grados]',
#                                                 'angulo medio entre Rc y nB y su desvstd [grados]',
#                                                 'angulo medio entre Rc y nBuV y su desvstd [grados]',
#                                                 'angulo medio entre Rc y nBdV y su desvstd [grados]',
#                                                 'angulo medio entre Rc y nBduV y su desvstd [grados]',
#                                                 'angulo medio entre Rc y nV y su desvstd [grados]']))


# coplanaridad variando intervalos up/downstream
    
datos5 = np.zeros([55,5])

#cantidad de intervalos para Bu/Bd y Vu/Vd que puedo tomar 
datos5[0,0] = Lu
datos5[0,1] = Ld
datos5[0,2] = Luv
datos5[0,3] = Ldv

#Bu medio entre todos los samples y su desvstd, Bd medio y su desvstd
datos5[1,0:3] = av_Bu_s
datos5[2,0:3] = std_Bu_s
datos5[3,0:3] = av_Bd_s
datos5[4,0:3] = std_Bd_s
#Vu medio entre todos los samples y su desvstd, Vd medio y su desvstd
datos5[5,0:3] = av_Vu_s
datos5[6,0:3] = std_Vu_s
datos5[7,0:3] = av_Vd_s
datos5[8,0:3] = std_Vd_s

#variando ambos intervalos

#nB medio entre todo los samples y su desvstd
datos5[9,0:3] = av_nB_s2
datos5[10,0:3] = std_nB_s2
#nBuV medio entre todo los samples y su desvstd
datos5[11,0:3] = av_nBuV_s2
datos5[12,0:3] = std_nBuV_s2
#nBdV medio entre todo los samples y su desvstd
datos5[13,0:3] = av_nBdV_s2
datos5[14,0:3] = std_nBdV_s2
#nBduV medio entre todo los samples y su desvstd
datos5[15,0:3] = av_nBduV_s2
datos5[16,0:3] = std_nBduV_s2
#nV medio entre todo los samples y su desvstd
datos5[17,0:3] = av_nV_s2
datos5[18,0:3] = std_nV_s2

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
datos5[25,0:3] = av_nB_su
datos5[26,0:3] = std_nB_su
#nBuV medio entre todo los samples y su desvstd
datos5[27,0:3] = av_nBuV_su
datos5[28,0:3] = std_nBuV_su
#nBdV medio entre todo los samples y su desvstd
datos5[29,0:3] = av_nBdV_su
datos5[30,0:3] = std_nBdV_su
#nBduV medio entre todo los samples y su desvstd
datos5[31,0:3] = av_nBduV_su
datos5[32,0:3] = std_nBduV_su
#nV medio entre todo los samples y su desvstd
datos5[33,0:3] = av_nV_su
datos5[34,0:3] = std_nV_su

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
datos5[40,0:3] = av_nB_sd
datos5[41,0:3] = std_nB_sd
#nBuV medio entre todo los samples y su desvstd
datos5[42,0:3] = av_nBuV_sd
datos5[43,0:3] = std_nBuV_sd
#nBdV medio entre todo los samples y su desvstd
datos5[44,0:3] = av_nBdV_sd
datos5[45,0:3] = std_nBdV_sd
#nBduV medio entre todo los samples y su desvstd
datos5[46,0:3] = av_nBduV_sd
datos5[47,0:3] = std_nBduV_sd
#nV medio entre todo los samples y su desvstd
datos5[48,0:3] = av_nV_sd
datos5[49,0:3] = std_nV_sd

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

#np.savetxt(path_analisis+'complanaridad_variacion_intervalos_shock_{}'.format(shock_date), datos5, delimiter = '\t',
#           header = '\n'.join(['{}'.format(shock_date),'cant samples Bu Bd Vu Vd',
#                                                 'Bu medio [nT]', 'std Bu [nT]', 'Bd medio [nT]', 'std Bd [nT]',
#                                                 'Vu medio [km/s]', 'std Vu [km/s]', 'Vd medio [km/s]', 'std Vd [km/s]',
#                                                 'VARIANDO UP/DOWN',
#                                                 'nB medio', 'std nB',
#                                                 'nBuV medio', 'std nBuV',
#                                                 'nBdV medio', 'std nBdV',
#                                                 'nBduV medio', 'std nBduV',
#                                                 'nV medio', 'std nV',
#                                                 'angulo Bu y nB medio std [grados]',
#                                                 'angulo Bu y nBuV medio std [grados]',
#                                                 'angulo Bu y nBdV medio std [grados]',
#                                                 'angulo Bu y nduV medio std [grados]',
#                                                 'angulo Bu y nV medio std [grados]'
#                                                 'cono error nB nBuV nBdV nBduV nV [grados]',
#                                                 'VARIANDO UP',
#                                                 'nB medio', 'std nB',
#                                                 'nBuV medio', 'std nBuV',
#                                                 'nBdV medio', 'std nBdV',
#                                                 'nBduV medio', 'std nBduV',
#                                                 'nV medio', 'std nV',
#                                                 'angulo Bu y nB medio std [grados]',
#                                                 'angulo Bu y nBuV medio std [grados]',
#                                                 'angulo Bu y nBdV medio std [grados]',
#                                                 'angulo Bu y nduV medio std [grados]',
#                                                 'angulo Bu y nV medio std [grados]'
#                                                 'VARIANDO DOWN',
#                                                 'nB medio', 'std nB',
#                                                 'nBuV medio', 'std nBuV',
#                                                 'nBdV medio', 'std nBdV',
#                                                 'nBduV medio', 'std nBduV',
#                                                 'nV medio', 'std nV',
#                                                 'angulo Bu y nB medio std [grados]',
#                                                 'angulo Bu y nBuV medio std [grados]',
#                                                 'angulo Bu y nBdV medio std [grados]',
#                                                 'angulo Bu y nduV medio std [grados]',
#                                                 'angulo Bu y nV medio std [grados]']))
