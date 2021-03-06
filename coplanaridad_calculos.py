# 0 uso modulo desde otro modulo
# 1 uso modulo y quiero que me haga plots y los guarde
MODO_coplanaridad = 0


from mag import shock_date
from delimitacionshock import B, Bx, By, Bz, t_mag
from delimitacionshock import t_swea, flujosenergia_swea, nivelesenergia_swea
from delimitacionshock import t_swia_mom, t_swia_spec, densidad_swia, velocidad_swia, velocidad_swia_norm, temperatura_swia, temperatura_swia_norm, flujosenergia_swia, nivelesenergia_swia
from delimitacionshock import B1, B2, V1, V2, Bd, Bu, Vd, Vu, std_Bd, std_Bu, std_Vu, std_Vd, i_u, f_u, i_d, f_d, iu_v, fu_v, id_v, fd_v, lim_t1u, lim_t2u, lim_t1d, lim_t2d, vel, v_nave
from subestructuras_calculos_2 import N, err_N, theta_N, Rc, err_Rc
from conservaciones import M_A, M_c
import coplanaridad_funciones as fcop

from coplanaridad_plots import CoplanaridadPLOTS as copplot
cpl = copplot()


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



#hay 5 normales que se pueden calcular a partir de coplanaridad
nB, nBuV, nBdV, nBduV, nV = fcop.norm_coplanar(Bd,Bu,Vd,Vu)
err_nB, err_nBuV, err_nBdV, err_nBduV, err_nV = fcop.err_norm_coplanar(Bd, Bu, Vd, Vu, std_Bu, std_Bd, std_Vu, std_Vd)


#dejo los conos de error de variacion de intervalos porque estos son muy grandes

##conos de error
#err_perp_nB = err_nB - np.dot(np.dot(err_nB,nB),nB)
#err_perp_nBuV = err_nBuV - np.dot(np.dot(err_nBuV,nBuV),nBuV)
#err_perp_nBdV = err_nBdV - np.dot(np.dot(err_nBdV,nBdV),nBdV)
#err_perp_nBduV = err_nBduV - np.dot(np.dot(err_nBduV,nBduV),nBduV)
#err_perp_nV = err_nV - np.dot(np.dot(err_nV,nB),nV)
#
#cono_err_nB = fcop.alpha(nB, nB + err_perp_nB)
#cono_err_nBuV = fcop.alpha(nBuV, nBuV + err_perp_nBuV)
#cono_err_nBdV = fcop.alpha(nBdV, nBdV + err_perp_nBdV)
#cono_err_nBduV = fcop.alpha(nBduV, nBduV + err_perp_nBduV)
#cono_err_nV = fcop.alpha(nV, nV + err_perp_nV)


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
err_thetaB = fcop.err_alpha(Bu, nB, std_Bu, err_nB)

thetaBuV = fcop.alpha(Bu,nBuV)
err_thetaBuV = fcop.err_alpha(Bu, nBuV, std_Bu, err_nBuV)

thetaBdV = fcop.alpha(Bu,nBdV)
err_thetaBdV = fcop.err_alpha(Bu, nBdV, std_Bu, err_nBdV)

thetaBduV = fcop.alpha(Bu,nBduV)
err_thetaBduV = fcop.err_alpha(Bu, nBduV, std_Bu, err_nBduV)

thetaV = fcop.alpha(Bu,nV)
err_thetaV = fcop.err_alpha(Bu, nV, std_Bu, err_nV)


#angulos con vector posicion de la nave en centro del shock    
thetaB_Rc = fcop.alpha(nB,Rc)
err_thetaB_Rc = fcop.err_alpha(nB, Rc, err_nB, err_Rc)

thetaBuV_Rc = fcop.alpha(nBuV,Rc)
err_thetaBuV_Rc = fcop.err_alpha(nBuV, Rc, err_nBuV, err_Rc)

thetaBdV_Rc = fcop.alpha(nBdV,Rc)
err_thetaBdV_Rc = fcop.err_alpha(nBdV, Rc, err_nBdV, err_Rc)

thetaBduV_Rc = fcop.alpha(nBduV,Rc)
err_thetaBduV_Rc = fcop.err_alpha(nBduV, Rc, err_nBduV, err_Rc)

thetaV_Rc = fcop.alpha(nV,Rc)
err_thetaV_Rc = fcop.err_alpha(nV, Rc, err_nV, err_Rc)


#angulos entre Vu y normales
thetaB_Vu = fcop.alpha(Vu,nB)
err_thetaB_Vu = fcop.err_alpha(Vu, nB, std_Vu, err_nB)

thetaBuV_Vu = fcop.alpha(Vu,nBuV)
err_thetaBuV_Vu = fcop.err_alpha(Vu, nBuV, std_Vu, err_nBuV)

thetaBdV_Vu = fcop.alpha(Vu,nBdV)
err_thetaBdV_Vu = fcop.err_alpha(Vu, nBdV, std_Vu, err_nBdV)

thetaBduV_Vu = fcop.alpha(Vu,nBduV)
err_thetaBduV_Vu = fcop.err_alpha(Vu, nBduV, std_Vu, err_nBduV)

thetaV_Vu = fcop.alpha(Vu,nV)
err_thetaV_Vu = fcop.err_alpha(Vu, nV, std_Vu, err_nV)


#angulos entre vel nave y normales
err_v_nave = np.array([1e-8,1e-8,1e-8])

thetaB_vnave = fcop.alpha(v_nave,nB)
err_thetaB_vnave = fcop.err_alpha(v_nave, nB, err_v_nave, err_nB)

thetaBuV_vnave = fcop.alpha(v_nave,nBuV)
err_thetaBuV_vnave = fcop.err_alpha(v_nave, nBuV, err_v_nave, err_nBuV)

thetaBdV_vnave = fcop.alpha(v_nave,nBdV)
err_thetaBdV_vnave = fcop.err_alpha(v_nave, nBdV, err_v_nave, err_nBdV)

thetaBduV_vnave = fcop.alpha(v_nave,nBduV)
err_thetaBduV_vnave = fcop.err_alpha(v_nave, nBduV, err_v_nave, err_nBduV)

thetaV_vnave = fcop.alpha(v_nave,nV)
err_thetaV_vnave = fcop.err_alpha(v_nave, nV, err_v_nave, err_nV)


#%% analisis de Bn

Bn_B = np.dot(np.array([Bx,By,Bz]).T, nB)
Bn_BuV = np.dot(np.array([Bx,By,Bz]).T, nBuV)
Bn_BdV = np.dot(np.array([Bx,By,Bz]).T, nBdV)
Bn_BduV = np.dot(np.array([Bx,By,Bz]).T, nBduV)
Bn_V = np.dot(np.array([Bx,By,Bz]).T, nV)

BN = np.dot(np.array([Bx,By,Bz]).T, N)

if MODO_coplanaridad == 1:
    
    fignum = 0
    figsize = (30,15)
    font_title = 45
    font_label = 40
    font_leg = 30
    lw = 2.5
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    ticks_l = 6
    ticks_w = 3
    grid_alpha = 0.8
    
    
    plt.figure(fignum, figsize = figsize)
    plt.suptitle(r'Componente normal del campo magnético - coplanaridad', fontsize = font_title)
    
    plot0 = plt.subplot(611)
    plt.setp(plot0.get_xticklabels(), visible = False)
    #plt.title('n1', fontsize = font_label)
    plt.plot(t_mag, Bn_B, linewidth = lw, color = colors[0])
    plt.axhline(y = 0, linewidth = lw, linestyle = 'dotted', color = colors[9])
    plt.ylabel(r'$B_n$ [nT]'+'\nn1', fontsize = font_label)
    #plt.xlim(xmin = 9.814988, xmax = 9.81946749)
    plt.tick_params(axis = 'both', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
    plt.grid(which = 'both', axis = 'both', linewidth = lw, linestyle = '--', alpha = grid_alpha)
    
    p2 = plt.subplot(612, sharex = plot0)
    plt.setp(p2.get_xticklabels(), visible = False)
    #plt.title('n2', fontsize = font_label)
    plt.plot(t_mag, Bn_BuV, linewidth = lw, color = colors[1])
    plt.axhline(y = 0, linewidth = lw, linestyle = 'dotted', color = colors[9])
    plt.ylabel(r'$B_n$ [nT]'+'\nn2', fontsize = font_label)
    plt.tick_params(axis = 'both', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
    plt.grid(which = 'both', axis = 'both', linewidth = lw, linestyle = '--', alpha = grid_alpha)
    
    p3 = plt.subplot(613, sharex = plot0)
    plt.setp(p3.get_xticklabels(), visible = False)
    #plt.title('n3', fontsize = font_label)
    plt.plot(t_mag, Bn_BdV, linewidth = lw, color = colors[2])
    plt.axhline(y = 0, linewidth = lw, linestyle = 'dotted', color = colors[9])
    plt.ylabel(r'$B_n$ [nT]'+'\nn3', fontsize = font_label)
    plt.tick_params(axis = 'both', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
    plt.grid(which = 'both', axis = 'both', linewidth = lw, linestyle = '--', alpha = grid_alpha)
    
    p4 = plt.subplot(614, sharex = plot0)
    plt.setp(p4.get_xticklabels(), visible = False)
    #plt.title('n4', fontsize = font_label)
    plt.plot(t_mag, Bn_BduV, linewidth = lw, color = colors[3])
    plt.axhline(y = 0, linewidth = lw, linestyle = 'dotted', color = colors[9])
    plt.ylabel(r'$B_n$ [nT]'+'\nn4', fontsize = font_label)
    plt.tick_params(axis = 'both', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
    plt.grid(which = 'both', axis = 'both', linewidth = lw, linestyle = '--', alpha = grid_alpha)
    
    p5 = plt.subplot(615, sharex = plot0)
    #plt.title('n5', fontsize = font_label)
    plt.plot(t_mag, Bn_V, linewidth = lw, color = colors[4])
    plt.axhline(y = 0, linewidth = lw, linestyle = 'dotted', color = colors[9])
    plt.xlabel(r'Tiempo [hora decimal]', fontsize = font_label)
    plt.ylabel(r'$B_n$ [nT]'+'\nn5', fontsize = font_label)
    plt.tick_params(axis = 'both', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
    plt.grid(which = 'both', axis = 'both', linewidth = lw, linestyle = '--', alpha = grid_alpha)
    
    p6 = plt.subplot(616, sharex = plot0)
    #plt.title('N', fontsize = font_label)
    plt.plot(t_mag, BN, linewidth = lw, color = colors[5])
    plt.axhline(y = 0, linewidth = lw, linestyle = 'dotted', color = colors[9])
    plt.xlabel(r'Tiempo [hora decimal]', fontsize = font_label)
    plt.ylabel(r'$B_n$ [nT]'+'\nN', fontsize = font_label)
    plt.tick_params(axis = 'both', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
    plt.grid(which = 'both', axis = 'both', linewidth = lw, linestyle = '--', alpha = grid_alpha)
    
    #plt.savefig(path_analisis+'Bn_coplanar{}'.format(shock_date))
    #plt.savefig(path_analisis+'Bn_coplanar{}.pdf'.format(shock_date))
    

#%% analisis de Bn

Bn_B = np.dot(np.array([Bx,By,Bz]).T, nB)
Bn_BuV = np.dot(np.array([Bx,By,Bz]).T, nBuV)
Bn_BdV = np.dot(np.array([Bx,By,Bz]).T, nBdV)
Bn_BduV = np.dot(np.array([Bx,By,Bz]).T, nBduV)
Bn_V = np.dot(np.array([Bx,By,Bz]).T, nV)

BN = np.dot(np.array([Bx,By,Bz]).T, N)

if MODO_coplanaridad == 1:
     
    fignum = 0
    figsize = (30,25)
    font_title = 45
    font_label = 40
    font_leg = 30
    lw = 2.5
    lw2 = 3.5
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    ticks_l = 6
    ticks_w = 3
    grid_alpha = 0.8
    
    xlim = [9.7, 10.1]
    
    
    plt.figure(fignum, figsize = figsize)
    plt.suptitle(r'$\bf{MAVEN,}$ $\bf{MAG}$  $\bf{Dec}$ $\bf{25,}$ $\bf{2014,}$ $\bf{9:42:00}$ $\bf{-}$ $\bf{10:06:00}$ $\bf{UTC}$', fontsize = font_title)
    plt.subplots_adjust(top=0.9, bottom=0.10, left=0.10, right=0.95, hspace=0.8, wspace=0.3)

    plot0 = plt.subplot(511)
    plt.setp(plot0.get_xticklabels(), visible = False)
    plt.plot(t_mag, B, linewidth = lw, color = colors[0])
    plot0.axes.axvspan(xmin = 9.814, xmax = 9.819, facecolor = 'y', alpha = 0.5)
    plt.axhline(y = 0, linewidth = lw, linestyle = 'dotted', color = colors[9])
    plt.ylabel(r'$|B_{SW,MSO}|$'+'\n[nT]', fontsize = font_label)
    plt.xlim(xmin = xlim[0], xmax = xlim[1])
    plt.yticks(np.arange(0,45,20))
    plt.ylim(0,40)
    plt.tick_params(axis = 'both', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
    plt.grid(which = 'both', axis = 'both', linewidth = lw, linestyle = '--', alpha = grid_alpha)
    
    p1 = plt.subplot(512)
    plt.setp(p1.get_xticklabels(), visible = False)
    plt.plot(t_mag, Bn_B, linewidth = lw, color = colors[1])
    p1.axes.axvspan(xmin = 9.814, xmax = 9.819, facecolor = 'y', alpha = 0.5)
    plt.axhline(y = 0, linewidth = lw2, linestyle = 'dotted', color = colors[9])
    plt.ylabel('$n_{cop1}$\n'+r'$B_n$ [nT]', fontsize = font_label)
    plt.xlim(xmin = xlim[0], xmax = xlim[1])
    plt.yticks(np.arange(-10,21,10))
    plt.ylim(-10,20)
    plt.tick_params(axis = 'both', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
    plt.grid(which = 'both', axis = 'both', linewidth = lw, linestyle = '--', alpha = grid_alpha)
    
    p4 = plt.subplot(513)
    plt.setp(p4.get_xticklabels(), visible = False)
    plt.plot(t_mag, Bn_BduV, linewidth = lw, color = colors[2])
    p4.axes.axvspan(xmin = 9.814, xmax = 9.819, facecolor = 'y', alpha = 0.5)
    plt.axhline(y = 0, linewidth = lw2, linestyle = 'dotted', color = colors[9])
    plt.ylabel('$n_{cop4}$\n'+r'$B_n$ [nT]', fontsize = font_label)
    plt.xlim(xmin = xlim[0], xmax = xlim[1])
    plt.yticks(np.arange(-10,21,10))
    plt.ylim(-10,20)
    plt.tick_params(axis = 'both', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
    plt.grid(which = 'both', axis = 'both', linewidth = lw, linestyle = '--', alpha = grid_alpha)
    
    p5 = plt.subplot(514)
    plt.setp(p5.get_xticklabels(), visible = False)
    plt.plot(t_mag, Bn_V, linewidth = lw, color = colors[3])
    p5.axes.axvspan(xmin = 9.814, xmax = 9.819, facecolor = 'y', alpha = 0.5)
    plt.axhline(y = 0, linewidth = lw2, linestyle = 'dotted', color = colors[9])
    plt.ylabel('\n$n_{cop5}$\n'+r'$B_n$ [nT]', fontsize = font_label)
    plt.xlim(xmin = xlim[0], xmax = xlim[1])
    plt.yticks(np.arange(-10,21,10))
    plt.ylim(-10,20)
    plt.tick_params(axis = 'both', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
    plt.grid(which = 'both', axis = 'both', linewidth = lw, linestyle = '--', alpha = grid_alpha)
    
    p6 = plt.subplot(515)
    plt.plot(t_mag, BN, linewidth = lw, color = colors[4])
    p6.axes.axvspan(xmin = 9.814, xmax = 9.819, facecolor = 'y', alpha = 0.5)
    plt.axhline(y = 0, linewidth = lw2, linestyle = 'dotted', color = colors[9])
    plt.xlabel(r'Time [hs]', fontsize = font_label)
    plt.ylabel('\n$n{fit}$\n'+r'$B_n$ [nT]', fontsize = font_label)
    plt.xlim(xmin = xlim[0], xmax = xlim[1])
    plt.yticks(np.arange(-10,21,10))
    plt.ylim(-10,20)
    plt.tick_params(axis = 'both', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
    plt.grid(which = 'both', axis = 'both', linewidth = lw, linestyle = '--', alpha = grid_alpha)
    
    plt.savefig(path_analisis+'Bn_coplanar{}'.format(shock_date))
    plt.savefig(path_analisis+'Bn_coplanar{}.pdf'.format(shock_date))   



#%%#####################################################################################################
########################################################################################################
########################################################################################################
########################################################################################################
#%% ----------------------- ANALISIS ESTABILIDAD DE CALCULOS EN SEMI-INTERVALOS ----------------------

#analizo que tan estables son las normales repitiendo los calculos con una y otra mitad de los intervalos

if MODO_coplanaridad == 1:
    
    #calculo campos B y V up/downstream para cada medio subintervalo del original
    
    half_u, half_d, half_Bu, half_Bd, std_half_Bu, std_half_Bd = fcop.campos_half(i_u, f_u, i_d, f_d, B1, B2)
    half_uv, half_dv, half_Vu, half_Vd, std_half_Vu, std_half_Vd = fcop.campos_half(iu_v, fu_v, id_v, fd_v, np.float64(V1), np.float64(V2))   
        
    
    #calculo normales combinando cada una de las mitades de los intervalos upstream y downstream
    #tengo 4 combinaciones en total
        
    half11_nB, half11_nBuV, half11_nBdV, half11_nBduV, half11_nV = fcop.norm_coplanar(half_Bd[0,:],half_Bu[0,:],half_Vd[0,:],half_Vu[0,:])
    half12_nB, half12_nBuV, half12_nBdV, half12_nBduV, half12_nV = fcop.norm_coplanar(half_Bd[1,:],half_Bu[0,:],half_Vd[1,:],half_Vu[0,:])
    half21_nB, half21_nBuV, half21_nBdV, half21_nBduV, half21_nV = fcop.norm_coplanar(half_Bd[0,:],half_Bu[1,:],half_Vd[0,:],half_Vu[1,:])
    half22_nB, half22_nBuV, half22_nBdV, half22_nBduV, half22_nV = fcop.norm_coplanar(half_Bd[1,:],half_Bu[1,:],half_Vd[1,:],half_Vu[1,:])
    
    err_half11_nB, err_half11_nBuV, err_half11_nBdV, err_half11_nBduV, err_half11_nV = fcop.err_norm_coplanar(half_Bd[0,:],half_Bu[0,:],half_Vd[0,:],half_Vu[0,:], std_half_Bd[0,:], std_half_Bu[0,:], std_half_Vd[0,:], std_half_Vu[0,:])
    err_half12_nB, err_half12_nBuV, err_half12_nBdV, err_half12_nBduV, err_half12_nV = fcop.err_norm_coplanar(half_Bd[1,:],half_Bu[0,:],half_Vd[1,:],half_Vu[0,:], std_half_Bd[1,:], std_half_Bu[0,:], std_half_Vd[1,:], std_half_Vu[0,:])
    err_half21_nB, err_half21_nBuV, err_half21_nBdV, err_half21_nBduV, err_half21_nV = fcop.err_norm_coplanar(half_Bd[0,:],half_Bu[1,:],half_Vd[0,:],half_Vu[1,:], std_half_Bd[0,:], std_half_Bu[1,:], std_half_Vd[0,:], std_half_Vu[1,:])
    err_half22_nB, err_half22_nBuV, err_half22_nBdV, err_half22_nBduV, err_half22_nV = fcop.err_norm_coplanar(half_Bd[1,:],half_Bu[1,:],half_Vd[1,:],half_Vu[1,:], std_half_Bd[1,:], std_half_Bu[1,:], std_half_Vd[1,:], std_half_Vu[1,:])
    
    #ordeno en vectores
    half_nB = np.array([half11_nB, half12_nB, half21_nB, half22_nB])
    half_nBuV = np.array([half11_nBuV, half12_nBuV, half21_nBuV, half22_nBuV])
    half_nBdV = np.array([half11_nBdV, half12_nBdV, half21_nBdV, half22_nBdV])
    half_nBduV = np.array([half11_nBduV, half12_nBduV, half21_nBduV, half22_nBduV])
    half_nV = np.array([half11_nV, half12_nV, half21_nV, half22_nV])
    
    err_half_nB = np.array([err_half11_nB, err_half12_nB, err_half21_nB, err_half22_nB])
    err_half_nBuV = np.array([err_half11_nBuV, err_half12_nBuV, err_half21_nBuV, err_half22_nBuV])
    err_half_nBdV = np.array([err_half11_nBdV, err_half12_nBdV, err_half21_nBdV, err_half22_nBdV])
    err_half_nBduV = np.array([err_half11_nBduV, err_half12_nBduV, err_half21_nBduV, err_half22_nBduV])
    err_half_nV = np.array([err_half11_nV, err_half12_nV, err_half21_nV, err_half22_nV])
    
    
    
    #comparo las angulos de las nuevas normales con la del fit
    # para cada tipo de normal tengo 4 angulos por cada una de
    # las combinaciones de medios subintervalos
    
    ang_N_nB, err_ang_N_nB = fcop.half_angulo_N(half_nB, N, err_half_nB, err_N)
    ang_N_nBuV, err_ang_N_nBuV = fcop.half_angulo_N(half_nBuV, N, err_half_nBuV, err_N)
    ang_N_nBdV, err_ang_N_nBdV = fcop.half_angulo_N(half_nBdV, N, err_half_nBdV, err_N)
    ang_N_nBduV, err_ang_N_nBduV = fcop.half_angulo_N(half_nBduV, N, err_half_nBduV, err_N)
    ang_N_nV, err_ang_N_nV = fcop.half_angulo_N(half_nV, N, err_half_nV, err_N)
    
    
    
    #si, para un dado tipo de normal, el angulo con la normal del fit de alguna sus 4 normales posibles
    # es zeda (=5 default) grados mayor que otro de esos 4, considero que los calculos con los intervalos
    # completos son potencialmente inestables y elijo la combinacion de mitades que menor angulo
    # me de para esa normal coplanar
    # si no se cumple esa condicion, salta el mensaje "los calculos son estables" 
    
    half_nB_best, ang_N_nB_best, ind_nB =  fcop.half_best_n(ang_N_nB, half_nB)
    half_nBuV_best, ang_N_nBuV_best, ind_nBuV =  fcop.half_best_n(ang_N_nBuV, half_nBuV)
    half_nBdV_best, ang_N_nBdV_best, ind_nBdV =  fcop.half_best_n(ang_N_nBdV, half_nBuV)
    half_nBduV_best, ang_N_nBduV_best, ind_nBduV =  fcop.half_best_n(ang_N_nBduV, half_nBduV)
    half_nV_best, ang_N_nV_best, ind_nV =  fcop.half_best_n(ang_N_nV, half_nV)
    
    #si para 2 de 5 normales coplanares los calculos me dan inestables, considero que los calculos
    # efectivamente son inestables para todas las normales y paso a elegir la mejor combinacion de
    # medios subintervalos
    
    #chequeo si existen las variables (si existen es porque se cumplio la condicion de half_best_n)
    try: ind_nB
    except NameError: ind_nB = None
    
    try: ind_nBuV
    except NameError: ind_nBuV = None
    
    try: ind_nBdV
    except NameError: ind_nBdV = None
    
    try: ind_nBduV
    except NameError: ind_nBduV = None
    
    try: ind_nV
    except NameError: ind_nV = None
    
    
    #vector de indices de mejor combinacion de medios subintervalos
    #si una dada normal coplanar era estable, entonces el indice de
    # mejor combinacion es None
    ind_comb = np.array([ind_nB, ind_nBuV, ind_nBdV, ind_nBduV, ind_nV])
    
    
    #chequeo si tengo por lo menos dos normales coplanares inestables, en ese caso
    # redefino los campos y normales con las de la mejor combinacion
    test1 = np.count_nonzero(ind_comb!=None)
    if test1 >= 2:
    
        #tomo como mejor combinacion de subintervalos aquella que más se repita en la eleccion de mejor normal
        
        def most_common(lst):
            return max(lst, key=lst.count)
        
        ind_best = most_common(list(ind_comb[ind_comb!=None])) #evaluo el elemento no nulo mas repetido
        
        #me quedo con las normales de la mejor combinacion de subintervalos
        mitad_nB = half_nB[ind_best,:]
        mitad_nBuV = half_nBuV[ind_best,:]
        mitad_nBdV = half_nBdV[ind_best,:]
        mitad_nBduV = half_nBduV[ind_best,:]
        mitad_nV = half_nV[ind_best,:]
        
        err_mitad_nB = err_half_nB[ind_best,:]
        err_mitad_nBuV = err_half_nBuV[ind_best,:]
        err_mitad_nBdV = err_half_nBdV[ind_best,:]
        err_mitad_nBduV = err_half_nBduV[ind_best,:]
        err_mitad_nV = err_half_nV[ind_best,:]
        
        err_perp_mitad_nB = err_mitad_nB - np.dot(np.dot(err_mitad_nB,mitad_nB),mitad_nB)
        err_perp_mitad_nBuV = err_mitad_nBuV - np.dot(np.dot(err_mitad_nBuV,mitad_nBuV),mitad_nBuV)
        err_perp_mitad_nBdV = err_mitad_nBdV - np.dot(np.dot(err_mitad_nBdV,mitad_nBdV),mitad_nBdV)
        err_perp_mitad_nBduV = err_mitad_nBduV - np.dot(np.dot(err_mitad_nBduV,mitad_nBduV),mitad_nBduV)
        err_perp_mitad_nV = err_mitad_nV - np.dot(np.dot(err_mitad_nV,mitad_nV),mitad_nV)
        
        cono_err_mitad_nB = fcop.alpha(mitad_nB, err_perp_mitad_nB)
        cono_err_mitad_nBuV = fcop.alpha(mitad_nBuV, err_perp_mitad_nBuV)
        cono_err_mitad_nBdV = fcop.alpha(mitad_nBdV, err_perp_mitad_nBdV)
        cono_err_mitad_nBduV = fcop.alpha(mitad_nBduV, err_perp_mitad_nBduV)
        cono_err_mitad_nV = fcop.alpha(mitad_nV, err_perp_mitad_nV)
        
        
        #angulos con normal del fit, para normales de la mejor combinacion de subintervalos
        mitad_angN_nB = ang_N_nB[ind_best]
        mitad_angN_nBuV = ang_N_nBuV[ind_best]
        mitad_angN_nBdV = ang_N_nBdV[ind_best]
        mitad_angN_nBduV = ang_N_nBduV[ind_best]
        mitad_angN_nV = ang_N_nV[ind_best]
        
        err_mitad_angN_nB = err_ang_N_nB[ind_best]
        err_mitad_angN_nBuV = err_ang_N_nBuV[ind_best]
        err_mitad_angN_nBdV = err_ang_N_nBdV[ind_best]
        err_mitad_angN_nBduV = err_ang_N_nBduV[ind_best]
        err_mitad_angN_nV = err_ang_N_nV[ind_best]
        
        
        #me quedo con los campos correspondientes a los mejores subintervalos
        if ind_best == 0 or 1:
            mitad_Bu = half_Bu[0:,]
            std_mitad_Bu = std_half_Bu[0:,]
            mitad_Vu = half_Vu[0:,]
            std_mitad_Vu = std_half_Vu[0:,]
            
        elif ind_best == 2 or 3:
            mitad_Bu = half_Bu[1:,]
            std_mitad_Bu = std_half_Bu[1:,]
            mitad_Vu = half_Vu[1:,]
            std_mitad_Vu = std_half_Vu[1:,]
        
        if ind_best == 0 or 2:
            mitad_Bd = half_Bd[0:,]
            std_mitad_Bd = std_half_Bd[0:,]
            mitad_Vd = half_Vd[0:,]
            std_mitad_Vd = std_half_Vd[0:,]
        
        elif ind_best == 1 or 3:  
            std_mitad_Bd = std_half_Bd[1:,]
            mitad_Bd = half_Bd[1:,]
            mitad_Vd = half_Vd[1:,]
            std_mitad_Vd = std_half_Vd[1:,]
    
    
        #angulos con campo upstream para las mejores normales de los semi-intervalos
        mitad_thetaB = fcop.alpha(mitad_Bu,mitad_nB)
        mitad_thetaBuV = fcop.alpha(mitad_Bu,mitad_nBuV)
        mitad_thetaBdV = fcop.alpha(mitad_Bu,mitad_nBdV)
        mitad_thetaBduV = fcop.alpha(mitad_Bu,mitad_nBduV)
        mitad_thetaV = fcop.alpha(mitad_Bu,mitad_nV)
        
        err_mitad_thetaB = fcop.err_alpha(mitad_Bu, mitad_nB, std_mitad_Bu, err_mitad_nB)
        err_mitad_thetaBuV = fcop.err_alpha(mitad_Bu, mitad_nBuV, std_mitad_Bu, err_mitad_nBuV)
        err_mitad_thetaBdV = fcop.err_alpha(mitad_Bu, mitad_nBdV, std_mitad_Bu, err_mitad_nBdV)
        err_mitad_thetaBduV = fcop.err_alpha(mitad_Bu, mitad_nBduV, std_mitad_Bu, err_mitad_nBduV)
        err_mitad_thetaV = fcop.err_alpha(mitad_Bu, mitad_nV, std_mitad_Bu, err_mitad_nV)
            
        
        
        #comparo los angulos de normal del fit entre las normales de los mejores subintervalos
        # y las normales de los intervalos completos. Si la diferencia entre 2 angulos con las
        # nuevas normales y las normales con los intervalos completos es mayor a kappa grados,
        # entonces me quedo con las normales y campos de los subintervalos
        
        kappa = 5
        
        mitad_angN = np.array([mitad_angN_nB, mitad_angN_nBuV, mitad_angN_nBdV, mitad_angN_nBduV, mitad_angN_nV])
        angN = np.array([fcop.alpha(nB,N), fcop.alpha(nBuV,N), fcop.alpha(nBdV,N), fcop.alpha(nBduV,N), fcop.alpha(nV,N)])
        resta_angN = abs(mitad_angN - angN)
        
        test2 = np.count_nonzero(resta_angN>=kappa)
        
        if test2 > 1: #si por lo menos dos normales me dan apreciablemente diferentes...
            print('los calculos en semi-intervalos son más estables')
        
        else: raise ValueError('los calculos en los intervalos completos son estables')
                
    
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

if MODO_coplanaridad == 1:
    
    #normales
    
    xticks_nx = np.arange(0.7, 0.76, 0.02)
    xticks_ny = np.arange(0.3, 0.4, 0.02)
    xticks_nz = np.arange(-0.62, -0.54, 0.02)
    
    cpl.hist_norm_boot(nB_boot, av_nB_boot, 1,
                       r'$n_{cop1} = \frac{(B_d \times B_u) \times \Delta B}{|(B_d \times B_u) \times \Delta B|}$', xticks_nx, xticks_ny, xticks_nz)
    plt.savefig(path_analisis+'hist_normalB_copl_boot_{}'.format(shock_date))
    plt.savefig(path_analisis+'hist_normalB_copl_boot_{}.pdf'.format(shock_date))
    
    cpl.hist_norm_boot(nBuV_boot, av_nBuV_boot, 2,
                       r'$n_{cop2} = \frac{(B_u \times \Delta V) \times \Delta B}{|(B_u \times \Delta V) \times \Delta B|}$', xticks_nx, xticks_ny, xticks_nz)
    plt.savefig(path_analisis+'hist_normalBuV_copl_boot_{}'.format(shock_date))
    plt.savefig(path_analisis+'hist_normalBuV_copl_boot_{}.pdf'.format(shock_date))
    
    cpl.hist_norm_boot(nBdV_boot, av_nBdV_boot, 3,
                       r'$n_{cop3} = \frac{(B_d \times \Delta V) \times \Delta B}{|(B_d \times \Delta V) \times \Delta B|}$', xticks_nx, xticks_ny, xticks_nz)
    plt.savefig(path_analisis+'hist_normalBdV_copl_boot_{}'.format(shock_date))
    plt.savefig(path_analisis+'hist_normalBdV_copl_boot_{}.pdf'.format(shock_date))
  
    cpl.hist_norm_boot(nBduV_boot, av_nBduV_boot, 4,
                       r'$n_{cop4} = \frac{(\Delta B \times \Delta V) \times \Delta B}{|(\Delta B \times \Delta V) \times \Delta B|}$', xticks_nx, xticks_ny, xticks_nz)
    plt.savefig(path_analisis+'hist_normalBduV_copl_boot_{}'.format(shock_date))
    plt.savefig(path_analisis+'hist_normalBduV_copl_boot_{}.pdf'.format(shock_date))
    
    cpl.hist_norm_boot(nV_boot, av_nV_boot, 5,
                       r'$n_{cop5} = \frac{V_d - V_u}{|V_d - V_u|}$', xticks_nx, xticks_ny, xticks_nz)
    plt.savefig(path_analisis+'hist_normalV_copl_boot_{}'.format(shock_date))
    plt.savefig(path_analisis+'hist_normalV_copl_boot_{}.pdf'.format(shock_date))
    
    
    #angulos
    
    cpl.hist_theta_boot(thetaB_boot, av_thetaB_boot, thetaBuV_boot, av_thetaBuV_boot,
                        thetaBdV_boot, av_thetaBdV_boot, thetaBduV_boot, av_thetaBduV_boot,
                        thetaV_boot, av_thetaV_boot, 6)    
    plt.savefig(path_analisis+'hist_thetaBun_copl_boot_{}'.format(shock_date))
    plt.savefig(path_analisis+'hist_thetaBun_copl_boot_{}.pdf'.format(shock_date))



    #plot para thetaBduV_boot
    plt.figure(200, figsize = (20,15))
    #plt.suptitle(r'$\bf{Método}$ $\bf{bootstrap}$,   $\theta_{Bn}^{cop4}$', fontsize = 45)
    
    plt.hist(thetaB_boot[:,0], bins = 60, color = 'C3', alpha = 0.8)
    plt.axvline(x = av_thetaB_boot, linewidth = 2.5, label = r'$\theta_{Bn}$ medio', color = 'k')
    plt.xlabel(r'$\theta_{Bn}$ [°]', fontsize = 40)
    plt.ylabel(r'Número de ocurrencias', fontsize = 40)
    plt.tick_params(axis = 'both', which = 'both', length = 6, width = 3, labelsize = 40)
    plt.xticks(np.arange(72,81,1))
    plt.legend(loc = 0, fontsize = 30)
    plt.grid(which = 'both', axis = 'both', linewidth = 2, linestyle = '--', alpha = 0.8)
    
    plt.savefig(path_analisis+'hist_thetaBduV_copl_boot_{}'.format(shock_date))
    plt.savefig(path_analisis+'hist_thetaBduV_copl_boot_{}.pdf'.format(shock_date))
    
    
#%%#####################################################################################################
########################################################################################################
########################################################################################################
########################################################################################################
#%%------------------- COPLANARIDAD CON VARIACION DE INTERVALOS UP/DOWNSTREAM --------------------------


#stack de matrices de [Bx,By,Bz] y [vel x,vel y,vel z] up/downstream para cada intervalo seleccionado


#ancho del intervalo en hora decimal (ancho de x minutos corresponde a x/60)
q = 5/60

Ba = fcop.intervalo(Bx,By,Bz,t_mag,lim_t1u,lim_t2u,q)
Bb = fcop.intervalo(Bx,By,Bz,t_mag,lim_t1d,lim_t2d,q)
Va = fcop.intervalo(vel[:,0],vel[:,1],vel[:,2],t_swia_mom,lim_t1u,lim_t2u,q)
Vb = fcop.intervalo(vel[:,0],vel[:,1],vel[:,2],t_swia_mom,lim_t1d,lim_t2d,q)

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

norm_Bu_s = np.empty(Lu)
err_norm_Bu_s = np.empty(Lu)
norm_Bd_s = np.empty(Ld)
err_norm_Bd_s = np.empty(Ld)
norm_Vu_s = np.empty(Luv)
err_norm_Vu_s = np.empty(Luv)
norm_Vd_s = np.empty(Ldv)
err_norm_Vd_s = np.empty(Ldv)

for i in range(Lu):
    Bu_s[i,:] = np.mean(Ba[:,:,i], axis = 0)
    err_Bu_s[i,:] = np.array([st.stdev(Ba[:,0,i]), st.stdev(Ba[:,1,i]), st.stdev(Ba[:,2,i])])
    norm_Ba = np.empty_like(Ba[:,0,i])
    for j in range(len(Ba[:,0,0])):
        norm_Ba[j] = np.linalg.norm([Ba[j,0,i], Ba[j,1,i], Ba[j,2,i]])
    norm_Bu_s[i] = np.mean(norm_Ba)
    err_norm_Bu_s[i] = st.stdev(norm_Ba)
        
for i in range(Ld):
    Bd_s[i,:] = np.mean(Bb[:,:,i], axis = 0)
    err_Bd_s[i,:] = np.array([st.stdev(Bb[:,0,i]), st.stdev(Bb[:,1,i]), st.stdev(Bb[:,2,i])])
    norm_Bb = np.empty_like(Bb[:,0,i])
    for j in range(len(Bb[:,0,0])):
        norm_Bb[j] = np.linalg.norm([Bb[j,0,i], Bb[j,1,i], Bb[j,2,i]])
    norm_Bd_s[i] = np.mean(norm_Bb)
    err_norm_Bd_s[i] = st.stdev(norm_Bb)
    
for i in range(Luv):
    Vu_s[i,:] = np.mean(Va[:,:,i], axis = 0)
    err_Vu_s[i,:] = np.array([st.stdev(np.float64(Va[:,0,i])), st.stdev(np.float64(Va[:,1,i])), st.stdev(np.float64(Va[:,2,i]))])
    norm_Va = np.empty_like(Va[:,0,i])
    for j in range(len(Va[:,0,0])):
        norm_Va[j] = np.linalg.norm([Va[j,0,i], Va[j,1,i], Va[j,2,i]])
    norm_Vu_s[i] = np.mean(norm_Va)
    err_norm_Vu_s[i] = st.stdev(norm_Va)

for i in range(Ldv):
    Vd_s[i,:] = np.mean(Vb[:,:,i], axis = 0)
    err_Vd_s[i,:] = np.array([st.stdev(np.float64(Vb[:,0,i])), st.stdev(np.float64(Vb[:,1,i])), st.stdev(np.float64(Vb[:,2,i]))])
    norm_Vb = np.empty_like(Vb[:,0,i])
    for j in range(len(Vb[:,0,0])):
        norm_Vb[j] = np.linalg.norm([Vb[j,0,i], Vb[j,1,i], Vb[j,2,i]])
    norm_Vd_s[i] = np.mean(norm_Vb)
    err_norm_Vd_s[i] = st.stdev(norm_Vb)


#modulos de los campos
#norm_Bu_s = np.linalg.norm(Bu_s, axis = 1)
#norm_Bd_s = np.linalg.norm(Bd_s, axis = 1)
#norm_Vu_s = np.linalg.norm(Vu_s, axis = 1)
#norm_Vd_s = np.linalg.norm(Vd_s, axis = 1)

#promedios de Bu, Bd, Vu y Vd entre todos los samples y sus std
    
av_Bu_s = np.mean(Bu_s, axis = 0)
std_Bu_s = np.array([st.stdev(Bu_s[:,0]), st.stdev(Bu_s[:,1]), st.stdev(Bu_s[:,2])])
av_norm_Bu_s = np.mean(norm_Bu_s)
std_norm_Bu_s = st.stdev(norm_Bu_s)
av_Bd_s = np.mean(Bd_s, axis = 0)
std_Bd_s = np.array([st.stdev(Bd_s[:,0]), st.stdev(Bd_s[:,1]), st.stdev(Bd_s[:,2])])
av_norm_Bd_s = np.mean(norm_Bd_s)
std_norm_Bd_s = st.stdev(norm_Bd_s)
av_Vu_s = np.mean(Vu_s, axis = 0)
std_Vu_s = np.array([st.stdev(Vu_s[:,0]), st.stdev(Vu_s[:,1]), st.stdev(Vu_s[:,2])])
av_norm_Vu_s = np.mean(norm_Vu_s)
std_norm_Vu_s = st.stdev(norm_Vu_s)
av_Vd_s = np.mean(Vd_s, axis = 0)
std_Vd_s = np.array([st.stdev(Vd_s[:,0]), st.stdev(Vd_s[:,1]), st.stdev(Vd_s[:,2])])
av_norm_Vd_s = np.mean(norm_Vd_s)
std_norm_Vd_s = st.stdev(norm_Vd_s)



#calculo normales y angulos variando ambos intervalos

#puedo hacer tantos como el menor de los samples
L_B = np.min([Lu, Ld]) 
L_BV = np.min([Lu, Ld, Luv, Ldv])

nB_s2 = np.empty([L_B,3])
err_nB_s2 = np.empty([L_B,3])
thetaB_s2 = np.empty(L_B)
err_thetaB_s2 = np.empty(L_B)

nBuV_s2 = np.empty([L_BV,3])
err_nBuV_s2 = np.empty([L_BV,3])
thetaBuV_s2 = np.empty(L_BV)
err_thetaBuV_s2 = np.empty(L_BV)

nBdV_s2 = np.empty([L_BV,3])
err_nBdV_s2 = np.empty([L_BV,3])
thetaBdV_s2 = np.empty(L_BV)
err_thetaBdV_s2 = np.empty(L_BV)

nBduV_s2 = np.empty([L_BV,3])
err_nBduV_s2 = np.empty([L_BV,3])
thetaBduV_s2 = np.empty(L_BV)
err_thetaBduV_s2 = np.empty(L_BV)

nV_s2 = np.empty([L_BV,3])
err_nV_s2 = np.empty([L_BV,3])
thetaV_s2 = np.empty(L_BV)
err_thetaV_s2 = np.empty(L_BV)

for i in range(L_B):    
    nB_s2[i,:], _, _, _, _ = fcop.norm_coplanar(Bd_s[i,:],Bu_s[i,:],Vd_s[i,:],Vu_s[i,:])
    err_nB_s2[i,:], _, _, _, _ = fcop.err_norm_coplanar(Bd_s[i,:],Bu_s[i,:],Vd_s[i,:],Vu_s[i,:],err_Bd_s[i,:],err_Bu_s[i,:],err_Vd_s[i,:],err_Vu_s[i,:])
    thetaB_s2[i] = fcop.alpha(Bu_s[i,:],nB_s2[i,:])
    err_thetaB_s2[i] = fcop.err_alpha(Bu_s[i,:],nB_s2[i,:],err_Bu_s[i,:],err_nB_s2[i,:])
for i in range(L_BV):    
    _, nBuV_s2[i,:], nBdV_s2[i,:], nBduV_s2[i,:], nV_s2[i,:] = fcop.norm_coplanar(Bd_s[i,:],Bu_s[i,:],Vd_s[i,:],Vu_s[i,:])
    _, err_nBuV_s2[i,:], err_nBdV_s2[i,:], err_nBduV_s2[i,:], err_nV_s2[i,:] = fcop.err_norm_coplanar(Bd_s[i,:],Bu_s[i,:],Vd_s[i,:],Vu_s[i,:],err_Bd_s[i,:],err_Bu_s[i,:],err_Vd_s[i,:],err_Vu_s[i,:])
    thetaBuV_s2[i] = fcop.alpha(Bu_s[i,:],nBuV_s2[i,:])
    err_thetaBuV_s2[i] = fcop.err_alpha(Bu_s[i,:],nBuV_s2[i,:],err_Bu_s[i,:],err_nBuV_s2[i,:])
    thetaBdV_s2[i] = fcop.alpha(Bu_s[i,:],nBdV_s2[i,:])
    err_thetaBdV_s2[i] = fcop.err_alpha(Bu_s[i,:],nBdV_s2[i,:],err_Bu_s[i,:],err_nBdV_s2[i,:])
    thetaBduV_s2[i] = fcop.alpha(Bu_s[i,:],nBduV_s2[i,:])
    err_thetaBduV_s2[i] = fcop.err_alpha(Bu_s[i,:],nBduV_s2[i,:],err_Bu_s[i,:],err_nBduV_s2[i,:])
    thetaV_s2[i] = fcop.alpha(Bu_s[i,:],nV_s2[i,:])
    err_thetaV_s2[i] = fcop.err_alpha(Bu_s[i,:],nV_s2[i,:],err_Bu_s[i,:],err_nV_s2[i,:])
    

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
err_nB_su = np.empty([Lu,3])
thetaB_su = np.empty(Lu)
err_thetaB_su = np.empty(Lu)

nBuV_su = np.empty([Luv,3])
err_nBuV_su = np.empty([Luv,3])
thetaBuV_su = np.empty(Luv)
err_thetaBuV_su = np.empty(Luv)

nBdV_su = np.empty([Luv,3])
err_nBdV_su = np.empty([Luv,3])
thetaBdV_su = np.empty(Luv)
err_thetaBdV_su = np.empty(Luv)

nBduV_su = np.empty([Luv,3])
err_nBduV_su = np.empty([Luv,3])
thetaBduV_su = np.empty(Luv)
err_thetaBduV_su = np.empty(Luv)

nV_su = np.empty([Luv,3])
err_nV_su = np.empty([Luv,3])
thetaV_su = np.empty(Luv)
err_thetaV_su = np.empty(Luv)

for i in range(Lu):    
    nB_su[i,:], _, _, _, _ = fcop.norm_coplanar(Bd,Bu_s[i,:],Vd,Vu_s[i,:])
    err_nB_su[i,:], _, _, _, _ = fcop.err_norm_coplanar(Bd,Bu_s[i,:],Vd,Vu_s[i,:],std_Bd,err_Bu_s[i,:],std_Vd,err_Vu_s[i,:])
    thetaB_su[i] = fcop.alpha(Bu_s[i,:],nB_su[i,:])
    err_thetaB_su[i] = fcop.err_alpha(Bu_s[i,:],nB_su[i,:],err_Bu_s[i,:],err_nB_su[i,:])
for i in range(Luv):    
    _, nBuV_su[i,:], nBdV_su[i,:], nBduV_su[i,:], nV_su[i,:] = fcop.norm_coplanar(Bd,Bu_s[i,:],Vd,Vu_s[i,:])
    _, err_nBuV_su[i,:], err_nBdV_su[i,:], err_nBduV_su[i,:], err_nV_su[i,:] = fcop.err_norm_coplanar(Bd,Bu_s[i,:],Vd,Vu_s[i,:],std_Bd,err_Bu_s[i,:],std_Vd,err_Vu_s[i,:])
    thetaBuV_su[i] = fcop.alpha(Bu_s[i,:],nBuV_su[i,:])
    err_thetaBuV_su[i] = fcop.err_alpha(Bu_s[i,:],nBuV_su[i,:],err_Bu_s[i,:],err_nBuV_su[i,:])
    thetaBdV_su[i] = fcop.alpha(Bu_s[i,:],nBdV_su[i,:])
    err_thetaBdV_su[i] = fcop.err_alpha(Bu_s[i,:],nBdV_su[i,:],err_Bu_s[i,:],err_nBdV_su[i,:])
    thetaBduV_su[i] = fcop.alpha(Bu_s[i,:],nBduV_su[i,:])
    err_thetaBduV_su[i] = fcop.err_alpha(Bu_s[i,:],nBduV_su[i,:],err_Bu_s[i,:],err_nBduV_su[i,:])
    thetaV_su[i] = fcop.alpha(Bu_s[i,:],nV_su[i,:])
    err_thetaV_su[i] = fcop.err_alpha(Bu_s[i,:],nV_su[i,:],err_Bu_s[i,:],err_nV_su[i,:])
    

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
err_nB_sd = np.empty([Ld,3])
thetaB_sd = np.empty(Ld)
err_thetaB_sd = np.empty(Ld)

nBuV_sd = np.empty([Ldv,3])
err_nBuV_sd = np.empty([Ldv,3])
thetaBuV_sd = np.empty(Ldv)
err_thetaBuV_sd = np.empty(Ldv)

nBdV_sd = np.empty([Ldv,3])
err_nBdV_sd = np.empty([Ldv,3])
thetaBdV_sd = np.empty(Ldv)
err_thetaBdV_sd = np.empty(Ldv)

nBduV_sd = np.empty([Ldv,3])
err_nBduV_sd = np.empty([Ldv,3])
thetaBduV_sd = np.empty(Ldv)
err_thetaBduV_sd = np.empty(Ldv)

nV_sd = np.empty([Ldv,3])
err_nV_sd = np.empty([Ldv,3])
thetaV_sd = np.empty(Ldv)
err_thetaV_sd = np.empty(Ldv)

for i in range(Ld):    
    nB_sd[i,:], _, _, _, _ = fcop.norm_coplanar(Bd_s[i,:],Bu,Vd_s[i,:],Vu)
    err_nB_sd[i,:], _, _, _, _ = fcop.err_norm_coplanar(Bd_s[i,:],Bu,Vd_s[i,:],Vu,err_Bd_s[i,:],std_Bu,err_Vd_s[i,:],std_Vu)
    thetaB_sd[i] = fcop.alpha(Bu,nB_sd[i,:])
    err_thetaB_sd[i] = fcop.err_alpha(Bu,nB_sd[i,:],std_Bu,err_nB_sd[i,:])
        
for i in range(Ldv):    
    _, nBuV_sd[i,:], nBdV_sd[i,:], nBduV_sd[i,:], nV_sd[i,:] = fcop.norm_coplanar(Bd_s[i,:],Bu,Vd_s[i,:],Vu)
    _, err_nBuV_sd[i,:], err_nBdV_sd[i,:], err_nBduV_sd[i,:], err_nV_sd[i,:] = fcop.err_norm_coplanar(Bd_s[i,:],Bu,Vd_s[i,:],Vu,err_Bd_s[i,:],std_Bu,err_Vd_s[i,:],std_Vu)
    thetaBuV_sd[i] = fcop.alpha(Bu,nBuV_sd[i,:])
    err_thetaBuV_sd[i] = fcop.err_alpha(Bu,nBuV_sd[i,:],std_Bu,err_nBuV_sd[i,:])
    thetaBdV_sd[i] = fcop.alpha(Bu,nBdV_sd[i,:])
    err_thetaBdV_sd[i] = fcop.err_alpha(Bu,nBdV_sd[i,:],std_Bu,err_nBdV_sd[i,:])
    thetaBduV_sd[i] = fcop.alpha(Bu,nBduV_sd[i,:])
    err_thetaBduV_sd[i] = fcop.err_alpha(Bu,nBduV_sd[i,:],std_Bu,err_nBduV_sd[i,:])
    thetaV_sd[i] = fcop.alpha(Bu,nV_sd[i,:])
    err_thetaV_sd[i] = fcop.err_alpha(Bu,nV_sd[i,:],std_Bu,err_nV_sd[i,:])
    

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

#en funcion del número de realización

if MODO_coplanaridad == 1:
    
    #campo magnetco
    
    cpl.campos_variacion_updown(Bu_s, av_Bu_s, norm_Bu_s, av_norm_Bu_s,
                                Bd_s, av_Bd_s, norm_Bd_s, av_norm_Bd_s,
                                10, 'B', np.arange(1,13,3), np.arange(6,24,3))
    plt.savefig(path_analisis+'BuBd_coplanarity_variacion_up_down{}'.format(shock_date))
    plt.savefig(path_analisis+'BuBd_coplanarity_variacion_up_down{}.pdf'.format(shock_date))
    
    
    #velocidad
    
    cpl.campos_variacion_updown(Vu_s, av_Vu_s, norm_Vu_s, av_norm_Vu_s,
                                Vd_s, av_Vd_s, norm_Vd_s, av_norm_Vd_s,
                                11, 'V', np.arange(-400,500,200), np.arange(-300,400,200))
    plt.savefig(path_analisis+'VuVd_coplanarity_variacion_up_down{}'.format(shock_date))
    plt.savefig(path_analisis+'VuVd_coplanarity_variacion_up_down{}.pdf'.format(shock_date))
    
    
    #normales
    
    xticks_ud = np.arange(0,16,5)
    xticks_u = np.arange(0,41,10)
    xticks_d = np.arange(0,16,5)
    
    cpl.norm_variacion_updown(nB_s2, av_nB_s2, nB_su, av_nB_su, nB_sd, av_nB_sd,
                              12, r'$n_{cop1} = \frac{(B_d \times B_u) \times \Delta B}{|(B_d \times B_u) \times \Delta B|}$', xticks_ud, xticks_u, xticks_d)
    plt.savefig(path_analisis+'nB_copl_variacion_up_down{}'.format(shock_date))
    plt.savefig(path_analisis+'nB_copl_variacion_up_down{}.pdf'.format(shock_date))
    
    cpl.norm_variacion_updown(nBuV_s2, av_nBuV_s2, nBuV_su, av_nBuV_su, nBuV_sd, av_nBuV_sd,
                              13, r'$n_{cop2} = \frac{(B_u \times \Delta V) \times \Delta B}{|(B_u \times \Delta V) \times \Delta B|}$', xticks_ud, xticks_u, xticks_d)
    plt.savefig(path_analisis+'nBuV_copl_variacion_up_down{}'.format(shock_date))
    plt.savefig(path_analisis+'nBuV_copl_variacion_up_down{}.pdf'.format(shock_date))
    
    cpl.norm_variacion_updown(nBdV_s2, av_nBdV_s2, nBdV_su, av_nBdV_su, nBdV_sd, av_nBdV_sd,
                              14, r'$n_{cop3} = \frac{(B_d \times \Delta V) \times \Delta B}{|(B_d \times \Delta V) \times \Delta B|}$', xticks_ud, xticks_u, xticks_d)
    plt.savefig(path_analisis+'nBdV_copl_variacion_up_down{}'.format(shock_date))
    plt.savefig(path_analisis+'nBdV_copl_variacion_up_down{}.pdf'.format(shock_date))
    
    cpl.norm_variacion_updown(nBduV_s2, av_nBduV_s2, nBduV_su, av_nBduV_su, nBduV_sd, av_nBduV_sd,
                              15, r'$n_{cop4} = \frac{(\Delta B \times \Delta V) \times \Delta B}{|(\Delta B \times \Delta V) \times \Delta B|}$', xticks_ud, xticks_u, xticks_d)
    plt.savefig(path_analisis+'nBduV_copl_variacion_up_down{}'.format(shock_date))
    plt.savefig(path_analisis+'nBduV_copl_variacion_up_down{}.pdf'.format(shock_date))
   
    cpl.norm_variacion_updown(nV_s2, av_nV_s2, nV_su, av_nV_su, nV_sd, av_nV_sd,
                              16, r'$n_{cop5} = \frac{V_d - V_u}{|V_d - V_u|}$', xticks_ud, xticks_u, xticks_d)
    plt.savefig(path_analisis+'nV_copl_variacion_up_down{}'.format(shock_date))
    plt.savefig(path_analisis+'nV_copl_variacion_up_down{}.pdf'.format(shock_date))
    
    
    #angulos

    cpl.theta_variacion_updown(thetaB_s2, av_thetaB_s2, thetaB_su, av_thetaB_su, thetaB_sd, av_thetaB_sd,
                              17, r'$\theta_{Bn}^{cop1}$', xticks_ud, xticks_u, xticks_d)
    plt.savefig(path_analisis+'thetaB_copl_variacion_up_down{}'.format(shock_date))
    plt.savefig(path_analisis+'thetaB_copl_variacion_up_down{}.pdf'.format(shock_date))
    
    cpl.theta_variacion_updown(thetaBuV_s2, av_thetaBuV_s2, thetaBuV_su, av_thetaBuV_su, thetaBuV_sd, av_thetaBuV_sd,
                              18, r'$\theta_{Bn}^{cop2}$', xticks_ud, xticks_u, xticks_d)
    plt.savefig(path_analisis+'thetaBuV_copl_variacion_up_down{}'.format(shock_date))
    plt.savefig(path_analisis+'thetaBuV_copl_variacion_up_down{}.pdf'.format(shock_date))
    
    cpl.theta_variacion_updown(thetaBdV_s2, av_thetaBdV_s2, thetaBdV_su, av_thetaBdV_su, thetaBdV_sd, av_thetaBdV_sd,
                              19, r'$\theta_{Bn}^{cop3}$', xticks_ud, xticks_u, xticks_d)
    plt.savefig(path_analisis+'thetaBdV_copl_variacion_up_down{}'.format(shock_date))
    plt.savefig(path_analisis+'thetaBdV_copl_variacion_up_down{}.pdf'.format(shock_date))
    
    cpl.theta_variacion_updown(thetaBduV_s2, av_thetaBduV_s2, thetaBduV_su, av_thetaBduV_su, thetaBduV_sd, av_thetaBduV_sd,
                              20, r'$\theta_{Bn}^{cop4}$', xticks_ud, xticks_u, xticks_d)
    plt.savefig(path_analisis+'thetaBduV_copl_variacion_up_down{}'.format(shock_date))
    plt.savefig(path_analisis+'thetaBduV_copl_variacion_up_down{}.pdf'.format(shock_date))
    
    cpl.theta_variacion_updown(thetaV_s2, av_thetaV_s2, thetaV_su, av_thetaV_su, thetaV_sd, av_thetaV_sd,
                              21, r'$\theta_{Bn}^{cop5}$', xticks_ud, xticks_u, xticks_d)
    plt.savefig(path_analisis+'thetaV_copl_variacion_up_down{}'.format(shock_date))
    plt.savefig(path_analisis+'thetaV_copl_variacion_up_down{}.pdf'.format(shock_date))

    
#%%

#como histogramas

if MODO_coplanaridad == 1:
    
    #campo magnetico
    
    cpl.hist_campos_variacion_updown(Bu_s, av_Bu_s, norm_Bu_s, av_norm_Bu_s, 22, 'Campo magnético upstream', r'$B_u$', r'$B_{ux}$', r'$B_{uy}$', r'$B_{uz}$')
    plt.savefig(path_analisis+'hist_Bu_copl_variacion_up_down{}'.format(shock_date))
    plt.savefig(path_analisis+'hist_Bu_copl_variacion_up_down{}.pdf'.format(shock_date))
    
    cpl.hist_campos_variacion_updown(Bd_s, av_Bd_s, norm_Bd_s, av_norm_Bd_s, 23, 'Campo magnético downstream', r'$B_d$', r'$B_{dx}$', r'$B_{dy}$', r'$B_{dz}$')
    plt.savefig(path_analisis+'hist_Bd_copl_variacion_up_down{}'.format(shock_date))
    plt.savefig(path_analisis+'hist_Bd_copl_variacion_up_down{}.pdf'.format(shock_date))
    
    #velocidad
    
    cpl.hist_campos_variacion_updown(Vu_s, av_Vu_s, norm_Vu_s, av_norm_Vu_s, 24, 'Velocidad flujo upstream', r'$V_u$', r'$V_{ux}$', r'$V_{uy}$', r'$V_{uz}$')
    plt.savefig(path_analisis+'hist_Vu_copl_variacion_up_down{}'.format(shock_date))
    plt.savefig(path_analisis+'hist_Vu_copl_variacion_up_down{}.pdf'.format(shock_date))
    
    cpl.hist_campos_variacion_updown(Vd_s, av_Vd_s, norm_Vd_s, av_norm_Vd_s, 25, 'Velocidad flujo downstream', r'$V_d$', r'$V_{dx}$', r'$V_{dy}$', r'$V_{dz}$')
    plt.savefig(path_analisis+'hist_Vd_copl_variacion_up_down{}'.format(shock_date))
    plt.savefig(path_analisis+'hist_Vd_copl_variacion_up_down{}.pdf'.format(shock_date))
    
    
    #normales
    
    cpl.hist_norm_variacion_updown(nB_s2, av_nB_s2, nB_su, av_nB_su, nB_sd, av_nB_sd,
                                   r'$n_1 = \frac{(B_d \times B_u) \times \Delta B}{|(B_d \times B_u) \times \Delta B|}$', 26)
    plt.savefig(path_analisis+'hist_nB_copl_variacion_up_down{}'.format(shock_date))
    plt.savefig(path_analisis+'hist_nB_copl_variacion_up_down{}.pdf'.format(shock_date))
    
    cpl.hist_norm_variacion_updown(nBuV_s2, av_nBuV_s2, nBuV_su, av_nBuV_su, nBuV_sd, av_nBuV_sd,
                                   r'$n_2 = \frac{(B_u \times \Delta V) \times \Delta B}{|(B_u \times \Delta V) \times \Delta B|}$', 27)
    plt.savefig(path_analisis+'hist_nBuV_copl_variacion_up_down{}'.format(shock_date))
    plt.savefig(path_analisis+'hist_nBuV_copl_variacion_up_down{}.pdf'.format(shock_date))
    
    cpl.hist_norm_variacion_updown(nBdV_s2, av_nBdV_s2, nBdV_su, av_nBdV_su, nBdV_sd, av_nBdV_sd,
                                   r'$n_3 = \frac{(B_d \times \Delta V) \times \Delta B}{|(B_d \times \Delta V) \times \Delta B|}$', 28)
    plt.savefig(path_analisis+'hist_nBdV_copl_variacion_up_down{}'.format(shock_date))
    plt.savefig(path_analisis+'hist_nBdV_copl_variacion_up_down{}.pdf'.format(shock_date))
    
    cpl.hist_norm_variacion_updown(nBduV_s2, av_nBduV_s2, nBduV_su, av_nBduV_su, nBduV_sd, av_nBduV_sd,
                                   r'$n_4 = \frac{(\Delta B \times \Delta V) \times \Delta B}{|(\Delta B \times \Delta V) \times \Delta B|}$', 29)
    plt.savefig(path_analisis+'hist_nBduV_copl_variacion_up_down{}'.format(shock_date))
    plt.savefig(path_analisis+'hist_nBduV_copl_variacion_up_down{}.pdf'.format(shock_date))
    
    cpl.hist_norm_variacion_updown(nV_s2, av_nV_s2, nV_su, av_nV_su, nV_sd, av_nV_sd,
                                   r'$n_5 = \frac{V_d - V_u}{|V_d - V_u|}$', 30)
    plt.savefig(path_analisis+'hist_nV_copl_variacion_up_down{}.pdf'.format(shock_date))
    
    
    #angulos
    
    cpl.hist_theta_variacion_updown(thetaB_s2, av_thetaB_s2, thetaB_su, av_thetaB_su, thetaB_sd, av_thetaB_sd,
                                   r'$n_1 = \frac{(B_d \times B_u) \times \Delta B}{|(B_d \times B_u) \times \Delta B|}$', 31)
    plt.savefig(path_analisis+'hist_thetaB_copl_variacion_up_down{}'.format(shock_date))
    plt.savefig(path_analisis+'hist_thetaB_copl_variacion_up_down{}.pdf'.format(shock_date))
    
    cpl.hist_theta_variacion_updown(thetaBuV_s2, av_thetaBuV_s2, thetaBuV_su, av_thetaBuV_su, thetaBuV_sd, av_thetaBuV_sd,
                                   r'$n_2 = \frac{(B_u \times \Delta V) \times \Delta B}{|(B_u \times \Delta V) \times \Delta B|}$', 32)
    plt.savefig(path_analisis+'hist_thetaBuV_copl_variacion_up_down{}'.format(shock_date))
    plt.savefig(path_analisis+'hist_thetaBuV_copl_variacion_up_down{}.pdf'.format(shock_date))
    
    cpl.hist_theta_variacion_updown(thetaBdV_s2, av_thetaBdV_s2, thetaBdV_su, av_thetaBdV_su, thetaBdV_sd, av_thetaBdV_sd,
                                   r'$n_3 = \frac{(B_d \times \Delta V) \times \Delta B}{|(B_d \times \Delta V) \times \Delta B|}$', 33)
    plt.savefig(path_analisis+'hist_thetaBdV_copl_variacion_up_down{}'.format(shock_date))
    plt.savefig(path_analisis+'hist_thetaBdV_copl_variacion_up_down{}.pdf'.format(shock_date))
    
    cpl.hist_theta_variacion_updown(thetaBduV_s2, av_thetaBduV_s2, thetaBduV_su, av_thetaBduV_su, thetaBduV_sd, av_thetaBduV_sd,
                                   r'$n_4 = \frac{(\Delta B \times \Delta V) \times \Delta B}{|(\Delta B \times \Delta V) \times \Delta B|}$', 34)
    plt.savefig(path_analisis+'hist_thetaBduV_copl_variacion_up_down{}'.format(shock_date))
    plt.savefig(path_analisis+'hist_thetaBduV_copl_variacion_up_down{}.pdf'.format(shock_date))
    
    cpl.hist_theta_variacion_updown(thetaV_s2, av_thetaV_s2, thetaV_su, av_thetaV_su, thetaV_sd, av_thetaV_sd,
                                   r'$n_5 = \frac{V_d - V_u}{|V_d - V_u|}$', 30)
    plt.savefig(path_analisis+'hist_thetaV_copl_variacion_up_down{}'.format(shock_date))
    plt.savefig(path_analisis+'hist_thetaV_copl_variacion_up_down{}.pdf'.format(shock_date))
    
    
    
#%%------------------------------- GUARDO RESULTADOS ------------------------------

if MODO_coplanaridad == 1:
    
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
    
    np.savetxt(path_analisis+'complanaridad_1sample_shock_{}'.format(shock_date), datos3, delimiter = '\t',
               header = '\n'.join(['{}'.format(shock_date),'nB',
                                                     'nBuV',
                                                     'nBdV',
                                                     'nBduV',
                                                     'nV',
                                                     'angulos entre normales y Bu [grados]',
                                                     'angulos entre normales y Rc [grados]',
                                                     'angulos entre normales y Vu',
                                                     'angulos entre normales y v nave']))
    
    
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
    
    np.savetxt(path_analisis+'complanaridad_boot_shock_{}'.format(shock_date), datos4, delimiter = '\t',
               header = '\n'.join(['{}'.format(shock_date),'cantidad de samples',
                                                     'Bu medio [nT]',
                                                     'desvstd Bu [nT]',
                                                     'Bd medio [nT]',
                                                     'desvstd Bd [nT]',
                                                     'Vu medio [km/s]',
                                                     'desvstd Vu [km/s]',
                                                     'Vd medio [km/s]',
                                                     'desvstd Vd [km/s]',
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
                                                     'angulo medio entre Bu y nB y su desvstd [grados]',
                                                     'angulo medio entre Bu y nBuV y su desvstd [grados]',
                                                     'angulo medio entre Bu y nBdV y su desvstd [grados]',
                                                     'angulo medio entre Bu y nBduV y su desvstd [grados]',
                                                     'angulo medio entre Bu y nV y su desvstd [grados]',
                                                     'angulo medio entre Rc y nB y su desvstd [grados]',
                                                     'angulo medio entre Rc y nBuV y su desvstd [grados]',
                                                     'angulo medio entre Rc y nBdV y su desvstd [grados]',
                                                     'angulo medio entre Rc y nBduV y su desvstd [grados]',
                                                     'angulo medio entre Rc y nV y su desvstd [grados]']))
    
    
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
    
    np.savetxt(path_analisis+'complanaridad_variacion_intervalos_shock_{}'.format(shock_date), datos5, delimiter = '\t',
               header = '\n'.join(['{}'.format(shock_date),'cant samples Bu Bd Vu Vd',
                                                     'Bu medio [nT]', 'std Bu [nT]', 'Bd medio [nT]', 'std Bd [nT]',
                                                     'Vu medio [km/s]', 'std Vu [km/s]', 'Vd medio [km/s]', 'std Vd [km/s]',
                                                     'VARIANDO UP/DOWN',
                                                     'nB medio', 'std nB',
                                                     'nBuV medio', 'std nBuV',
                                                     'nBdV medio', 'std nBdV',
                                                     'nBduV medio', 'std nBduV',
                                                     'nV medio', 'std nV',
                                                     'angulo Bu y nB medio std [grados]',
                                                     'angulo Bu y nBuV medio std [grados]',
                                                     'angulo Bu y nBdV medio std [grados]',
                                                     'angulo Bu y nduV medio std [grados]',
                                                     'angulo Bu y nV medio std [grados]'
                                                     'cono error nB nBuV nBdV nBduV nV [grados]',
                                                     'VARIANDO UP',
                                                     'nB medio', 'std nB',
                                                     'nBuV medio', 'std nBuV',
                                                     'nBdV medio', 'std nBdV',
                                                     'nBduV medio', 'std nBduV',
                                                     'nV medio', 'std nV',
                                                     'angulo Bu y nB medio std [grados]',
                                                     'angulo Bu y nBuV medio std [grados]',
                                                     'angulo Bu y nBdV medio std [grados]',
                                                     'angulo Bu y nduV medio std [grados]',
                                                     'angulo Bu y nV medio std [grados]'
                                                     'VARIANDO DOWN',
                                                     'nB medio', 'std nB',
                                                     'nBuV medio', 'std nBuV',
                                                     'nBdV medio', 'std nBdV',
                                                     'nBduV medio', 'std nBduV',
                                                     'nV medio', 'std nV',
                                                     'angulo Bu y nB medio std [grados]',
                                                     'angulo Bu y nBuV medio std [grados]',
                                                     'angulo Bu y nBdV medio std [grados]',
                                                     'angulo Bu y nduV medio std [grados]',
                                                     'angulo Bu y nV medio std [grados]']))
