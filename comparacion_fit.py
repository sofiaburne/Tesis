
from mag import shock_date
from delimitacion_shock import L, N, theta_N, theta_NRc, Rc, v_nave
from calculos_coplanaridad import nB, nBuV, nBdV, nBduV, nV, thetaB, thetaBuV, thetaBdV, thetaBduV, thetaV, cono_err_nB, cono_err_nBuV, cono_err_nBdV, cono_err_nBduV, cono_err_nV
import funciones_coplanaridad as fcop
import funciones_fit_bowshock as fbow


from importlib import reload
import numpy as np
import os


'''
Carpeta para guardar archivos de resultados
(puede que ya se haya creado al correr mag.py)
ojo: para dia con mas de un shock hacer subcarpetas a mano
'''

path_analisis = r'C:\Users\sofia\Documents\Facultad\Tesis\Analisis/{}/'.format(shock_date)
if not os.path.exists(path_analisis):
    os.makedirs(path_analisis)

#%%

#comparacion entre normales y angulos con campo upstream

#angulo entre normal del fit y nB
angulo_normsB = fcop.alpha(nB,N)
#diferencia del angulo con campo upstream de nB y normal fit
dif_thetasB = abs(thetaB - theta_N)

#angulo entre normal del fit y nBuV
angulo_normsBuV = fcop.alpha(nBuV,N)
#diferencia del angulo con campo upstream de nBuV y normal fit
dif_thetasBuV = abs(thetaBuV - theta_N)

#angulo entre normal del fit y nBdV
angulo_normsBdV = fcop.alpha(nBdV,N)
#diferencia del angulo con campo upstream de nBdV y normal fit
dif_thetasBdV = abs(thetaBdV - theta_N) 

#angulo entre normal del fit y nBduV
angulo_normsBduV = fcop.alpha(nBduV,N)
#diferencia del angulo con campo upstream de nBduV y normal fit
dif_thetasBduV = abs(thetaBduV - theta_N) 

#angulo entre normal del fit y nV
angulo_normsV = fcop.alpha(nV,N)
#diferencia del angulo con campo upstream de nV y normal fit
dif_thetasV = abs(thetaV - theta_N)  


#veo si la normal del fit esta dentro del cono de error de las normales coplanares

if cono_err_nB < angulo_normsB:
    print('normal del fit fuera del cono de error de nB')
else:
    print('normal del fit dentro del cono de error de nB')

if cono_err_nBuV < angulo_normsBuV:
    print('normal del fit fuera del cono de error de nBuV')
else:
    print('normal del fit dentro del cono de error de nBuV')

if cono_err_nBdV < angulo_normsBdV:
    print('normal del fit fuera del cono de error de nBdV')
else:
    print('normal del fit dentro del cono de error de nBdV')

if cono_err_nBduV < angulo_normsBduV:
    print('normal del fit fuera del cono de error de nBduV')
else:
    print('normal del fit dentro del cono de error de nBduV')

if cono_err_nV < angulo_normsV:
    print('normal del fit fuera del cono de error de nV')
else:
    print('normal del fit dentro del cono de error de nV')

#%%

#ploteo fit readaptado de Vignes
ax = fbow.plot_implicit(fbow.hiperbola, Rc, L)

#ploteo punto posición de la nave en el centro del shock    
ax.scatter(Rc[0], Rc[1], Rc[2], color="g", s=100)

#ploteo normales coplanares
ax.quiver(Rc[0], Rc[1], Rc[2], nB[0], nB[1], nB[2], length = 2, linewidth = 5, arrow_length_ratio = 0.1, color = 'C0', normalize = True, label = '$n_1$')
ax.quiver(Rc[0], Rc[1], Rc[2], nBuV[0], nBuV[1], nBuV[2], length = 2, linewidth = 5, arrow_length_ratio = 0.1, color = 'C1', normalize = True, label = '$n_2$')
ax.quiver(Rc[0], Rc[1], Rc[2], nBdV[0], nBdV[1], nBdV[2], length = 2, linewidth = 5, arrow_length_ratio = 0.1, color = 'C2', normalize = True, label = '$n_3$')
ax.quiver(Rc[0], Rc[1], Rc[2], nBduV[0], nBduV[1], nBduV[2], length = 2, linewidth = 5, arrow_length_ratio = 0.1, color = 'C3', normalize = True, label = '$n_4$')
ax.quiver(Rc[0], Rc[1], Rc[2], nV[0], nV[1], nV[2], length = 2, linewidth = 5, arrow_length_ratio = 0.1, color = 'C4', normalize = True, label = '$n_5$')

#ploteo normal del fit
n_fit = fbow.norm_fit_MGS(Rc[0],Rc[1],Rc[2],L)
ax.quiver(Rc[0], Rc[1], Rc[2], n_fit[0], n_fit[1], n_fit[2], length = 2, linewidth = 5, arrow_length_ratio = 0.1, color = 'C5', normalize = True, label = 'normal fit')

#ploteo vector vel nave en el shock
ax.quiver(Rc[0], Rc[1], Rc[2], v_nave[0], v_nave[1], v_nave[2], length = 2, linewidth = 5, linestyle = '--', arrow_length_ratio = 0.1, color = 'C6', normalize = True, label = 'velocidad nave')

#dibujo esfera = Marte
x_esfera, y_esfera, z_esfera = fbow.esfera()
ax.plot_surface(x_esfera, y_esfera, z_esfera, color="r")

#ploteo conos de error de n coplanares calculada
#eje_coneB = np.array([[Rc[0],Rc[1],Rc[2]],[nB[0],nB[1],nB[2]]])
#x_coneB, y_coneB, z_coneB = fbow.cone(cono_err_nB*90/np.pi, eje_coneB,1)
#ax.plot_wireframe(x_coneB, y_coneB, z_coneB, color='C7')
#
#eje_coneBuV = np.array([[Rc[0],Rc[1],Rc[2]],[nBuV[0],nBuV[1],nBuV[2]]])
#x_coneBuV, y_coneBuV, z_coneBuV = fbow.cone(cono_err_nBuV*90/np.pi, eje_coneBuV,1)
#ax.plot_wireframe(x_coneBuV, y_coneBuV, z_coneBuV, color='C8')
#
#eje_coneBdV = np.array([[Rc[0],Rc[1],Rc[2]],[nBdV[0],nBdV[1],nBdV[2]]])
#x_coneBdV, y_coneBdV, z_coneBdV = fbow.cone(cono_err_nBdV*90/np.pi, eje_coneBdV,1)
#ax.plot_wireframe(x_coneBdV, y_coneBdV, z_coneBdV, color='C9')
#
#eje_coneBduV = np.array([[Rc[0],Rc[1],Rc[2]],[nBduV[0],nBduV[1],nBduV[2]]])
#x_coneBduV, y_coneBduV, z_coneBduV = fbow.cone(cono_err_nBduV*90/np.pi, eje_coneBduV,1)
#ax.plot_wireframe(x_coneBduV, y_coneBduV, z_coneBduV, color='C10')
#
#eje_coneV = np.array([[Rc[0],Rc[1],Rc[2]],[nV[0],nV[1],nV[2]]])
#x_coneV, y_coneV, z_coneV = fbow.cone(cono_err_nV*90/np.pi, eje_coneV,1)
#ax.plot_wireframe(x_coneV, y_coneV, z_coneV, color='C11')


ax.legend(loc=0, fontsize=20)

#plt.savefig(path_analisis+'vectores_sobre_fit_bowshock_{}'.format(shock_date))


#%%------------------------------- GUARDO RESULTADOS ------------------------------


datos7 = np.zeros([5,5])

#normal del fit
datos7[0,0:3] = N
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
           header = '\n'.join(['{}'.format(shock_date),'normal del fit',
                                                 'angulo Bu y norm fit [grados]',
                                                 'angulo Rc y norm fit [grados]',
                                                 'angulo entre norm fit y nB, nBuV, nBdV, nBduV, nV [grados]',
                                                 'diferencia entre angulo Bu y norm fit y Bu y nB, nBuV, nBdV, nBduV, nVd [grados]']))
