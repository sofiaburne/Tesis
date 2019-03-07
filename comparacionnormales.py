# 0 uso modulo desde otro modulo
# 1 uso modulo y quiero que me haga plots y los guarde
MODO_fit = 1


from mag import shock_date
from delimitacionshock import v_nave
#from subestructuras_calculos import L, N, theta_N, theta_NRc, Rc
#from coplanaridad_calculos import nB, nBuV, nBdV, nBduV, nV, thetaB, thetaBuV, thetaBdV, thetaBduV, thetaV, cono_err_nB, cono_err_nBuV, cono_err_nBdV, cono_err_nBduV, cono_err_nV
#from mva import cono_err_x3, x, thetaMVA
#n_mva = x[2,:]
import coplanaridad_funciones as fcop
import bowshock_funciones as fbow


from importlib import reload
import numpy as np
import matplotlib.pyplot as plt
import os


path_analisis = r'C:\Users\sofia\Documents\Facultad\Tesis\Analisis/{}/'.format(shock_date)
if not os.path.exists(path_analisis):
    os.makedirs(path_analisis)

#%%
L = 2.06564282725141
N = np.array([ 0.87080225,  0.26450542, -0.41441564])
theta_N = 69.02464886575244
theta_NRc = 55.60348473748498
Rc = np.array([ 0.21370354,  1.32791652, -2.08052212])

nB = np.array([ 0.93809579, -0.11924487, -0.32520295])
nBuV = np. array([ 0.72598442,  0.34536685, -0.59470022])
nBdV = np.array([ 0.72943029,  0.34113676, -0.59292256])
nBduV = np.array([ 0.7312194 ,  0.33892431, -0.59198691])
nV = np.array([ 0.7503115 ,  0.37426314, -0.5449402 ])
thetaB = 75.69540417002123
thetaBuV = 78.11906010799035
thetaBdV = 78.07335894456841
thetaBduV = 78.0496594184282
thetaV = 74.53904359490727
cono_err_nB = 3.864464034524949
cono_err_nBuV = 1.0236434346889554
cono_err_nBdV = 0.927137156225793
cono_err_nBduV = 0.910804618042254
cono_err_nV = 0.3491447802817096

cono_err_x3 = 0.2881386029310508
thetaMVA = 71.92927737118515
n_mva = np.array([ 0.00243827, -0.47471115,  0.88013827])

#%%

#angulo entre normales coplanares/mva y normal del fit

angulo_normsB = fcop.alpha(nB,N)
angulo_normsBuV = fcop.alpha(nBuV,N)
angulo_normsBdV = fcop.alpha(nBdV,N)
angulo_normsBduV = fcop.alpha(nBduV,N)
angulo_normsV = fcop.alpha(nV,N)

angulo_normsMVA = fcop.alpha(n_mva,N)


#comparacion de angulos con Bu de normal del fit y normales coplanres/mva

dif_thetasB = abs(thetaB - theta_N)
dif_thetasBuV = abs(thetaBuV - theta_N)
dif_thetasBdV = abs(thetaBdV - theta_N) 
dif_thetasBduV = abs(thetaBduV - theta_N) 
dif_thetasV = abs(thetaV - theta_N)

dif_thetasMVA = abs(thetaMVA - theta_N) 


#veo si la normal del fit esta dentro del cono de error de las normales coplanares/mva

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

if cono_err_x3 < angulo_normsV:
    print('normal del fit fuera del cono de error de n_mva')
else:
    print('normal del fit dentro del cono de error de n_mva')

#%%
    
if MODO_fit == 1:
    
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
    
    #ploteo normal MVA
    #ax.quiver(Rc[0], Rc[1], Rc[2], n_mva[0], n_mva[1], n_mva[2], length = 2, linewidth = 5, arrow_length_ratio = 0.1, color = 'C7', normalize = True, label = '$n_{MVA}$')
    
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
    
   # plt.savefig(path_analisis+'vectores_sobre_fit_bowshock_{}'.format(shock_date))
   # plt.savefig(path_analisis+'vectores_sobre_fit_bowshock_{}.pdf'.format(shock_date))


#%%------------------------------- GUARDO RESULTADOS ------------------------------

if  MODO_fit == 1:
    
    
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
