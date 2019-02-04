# 0 uso modulo desde otro modulo
# 1 uso modulo y quiero que me haga plots y los guarde
MODO_mva = 0


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import random
import statistics as st
import os

from mag import shock_date
from delimitacionshock import Bx, By, Bz, t_mag, Bu
from subestructuras_calculos import N
import coplanaridad_funciones as fcop

B_vec = np.array([Bx, By, Bz]).T

path_analisis = r'C:\Users\sofia\Documents\Facultad\Tesis\Analisis/{}/'.format(shock_date)
if not os.path.exists(path_analisis):
    os.makedirs(path_analisis)

#%% (EJEMPLO) datos 5:18:20.49 a 5:19:26 Oct 19 1984

path = r'C:\Users\sofia\Documents\Facultad\Tesis\Ejercicio MVA cap8/'
t_mag, B_x, B_y, B_z, v_x, v_y, v_z, N = np.loadtxt(path+'datos.txt', skiprows = 1, unpack = True)
B_vec = np.array([B_x, B_y, B_z]).T
            
#unidades: T(s), B(nT), v(km/s), N(particles/cm^3)

#%%----------------------------------- FUNCIONES GENERALES -------------------------------------------


#elementos matriz de covarianza 
def cov(x):
    A = np.empty([3,3])
    for i in range(3):
        for j in range(3):
            A[i,j] = np.mean(x[:,i]*x[:,j]) - np.mean(x[:,i])*np.mean(x[:,j])
    return A




#reordenamiento de autovalores y autovectores (autovalores de mayor a menor)
def ordered(a, u):
    #autovectores ordenados de menor a mayor (provisorio)
    orden = np.argsort(a) #indices de elementos de a de menor a mayor
    w = [u[:,0], u[:,1], u[:,2]]
    w = np.array([w[i] for i in orden]).T
    #de mayor a menor
    x = np.array([w[:,2], w[:,1], w[:,0]]).T
    l= sorted(a, key = float, reverse = True)
    return(l, x)




#errores autovectores (err_v[i,j] es el angulo de rot del autovector i respecto del autovector j) 
def err_vec(m, l):
    err_v = np.empty([3,3])
    for i in range(3):
        for j in range(3):
            if i != j:
                err_v[i,j] = np.sqrt(l[2]/(m-1) * (l[i]+l[j]-l[2])/(l[i]-l[j])**2)
                if err_v[i,j] > np.pi : err_v[i,j] = abs(2*np.pi - err_v[i,j])
            else:
                err_v[i,j] = 0.0 #pongo esto porque la formula del error vale solo para i disntito de j
    return err_v




#error estadistico de la componente normal del campo B (ie, error de <Bn>)
def err_B(l_3, m, err_v32, err_v31, x, B):
    err_B3 = np.sqrt(l_3/(m-1) + (err_v32 * np.dot(B,x[1,:]))**2 + (err_v31 * np.dot(B,x[0,:]))**2)
    return err_B3




#metodo bootstrap
def boot(B, N):
    
    norm = np.empty([N,3])
    b_mean = np.empty([N,3])
    E = np.empty([N,5]) #col de E: err_13 / err_13_grad / err_23 / err_23_grad / err_B3
    
    
    C = list(B)
    
    for i in range(N):
        
        b = [random.choice(C) for _ in range(len(C))]
        b = np.array(b)
        
        b_mean[i,:] = np.mean(b, axis = 0).T
        
    
        # HAGO MVA AL BOOTSTRAP SAMPLE:
        
        m = cov(b)
        q, z = np.linalg.eig(m) #cada columna de u es un autovector distinto (componentes en cada fila)
        #de mayor a menor:
        q, z = ordered(q, z)
        
        #fuerzo normal externa
        if z[0,2] < 0: z[:,2] = - z[:,2]
        
        #armo terna con los autovectores respetando x3 normal exterior
        w = np.array([z[:,0], np.cross(z[:,2],z[:,0]), z[:,2]]) #cada fila es un autovector distinto
                
        #renombro todo para no tener mil variables.
        norm[i,:] = w[2,:]
        
        #errores
        err_w = err_vec(len(b[:,0]), q) #en radianes
        err_w_grad = err_w*180/(np.pi)  #en grados
        err_bw3 = err_B(q[2], len(b[:,0]), err_w[2,1], err_w[2,0], w, np.mean(b, axis = 0))
        
        E[i,0] = err_w[0,2]
        E[i,1] = err_w_grad[0,2]
        E[i,2] = err_w[1,2]
        E[i,3] = err_w_grad[1,2]
        E[i,4] = err_bw3
        
    return E, norm, b_mean


#%%################################################################################################################
###################################################################################################################
###################################################################################################################
###################################################################################################################
#%%------------------------------ MATRIZ DE COV: AUTOVALORES Y AUTOVECTORES --------------------------------------


M = cov(B_vec)
a, u = np.linalg.eig(M) #cada columna de u es un autovector distinto (componentes en cada fila)
#de mayor a menor:
l, y = ordered(a, u)

#fuerzo normal externa (componente en x_MSO positiva)
if y[0,2] < 0: y[:,2] = - y[:,2] 

#como M es simetrica, los autovectores son ortonormales
#pero pueden formar una terna no necesariamente en el orden (x1,x2,x3)
#lo fuerzo respetando que x3 sea normal externa
x = np.array([y[:,0], np.cross(y[:,2],y[:,0]), y[:,2]]) #cada fila es un autovector distinto


##corrijo signos para el ejemplo
#sgn = np.array([-1,-1])
#x = np.array([sgn[0]*y[:,0], sgn[1]*np.cross(y[:,2],y[:,0]), y[:,2]])



# componentes del campo magnetico a lo largo de cada autovector
B1 = np.empty(len(B_vec[:,0]))
B2 = np.empty(len(B_vec[:,0]))
B3 = np.empty(len(B_vec[:,0]))

for m in range(len(B_vec[:,0])):
    B1[m] = np.dot(B_vec[m,:], x[0,:])
    B2[m] = np.dot(B_vec[m,:], x[1,:])
    B3[m] = np.dot(B_vec[m,:], x[2,:])
    

#angulo con Bu
thetaMVA = fcop.alpha(Bu,x[2,:])
   

# errores 

m = len(B_vec[:,0])

#errores de los autovectores (la componente ij es la rotacion del autovector i respecto del j)
err_x = err_vec(m, l) #en radianes
err_x_grad = err_x*180/(np.pi) #en grados

#máximo cono de error para la normal
cono_err_x3 = max(err_x_grad[2,0],err_x_grad[2,1])


#error <Bn> (en unidades de campo magnetico)
err_Bx3 = err_B(l[2], m, err_x[2,1], err_x[2,0], x, np.mean(B_vec, axis = 0))


#%% ESTUDIO Bn

if MODO_mva == 1:
    
    fignum = 0
    figsize = (30,15)
    font_title = 30
    font_label = 30
    font_leg = 26
    lw = 3
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    ticks_l = 6
    ticks_w = 3
    grid_alpha = 0.8
    
    
    plt.figure(fignum, figsize = figsize)
    plt.suptitle(r'Componente normal del campo magnético - MVA', fontsize = font_title)
    plt.plot(t_mag, B3, linewidth = lw, color = colors[0])
    plt.axhline(y = 0, linewidth = lw, linestyle = 'dotted', color = colors[9])
    plt.ylabel(r'$B_n$ [nT]', fontsize = font_label)
    plt.tick_params(axis = 'both', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
    plt.grid(which = 'both', axis = 'both', linewidth = lw, linestyle = '--', alpha = grid_alpha)
    
    plt.savefig(path_analisis+'Bn_mva{}'.format(shock_date))
    plt.savefig(path_analisis+'Bn_mva{}.pdf'.format(shock_date))


#%% HODOGRAMAS

if MODO_mva == 1:
    
    fignum = 0
    figsize = (20,15)
    font_title = 30
    font_label = 30
    font_leg = 26
    lw = 3
    msize = 15
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    ticks_l = 6
    ticks_w = 3
    xtick_spacing = 10
    ytick_spacing = 10
    grid_alpha = 0.8
    
    
    f, (g1, g2) = plt.subplots(1,2, figsize = figsize)
    plt.subplots_adjust(top=0.92, bottom=0.10, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
    
    
    g1.plot(B2, B1, linewidth = lw)
    g1.plot(B2[0], B1[0], 'go', ms = msize, label = r'$t_i$')
    g1.plot(B2[-1], B1[-1], 'gX', ms = msize, label = r'$t_f$')
    g1.axvline(x=0, linewidth = lw, color = 'k')
    g1.axhline(y=0, linewidth = lw, color = 'k')
    g1.set_xlabel(r'$\vec{B} \cdot \vec{x}_2$', size = font_label)
    g1.set_ylabel(r'$\vec{B} \cdot \vec{x}_1$', size = font_label)
    g1.tick_params(axis = 'both', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
    g1.xaxis.set_major_locator(ticker.MultipleLocator(xtick_spacing))
    g1.yaxis.set_major_locator(ticker.MultipleLocator(ytick_spacing))
    g1.legend(loc = 0, fontsize = font_leg)
    g1.grid(which = 'both', axis = 'both', linewidth = lw, linestyle = '--', alpha = grid_alpha)
    
    g2.plot(B3, B1, linewidth = lw, color = 'r')
    g2.plot(B3[0], B1[0], 'go', ms = msize, label = r'$t_i$')
    g2.plot(B3[-1], B1[-1], 'gX', ms = msize, label = r'$t_f$')
    g2.axvline(x=0, linewidth = lw, color = 'k')
    g2.axhline(y=0, linewidth = lw, color = 'k')
    g2.set_xlabel(r'$\vec{B} \cdot \vec{x}_3$', size = font_label)
    g2.set_ylabel(r'$\vec{B} \cdot \vec{x}_1$', size = font_label)
    g2.tick_params(axis = 'both', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
    g2.xaxis.set_major_locator(ticker.MultipleLocator(xtick_spacing))
    g2.yaxis.set_major_locator(ticker.MultipleLocator(ytick_spacing))
    g2.legend(loc = 0, fontsize = font_leg)
    g2.grid(which = 'both', axis = 'both', linewidth = lw, linestyle = '--', alpha = grid_alpha)
    
    
    plt.savefig(path_analisis+'hodogramas_mva{}'.format(shock_date))
    plt.savefig(path_analisis+'hodogramas_mva{}.pdf'.format(shock_date))
    

#%%################################################################################################################
###################################################################################################################
###################################################################################################################
###################################################################################################################
#%%------------------------------ METODO BOOTSTRAP --------------------------------------

E, x3, b_medio = boot(B_vec,1000)

b3 = np.empty(len(x3))
for i in range(len(x3)):
    b3[i] = np.dot(b_medio[i,:],x3[i,:])



# promedios bootstrap y desviaciones estandar
    
b3_av = np.array([np.mean(b3), st.stdev(b3)])
err13_av = np.array([np.mean(E[:,0]), st.stdev(E[:,0])])
err23_av = np.array([np.mean(E[:,2]), st.stdev(E[:,2])])

#normal calculada como el promedio componente a componente normalizado
normal_boot = np.array([np.mean(x3[:,0]), np.mean(x3[:,1]), np.mean(x3[:,2])])/(np.linalg.norm(np.array([np.mean(x3[:,0]), np.mean(x3[:,1]), np.mean(x3[:,2])])))


# histogramas bootstrap

hist_b3, bin_b3 = np.histogram(b3, np.linspace(-6,9,100))
hist_err13, bin_err13 = np.histogram(E[:,0], np.linspace(-0.5,0.5,50))
hist_err23, bin_err23 = np.histogram(E[:,2], np.linspace(-0.3,0.3,50))








