import numpy as np
import matplotlib.pyplot as plt
import random
import statistics as st

#%% funciones

#elementos matriz de covarianza 
def cov(x):
    A = np.empty([3,3])
    for i in range(3):
        for j in range(3):
            A[i,j] = np.mean(x[:,i]*x[:,j]) - np.mean(x[:,i])*np.mean(x[:,j])
    return A

#reordenamiento de autovalores y autovectores
def ordered(a, u):
    #autovectores ordenados de menor a mayor (provisorio)
    orden = np.argsort(a) #indices de elementos de a de menor a mayor
    w = [u[:,0], u[:,1], u[:,2]]
    w = np.array([w[i] for i in orden]).T
    #de mayor a menor
    x = np.array([w[:,2], w[:,1], w[:,0]]).T
    l= sorted(a, key = float, reverse = True)
    return(l, x)

#errores autovectores
def err_vec(M, l):
    err_v = np.empty([3,3])
    for i in range(3):
        for j in range(3):
            if i != j:
                err_v[i,j] = np.sqrt(l[2]/(M-1) * (l[i]+l[j]-l[2])/(l[i]-l[j])**2)
            else:
                err_v[i,j] = 0.0 #pongo esto porque la formula del error vale solo para i disntito de j
    return err_v

#error estadistico de la componente normal del campo B
def err_B(l_3, M, err_v32, err_v31, x, B):
    err_B3 = np.sqrt(l_3/(M-1) + (err_v32 * np.dot(B,x[1,:]))**2 + (err_v31 * np.dot(B,x[0,:]))**2)
    return err_B3

#errores metodo bootstrap
def boot(B, N):
    A = np.empty([N,5]) #col de A: err_13 / err_13_grad / err_23 / err_23_grad / err_B3
    b_mean = np.empty([N,3])
    norm = np.empty([N,3])
    
    C = list(B)
    for i in range(N):
        b = [random.choice(C) for _ in range(len(C))]
        b = np.array(b)
        
        b_mean[i,:] = np.mean(b, axis = 0).T
        
        # HAGO MVA AL BOOTSTRAP SAMPLE:
        
        m = cov(b)
        q, w = np.linalg.eig(m) #cada columna de u es un autovector distinto (componentes en cada fila)
        #de mayor a menor:
        q, w = ordered(q, w)
        #armo terna con los autovectores (x3 normal exterior)
        w = np.array([w[:,0],w[:,1],np.cross(w[:,0],w[:,1])]) #cada fila es un autovector distinto
        #renombro todo para no tener mil variables.
        norm[i,:] = w[2,:]
        
        #errores
        err_w = err_vec(len(b[:,0]), q) #en radianes
        err_w_grad = err_w*180/(np.pi)
        err_bw3 = err_B(q[2], len(b[:,0]), err_w[2,1], err_w[2,0], w, np.mean(b, axis = 0))
        
        A[i,0] = err_w[0,2]
        A[i,1] = err_w_grad[0,2]
        A[i,2] = err_w[1,2]
        A[i,3] = err_w_grad[1,2]
        A[i,4] = err_bw3
    return A, norm, b_mean


#%%#########################################################################################################
#########################################################################################################
#%% datos 5:18:20.49 a 5:19:26 Oct 19 1984

path = r'C:\Users\sofia\Documents\Facultad\Tesis\Ejercicio MVA/'
T, B_x, B_y, B_z, v_x, v_y, v_z, N = np.loadtxt(path+'datos.txt', skiprows = 1, unpack = True)
B = np.array([B_x, B_y, B_z]).T
            
#unidades: T(s), B(nT), v(km/s), N(particles/cm^3)

#%% matriz de covarianza: autovalores y autovectores  

M = cov(B)
a, u = np.linalg.eig(M) #cada columna de u es un autovector distinto (componentes en cada fila)
#de mayor a menor:
l, y = ordered(a, u)
#armo terna con los autovectores (x3 normal exterior)
sgn = np.array([-1,-1,1]) #corrijo signos
x = np.array([sgn[0]*y[:,0],sgn[1]*y[:,1],sgn[2]*np.cross(y[:,0],y[:,1])]) #cada fila es un autovector distinto
#chequeo x3 normal exterior o interior:
condicion = 'si' #provisorio (tengo que ver qué me determina la orientacion de X)
if condicion is 'si':
    print('x3 = normal exterior')
else:
    print('x3 = normal interior')


#%% hodogramas

B1 = np.empty(len(B[:,0]))
B2 = np.empty(len(B[:,0]))
B3 = np.empty(len(B[:,0]))

for m in range(len(B[:,0])):
    B1[m] = np.dot(B[m,:], x[0,:])
    B2[m] = np.dot(B[m,:], x[1,:])
    B3[m] = np.dot(B[m,:], x[2,:])

#%% graficos separados
    
#plt.figure(3)
#plt.plot(B2, B1, linewidth = 3)
#plt.xlabel('B2')
#plt.ylabel('B1')
#plt.ylim(-60,60)
#plt.xlim(-60,20)
#
#plt.figure(4)
#plt.plot(B3, B1, linewidth = 3, c = 'r')
#plt.xlabel('B3')
#plt.ylabel('B1')
#plt.ylim(-60,60)
#plt.xlim(-20,20)
#%% graficos juntos pero distintas escalas en x
#
#plt.figure(5)
#
#plt.subplot(121)
#plt.xticks(np.arange(-60,40,20), size=25)
#plt.yticks(np.arange(-60,80,20), size=25)
#plt.xlabel('B2', size=25)
#plt.ylabel('B1', size=25)
#plt.ylim(-60,60)
#plt.xlim(-60,20)
#plt.plot(B2, B1, linewidth = 4)
#plt.axvline(x=0, linewidth = 3, color = 'k')
#plt.axhline(y=0, linewidth = 3, color = 'k')
#
#
#plt.subplot(122)
#plt.xticks(np.arange(-20,40,20), size=25)
#plt.yticks(np.arange(-60,80,20), size=25)
#plt.xlabel('B3', size=25)
#plt.ylabel('B1', size=25)
#plt.ylim(-60,60)
#plt.xlim(-20,20)
#plt.plot(B3, B1, linewidth = 4, c = 'r')
#plt.axvline(x=0, linewidth = 3, color = 'k')
#plt.axhline(y=0, linewidth = 3, color = 'k')
#%% graficos juntos mismas escalas y mismos limites

#plt.figure(5)
#
#g = plt.subplot(121)
#plt.xticks(np.arange(-60,40,20), size=25)
#plt.yticks(np.arange(-60,80,20), size=25)
#plt.ylim(-60,60)
#plt.xlim(-60,20)
#plt.xlabel('B2', size=25)
#plt.ylabel('B1', size=25)
#plt.plot(B2, B1, linewidth = 4)
#plt.axvline(x=0, linewidth = 3, color = 'k')
#plt.axhline(y=0, linewidth = 3, color = 'k')
#
#
#plt.subplot(122, sharex = g, sharey = g)
#plt.xticks(size=25)
#plt.yticks(size=25)
#plt.xlabel('B3', size=25)
#plt.ylabel('B1', size=25)
#plt.plot(B3, B1, linewidth = 4, c = 'r')
#plt.axvline(x=0, linewidth = 3, color = 'k')
#plt.axhline(y=0, linewidth = 3, color = 'k')

#%% graficos juntos con misma escala (calculada a "mano")
x_g1 = np.arange(-60,40,20)
x_g2 = np.arange(-20,40,20) 

f, (g1, g2) = plt.subplots(1,2, gridspec_kw = {'width_ratios':[(len(x_g1)-1)/(len(x_g2)-1), 1]})

g1.plot(B2, B1, linewidth = 4)
g1.plot(B2[0], B1[0], 'go', ms = 15)
g1.plot(B2[-1], B1[-1], 'gX', ms = 15)
g1.axvline(x=0, linewidth = 3, color = 'k')
g1.axhline(y=0, linewidth = 3, color = 'k')
g1.set_xticks(x_g1)
g1.set_xlim(x_g1[0],x_g1[-1])
g1.axes.tick_params('both', labelsize = 25)
g1.set_xlabel('B2', size=25)
g1.set_ylabel('B1', size=25)

g2.plot(B3, B1, linewidth = 4, color = 'r')
g2.plot(B3[0], B1[0], 'go', ms = 15)
g2.plot(B3[-1], B1[-1], 'gX', ms = 15)
g2.axvline(x=0, linewidth = 3, color = 'k')
g2.axhline(y=0, linewidth = 3, color = 'k')
g2.set_xticks(x_g2)
g2.set_xlim(x_g2[0],x_g2[-1])
g2.axes.tick_params('both', labelsize = 25)
g2.set_xlabel('B3', size=25)
g2.set_ylabel('B1', size=25)

#%% errores sin bootstrap

M = len(B[:,0])

err_x = err_vec(M, l) #en radianes
err_x_grad = err_x*180/(np.pi)

err_Bx3 = err_B(l[2], M, err_x[2,1], err_x[2,0], x, np.mean(B, axis = 0))

#%% valores con sus errores

normal = np.empty([3,2])
if err_x_grad[0,2] >= err_x_grad[1,2]:
    normal[:,0] = x[2,:]
    normal[:,1] = err_x[0,2]
    print('normal =', normal)
else:
    normal[:,0] = x[2,:]
    normal[:,1] = err_x_grad[1,2]
    print('normal =', normal)
    
Bx3 = np.array([np.dot(np.mean(B, axis = 0), x[2,:]), err_Bx3])
print('<B>x3 =', Bx3)

#%% angulo entre <B> y la "normal mas grande" corregir!!!!!!!

n_big = normal[:,0] + normal[0,1]*(np.pi)/180 #sumo en radianes
ang = np.arccos(np.dot(n_big,np.mean(B, axis = 0))/(np.linalg.norm(n_big)*np.linalg.norm(np.mean(B, axis = 0))))*180/(np.pi)
#%%########################################################################################################################################################################
###########################################################################################################################################################################
#%% bootstrap
A, x3, b_medio = boot(B,1000)

b3 = np.empty(len(x3))
for i in range(len(x3)):
    b3[i] = np.dot(b_medio[i,:],x3[i,:])

#%% promedios bootstrap y desviaciones estandar

b3_av = np.array([st.mean(b3), st.stdev(b3)])
err13_av = np.array([st.mean(A[:,0]), st.stdev(A[:,0])])
err23_av = np.array([st.mean(A[:,2]), st.stdev(A[:,2])])

#normal calculada como el promedio componente a componente normalizado
normal_boot = np.array([np.mean(x3[:,0]), np.mean(x3[:,1]), np.mean(x3[:,2])])/(np.linalg.norm(np.array([np.mean(x3[:,0]), np.mean(x3[:,1]), np.mean(x3[:,2])])))


#%% histogramas bootstrap

hist_b3, bin_b3 = np.histogram(b3, np.linspace(-6,9,100))
hist_err13, bin_err13 = np.histogram(A[:,0], np.linspace(-0.5,0.5,50))
hist_err23, bin_err23 = np.histogram(A[:,2], np.linspace(-0.3,0.3,50))
#%% graficos histogramas

plt.figure(11)
plt.plot(bin_b3[:-1], hist_b3)
plt.axvline(x=0)

plt.figure(12)
plt.plot(bin_err13[:-1], hist_err13)
plt.axvline(x=0)

plt.figure(13)
plt.plot(bin_err23[:-1], hist_err23)


#%%






