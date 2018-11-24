
#importo mis programitas
from mag import B, Bx, By, Bz, x, y, z, t_mag, shock_date
import swea_swia as ss
from swea_swia import t_swea, flujosenergia_swea, nivelesenergia_swea, t_swia_mom, t_swia_spec, densidad_swia, velocidad_swia, velocidad_swia_norm, temperatura_swia, temperatura_swia_norm, flujosenergia_swia, nivelesenergia_swia
import funciones_coplanaridad as fcop
import funciones_fit_bowshock as fbow
#funciones MVA


from importlib import reload
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as clr
import statistics as st
import os
from sympy.solvers import solve
from sympy import Symbol

#%% 
'''
Carpeta para guardar archivos de resultados
(puede que ya se haya creado al correr mag.py)
ojo: para dia con mas de un shock hacer subcarpetas a mano
'''

path_analisis = r'C:\Users\sofia\Documents\Facultad\Tesis\Analisis/{}/'.format(shock_date)
if not os.path.exists(path_analisis):
    os.makedirs(path_analisis)
    
#%%----------------------------------- FUNCIONES GENERALES -------------------------------------------

#para buscar la posicion de la nave a la entrada/salida de una orbita (posicion mas alejada de Marte)
def orbita(t1,t2,t,x,y,z): #tengo que elegir un rango horario en donde hacer la busqueda: t1 un poco antes del shock de salida anterior y t2 donde ya sepa que la nave esta en la zona upstream
    i1 = (np.abs(t-t1)).argmin() #busca el indice del elemento con valor mas cercano a t1
    i2 = (np.abs(t-t2)).argmin()
    l = np.abs(i1-i2)
    r = np.empty([l]) #array de distancias MAVEN-Marte en el intervalo entre t1 y t2
    for i in range(i1,i2,1):
        r[i-i1] = np.linalg.norm([x[i],y[i],z[i]])
    k = np.argmax(r)
    Ro = np.array([x[k], y[k], z[k]]) #distancia maxima MAVEN-Marte en ese intervalo
    j = k + i1 #indice de datos correspondientes a Ro (en el set de datos completo)
    return Ro, j




#velocidad de la nave en el shock
def vel_nave(x,y,z,t,i_u,f_d):
    '''
    La nave va despacio hasta que entra en las apoapsides,
    asi que la vel de la nave conviene calcularla en el tramo
    upstream+shock+dowstream.
    '''
    #en intervalo upstream+shock+downstream
    #necesito un if por si el shock es de entrada o de salida
    if i_u < f_d:
        xs, ys, zs = x[i_u:f_d], y[i_u:f_d], z[i_u:f_d]
    else:
        xs, ys, zs = x[f_d:i_u], y[f_d:i_u], z[f_d:i_u]
    l = len(xs)
    vx, vy, vz = np.empty(l-1), np.empty(l-1), np.empty(l-1)
    for i in range(l-1):
        #transformo para que me de en km/s (t esta en hora dec y xyz en RM)
        vx[i] = 3390*(xs[i+1]-xs[i])/(3600*(t[i+1]-t[i]))
        vy[i] = 3390*(ys[i+1]-ys[i])/(3600*(t[i+1]-t[i]))
        vz[i] = 3390*(zs[i+1]-zs[i])/(3600*(t[i+1]-t[i]))
    
    v_nave = np.array([np.mean(vx),np.mean(vy),np.mean(vz)])
    return v_nave




#velocidadHTF
def V_HTF(n, vu, Bu):
    V = np.cross(n,np.cross(vu,Bu))/(np.dot(n,Bu))
    # v' en el HTF es v'=v-V
    return V    


#%%#####################################################################################################
########################################################################################################
########################################################################################################
########################################################################################################
#%%------------------------------ PERFILES EN LA ORBITA DEL SHOCK --------------------------------------

#subselecciono datos en la orbita de interes

#plot de |B| para determinar orbita del shock de interes
plt.figure(0, tight_layout = True)
plt.title('Plot para seleccionar apoápsides', fontsize = 20)
plt.plot(t_mag, B, linewidth = 2)
plt.xlabel('Tiempo [hora decimal]', fontsize = 20)
plt.ylabel(r'$|\vec{B}|$ [nT]', fontsize = 20)
plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
plt.grid(axis = 'both', which = 'both', alpha = 0.8, linewidth = 2, linestyle = '--')
plt.legend(loc = 0, fontsize = 15)

#%%

#apoapsides antes y despues del choque marcan la orbita en la que sucede el choque
t_apo11 = 7.24 #* cambiar a mano
t_apo12 = 9.24 #*
t_apo21 = 11.69 #*
t_apo22 = 14.19 #*
R_apo1, iapo1 = orbita(t_apo11,t_apo12,t_mag,x,y,z)
R_apo2, iapo2 = orbita(t_apo21,t_apo22,t_mag,x,y,z)


#busco indices correspondientes a iapo1 y iapo2 de SWIA y SWEA
Tapo1 = t_mag[iapo1]
Tapo2 = t_mag[iapo2]
iapo1_swea = (np.abs(t_swea-Tapo1)).argmin()
iapo2_swea = (np.abs(t_swea-Tapo2)).argmin()
iapo1_swia_mom = (np.abs(t_swia_mom-Tapo1)).argmin()
iapo2_swia_mom = (np.abs(t_swia_mom-Tapo2)).argmin()
iapo1_swia_spec = (np.abs(t_swia_spec-Tapo1)).argmin()
iapo2_swia_spec = (np.abs(t_swia_spec-Tapo2)).argmin()


#me quedo con los datos en el intervalo de interes
B = B[iapo1:iapo2]
Bx = Bx[iapo1:iapo2]
By = By[iapo1:iapo2]
Bz = Bz[iapo1:iapo2]
x = x[iapo1:iapo2]
y = y[iapo1:iapo2]
z = z[iapo1:iapo2]
t_mag = t_mag[iapo1:iapo2]
t_swea = t_swea[iapo1_swea:iapo2_swea]
flujosenergia_swea = flujosenergia_swea[iapo1_swea:iapo2_swea,:]
t_swia_mom = t_swia_mom[iapo1_swia_mom:iapo2_swia_mom]
densidad_swia = densidad_swia[iapo1_swia_mom:iapo2_swia_mom]
velocidad_swia = velocidad_swia[iapo1_swia_mom:iapo2_swia_mom,:]
velocidad_swia_norm = velocidad_swia_norm[iapo1_swia_mom:iapo2_swia_mom]
temperatura_swia = temperatura_swia[iapo1_swia_mom:iapo2_swia_mom,:]
temperatura_swia_norm = temperatura_swia_norm[iapo1_swia_mom:iapo2_swia_mom]
t_swia_spec = t_swia_spec[iapo1_swia_spec:iapo2_swia_spec]
flujosenergia_swia = flujosenergia_swia[iapo1_swia_spec:iapo2_swia_spec,:]

#%%

#ploteo magnitudes importantes

#selecciono algunas curvas de energia para plotear
indices_energias_swea_shock, nivelesenergia_swea_shock, flujosenergia_swea_shock = ss.rango_energias_shock(10,1000,100,nivelesenergia_swea,flujosenergia_swea)       
indices_energias_swia_shock, nivelesenergia_swia_shock, flujosenergia_swia_shock = ss.rango_energias_shock(100,10000,1000,nivelesenergia_swia,flujosenergia_swia)       



f0, ((p1,p4), (p2,p5), (p3,p6)) = plt.subplots(3,2, sharex = True) #ojo con sharex y los distintos inst

f0.suptitle('Datos MAVEN {}'.format(shock_date), fontsize = 20)
f0.tight_layout()

#plots de espectros swea

#curvas de flujos 
#for i in range(len(indices_energias_swea_shock)):
#    p1.semilogy(t_swea, flujosenergia_swea_shock[:,i], linewidth = 2, label = r'${} eV$'.format(int(nivelesenergia_swea_shock[i])))
#
#p1.set_ylabel('Flujo electrones\n[$(cm^2 sr s)^{-1}$]', fontsize = 20)
#p1.axes.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
#p1.axes.grid(axis = 'both', which = 'major', alpha = 0.8, linewidth = 2, linestyle = '--')
#p1.legend(loc = 0, fontsize = 15, ncol = 2)

#espectro continuo
spec1 = p1.contourf(t_swea, nivelesenergia_swea, flujosenergia_swea.T, locator=ticker.LogLocator(), cmap='jet')
divider = make_axes_locatable(p1)
cax = divider.append_axes('top', size='5%', pad=0.3)
f0.colorbar(spec1, cax=cax, orientation='horizontal')
p1.axes.set_yscale('log')
p1.set_ylabel('Flujo electrones\n[$(cm^2 sr s)^{-1}$]', fontsize = 20)
p1.axes.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
p1.axes.grid(axis = 'both', which = 'major', alpha = 0.8, linewidth = 2, linestyle = '--')



#plots de espectros swia

#curvas de flujos
#for i in range(len(indices_energias_swia_shock)):
#    p2.semilogy(t_swia_spec, flujosenergia_swia_shock[:,i], linewidth = 2, label = r'${} eV$'.format(int(nivelesenergia_swia_shock[i])))
#
#p2.set_ylabel('Flujo iones\n[$(cm^2 sr s)^{-1}$]', fontsize = 20)
#p2.axes.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
#p2.axes.grid(axis = 'both', which = 'major', alpha = 0.8, linewidth = 2, linestyle = '--')
#p2.legend(loc = 0, fontsize = 15, ncol = 2)

#espectros continuos
spec2 = p2.contourf(t_swea, nivelesenergia_swea, flujosenergia_swea.T, locator=ticker.LogLocator(), cmap='jet')
divider = make_axes_locatable(p2)
cax = divider.append_axes('top', size='5%', pad=0.3)
f0.colorbar(spec2, cax=cax, orientation='horizontal')
p2.axes.set_yscale('log')
p2.set_ylabel('Flujo iones\n[$(cm^2 sr s)^{-1}$]', fontsize = 20)
p2.axes.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
p2.axes.grid(axis = 'both', which = 'major', alpha = 0.8, linewidth = 2, linestyle = '--')


#plot temperatura swia
p3.plot(t_swia_mom, temperatura_swia[:,0], linewidth = 2, label = r'$T_x$')
p3.plot(t_swia_mom, temperatura_swia[:,1], linewidth = 2, label = r'$T_y$')
p3.plot(t_swia_mom, temperatura_swia[:,2], linewidth = 2, label = r'$T_z$')
p3.plot(t_swia_mom, temperatura_swia_norm, linewidth = 2, label = r'T')
p3.set_ylabel('Temperatura iones\n[eV]', fontsize = 20)
p3.set_xlabel('Tiempo\n[hora decimal]', fontsize = 20)
p3.axes.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
p3.axes.grid(axis = 'both', which = 'both', alpha = 0.8, linewidth = 2, linestyle = '--')
p3.legend(loc = 0, fontsize = 15)


#plot densidad swia
p4.plot(t_swia_mom, densidad_swia, linewidth = 2)
p4.axes.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
p4.axes.grid(axis = 'both', which = 'both', alpha = 0.8, linewidth = 2, linestyle = '--')
p4.set_ylabel('$n_p$\n[$cm^{-3}$]', fontsize = 20)


#plot velocidad MSO, modulo y componentes
p5.plot(t_swia_mom, velocidad_swia[:,0], linewidth = 2, label = r'$V_x$')
p5.plot(t_swia_mom, velocidad_swia[:,1], linewidth = 2, label = r'$V_y$')
p5.plot(t_swia_mom, velocidad_swia[:,2], linewidth = 2, label = r'$V_z$')
p5.plot(t_swia_mom, velocidad_swia_norm, linewidth = 2, label = r'V')
p5.set_ylabel('Velocidad MSO\n[km/s]', fontsize = 20)
p5.axes.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
p5.axes.grid(axis = 'both', which = 'both', alpha = 0.8, linewidth = 2, linestyle = '--')
p5.legend(loc = 0, fontsize = 15)


#plots de modulo de B y de sus componentes
p6.plot(t_mag, Bx, linewidth = 2, label = r'$B_x$')
p6.plot(t_mag, By, linewidth = 2, label = r'$B_y$')
p6.plot(t_mag, Bz, linewidth = 2, label = r'$B_z$')
p6.plot(t_mag, B, linewidth = 2, label = r'B')
p6.set_xlabel('Tiempo\n[hora decimal]', fontsize = 20)
p6.set_ylabel('Campo magnético\n[nT]', fontsize = 20)
p6.axes.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
p6.axes.grid(axis = 'both', which = 'both', alpha = 0.8, linewidth = 2, linestyle = '--')
p6.legend(loc = 0, fontsize = 15)

#f0.savefig(path_analisis+'datos_MAVEN_{}'.format(shock_date))


#para comparar con paper Cesar

#ev11 = (np.abs(nivelesenergia_swea - 11)).argmin()
#ev61 = (np.abs(nivelesenergia_swea - 61)).argmin()
#ev191 = (np.abs(nivelesenergia_swea - 191)).argmin()
#
#plt.figure(200, tight_layout = True)
#plt.semilogy(t_swea, flujosenergia_swea[:,ev11] , linewidth = 2, label = r'${} eV$'.format(int(nivelesenergia_swea[ev11])))
#plt.semilogy(t_swea, flujosenergia_swea[:,ev61] , linewidth = 2, label = r'${} eV$'.format(int(nivelesenergia_swea[ev61])))
#plt.semilogy(t_swea, flujosenergia_swea[:,ev191] , linewidth = 2, label = r'${} eV$'.format(int(nivelesenergia_swea[ev191])))
#plt.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
#plt.ylabel('Flujo electrones\n[$(cm^2 sr s)^{-1}$]', fontsize = 20)
#plt.xlabel('Tiempo\n[hora decimal]', fontsize = 20)
#plt.grid(axis = 'both', which = 'major', alpha = 0.8, linewidth = 2, linestyle = '--')
#plt.legend(loc = 0, fontsize = 15)

#%%#####################################################################################################
########################################################################################################
########################################################################################################
########################################################################################################
#%%------------------------------------- DELIMITACION DEL SHOCK ----------------------------------------

'''
delimito las regiones up/downsteam y centro del shock
a partir del perfil del modulo de B, V y la densidad
'''

fig = plt.figure(1, tight_layout = True)
host = fig.add_subplot(111)
par1 = host.twinx()
par2 = host.twinx()

host.set_xlabel('Tiempo\n[hora decimal]', fontsize = 30)
host.set_ylabel('B\n[nT]', color = 'C0', fontsize = 30)
par1.set_ylabel('$V$ MSO\n[km/s]', color = 'C1', fontsize = 30)
par2.set_ylabel('$n_p$\n[$cm^{-3}$]',color = 'C2', fontsize = 30)

par1.axes.tick_params(axis = 'y', which = 'both', length = 6, width = 3, labelsize = 30)
par2.axes.tick_params(axis = 'y', which = 'both', length = 6, width = 3, labelsize = 30)
host.axes.tick_params(axis = 'both', which = 'both', length = 6, width = 3, labelsize = 30)
host.axes.grid(axis = 'both', which = 'both', alpha = 0.8, linewidth = 2, linestyle = '--')

p1, = host.plot(t_mag, B, linewidth = 3, color = 'C0', label = '$B$')
p2, = par1.plot(t_swia_mom, velocidad_swia_norm, linewidth = 3, color = 'C1', label = '$V$')
p3, = par2.plot(t_swia_mom, densidad_swia, linewidth = 3, color = 'C2', label = '$n_p$')

lns = [p1, p2, p3]
host.legend(handles = lns, loc = 5, fontsize = 30)

# alineacion tercer eje Y
par2.spines['left'].set_position(('outward', 120))      
par2.yaxis.set_label_position('left')
par2.yaxis.set_ticks_position('left')

#fig.savefig(path_analisis+'B_V_densidad_{}'.format(shock_date))

#%%

'''
los limites del shock los marco sobre t_mag, pero
los determino mirando el perfil de densidad, B y vel.
'''

#indices regiones up/dowstream para t_mag
t_id = 9.957 #*
t_fd = 10.1868 #*
t_iu = 9.514 #*
i_d = (np.abs(t_mag-t_id)).argmin()
f_d = (np.abs(t_mag-t_fd)).argmin()
i_u = (np.abs(t_mag-t_iu)).argmin()
f_u = i_u + np.abs(i_d-f_d)

#ancho en minutos de los intervalos up/downstream
ancho_updown = (t_mag[f_u]-t_mag[i_u])*60
print(ancho_updown)

#busco los indices correspondientes para el campo de vel
iu_v = (np.abs(t_mag[i_u]-t_swia_mom)).argmin()
fu_v = (np.abs(t_mag[f_u]-t_swia_mom)).argmin()
id_v = (np.abs(t_mag[i_d]-t_swia_mom)).argmin()
fd_v = (np.abs(t_mag[f_d]-t_swia_mom)).argmin()

#velocidad de la nave en km/s
v_nave =  vel_nave(x,y,z,t_mag,i_u,f_d)
norm_v_nave = np.linalg.norm(v_nave) #tiene que ser menor a 6 km/s (vel escape de Marte)

#ancho temporal del shock en s
t_ancho_temp1 =  9.8127 #*
t_ancho_temp2 =  9.8220 #*
ancho_shock_temp = 3600*abs(t_ancho_temp1 - t_ancho_temp2)
#ancho espacial del shock en km
ancho_shock = ancho_shock_temp*np.array([abs(v_nave[0]), abs(v_nave[1]), abs(v_nave[2])])
norm_ancho_shock = np.linalg.norm(ancho_shock)

#indice centro del shock
tc = 9.8199 #*
C = (np.abs(t_mag-tc)).argmin()
#C = i_u + int((f_d-i_u)/2) #mala forma de determinar el centro
#posicion de la nave en el centro del shock
Rc = np.array([x[C], y[C], z[C]])


#paso a medir velocidades en SR shock
vel = np.empty_like(velocidad_swia)
for i in range(3):
    vel[:,i] = velocidad_swia[:,i] - v_nave[i]

norm_vel = np.sqrt(vel[:,0]**2 + vel[:,1]**2 + vel[:,2]**2)


#campos en regiones upstream y downstream

B1 = np.array([Bx[i_u:f_u], By[i_u:f_u], Bz[i_u:f_u]]).T
B2 = np.array([Bx[i_d:f_d], By[i_d:f_d], Bz[i_d:f_d]]).T

#vectores Bu Bd
Bu = np.mean(B1, axis = 0)
Bd = np.mean(B2, axis = 0)
std_Bu = np.array([st.stdev(B1[:,0]), st.stdev(B1[:,1]), st.stdev(B1[:,2])])
std_Bd = np.array([st.stdev(B2[:,0]), st.stdev(B2[:,1]), st.stdev(B2[:,2])])

#modulos de Bu y Bd
norm_B1 = np.empty_like(B1[:,0])
for i in range(len(B1)):
    norm_B1[i] = np.linalg.norm([B1[i,0], B1[i,1], B1[i,2]])
norm_Bu = np.mean(norm_B1)
std_norm_Bu = st.stdev(norm_B1)

norm_B2 = np.empty_like(B2[:,0])
for i in range(len(B2)):
    norm_B2[i] = np.linalg.norm([B2[i,0], B2[i,1], B2[i,2]])
norm_Bd = np.mean(norm_B2)
std_norm_Bd = st.stdev(norm_B2)


V1 = np.array([vel[iu_v:fu_v,0], vel[iu_v:fu_v,1], vel[iu_v:fu_v,2]]).T
V2 = np.array([vel[id_v:fd_v,0], vel[id_v:fd_v,1], vel[id_v:fd_v,2]]).T

#ojo: por como selecciono los indices para V1 y V2 puede que no queden del mismo tamaño
if (len(V1)) != (len(V2)): print('V1 y V2 tienen distinta cant de elementos')

#vectores Vu Vd
Vu = np.mean(V1, axis = 0)
Vd = np.mean(V2, axis = 0)
#no puedo calcular las std por un tema de que son float32
#std_Vu = np.array([st.stdev(V1[:,0]), st.stdev(V1[:,1]), st.stdev(V1[:,2])])
#std_Vd = np.array([st.stdev(V2[:,0]), st.stdev(V2[:,1]), st.stdev(V2[:,2])])

#modulos de Vu y Vd
norm_V1 = np.empty_like(V1[:,0])
for i in range(len(V1)):
    norm_V1[i] = np.linalg.norm([V1[i,0], V1[i,1], V1[i,2]])
norm_Vu = np.mean(norm_V1)
std_norm_Vu = st.stdev(norm_V1)

norm_V2 = np.empty_like(V2[:,0])
for i in range(len(V2)):
    norm_V2[i] = np.linalg.norm([V2[i,0], V2[i,1], V2[i,2]])
norm_Vd = np.mean(norm_V2)
std_norm_Vd = st.stdev(norm_V2)


#normal del shock reescalando el fit macro del bowshock
L = fbow.L_fit(Rc) #redefino L para que el fit contenga el centro de mi shock y calculo normal del fit
N = fbow.norm_fit_MGS(Rc[0], Rc[1], Rc[2], L)
#angulo entre campo upstream y normal del fit
theta_N = fcop.alpha(Bu,N)
#angulo entre posicion de la nave en el centro del shock y normal del fit
theta_NRc = fcop.alpha(Rc,N)


#para variar intervalos up/down

#limites extremos donde encontrar posibles regiones up/down
lim_t1u = 9 #*
lim_t2u = 9.7799 #*
lim_t1d = 9.96 #*
lim_t2d = 10.3699 #*

#indices regiones up/dowstream para t_mag para intervalos de 5min
t_id5 = 10.065 #*
t_fd5 = 10.1484 #*
t_iu5 = 9.5 #*
i_d5 = (np.abs(t_mag-t_id5)).argmin()
f_d5 = (np.abs(t_mag-t_fd5)).argmin()
i_u5 = (np.abs(t_mag-t_iu5)).argmin()
f_u5 = i_u5 + np.abs(i_d5-f_d5)

#ancho en minutos de los intervalos up/downstream
ancho_updown5 = (t_mag[f_u5]-t_mag[i_u5])*60
print(ancho_updown5)

#busco los indices correspondientes para el campo de vel
iu_v5 = (np.abs(t_mag[i_u5]-t_swia_mom)).argmin()
fu_v5 = (np.abs(t_mag[f_u5]-t_swia_mom)).argmin()
id_v5 = (np.abs(t_mag[i_d5]-t_swia_mom)).argmin()
fd_v5 = (np.abs(t_mag[f_d5]-t_swia_mom)).argmin()

#%%

'''
Vuelvo a graficar los datos del shock marcando regiones up/downstram y centro del shock.
Ahora las velocidades estan medidas en SR shock MSO (reste vel de la nave).
'''

f1, ((g1,g4), (g2,g5), (g3,g6)) = plt.subplots(3,2, sharex = True) #ojo con sharex y los distintos inst

f1.suptitle('Datos MAVEN {}'.format(shock_date), fontsize = 20)
f1.tight_layout()

#plots de espectros swea

#curvas flujos
#for i in range(len(indices_energias_swea_shock)):
#    g1.semilogy(t_swea, flujosenergia_swea_shock[:,i], linewidth = 2, label = r'${} eV$'.format(int(nivelesenergia_swea_shock[i])))
#
#g1.axvline(x = t_mag[C], linewidth = 2, color = 'k', label = 'Shock center')
#g1.axes.axvspan(xmin = t_mag[i_u], xmax = t_mag[f_u], facecolor = 'r', alpha = 0.3, label = 'Upstream')
#g1.axes.axvspan(xmin = t_mag[i_d], xmax = t_mag[f_d], facecolor = 'y', alpha = 0.3, label = 'Downstream')
#g1.set_ylabel('Flujo electrones\n[$(cm^2 sr s)^{-1}$]', fontsize = 20)
#g1.axes.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
#g1.axes.grid(axis = 'both', which = 'major', alpha = 0.8, linewidth = 2, linestyle = '--')
#g1.legend(loc = 0, fontsize = 15, ncol = 2)

#espectro continuo
spec_1 = g1.contourf(t_swea, nivelesenergia_swea, flujosenergia_swea.T, locator=ticker.LogLocator(), cmap='jet')
g1.axvline(x = t_mag[C], linewidth = 2, color = 'k', label = 'Shock center')
g1.axes.axvspan(xmin = t_mag[i_u], xmax = t_mag[f_u], facecolor = 'r', alpha = 0.5, label = 'Upstream')
g1.axes.axvspan(xmin = t_mag[i_d], xmax = t_mag[f_d], facecolor = 'y', alpha = 0.5, label = 'Downstream')
divider = make_axes_locatable(g1)
cax = divider.append_axes('top', size='5%', pad=0.3)
f1.colorbar(spec_1, cax=cax, orientation='horizontal')
g1.axes.set_yscale('log')
g1.set_ylabel('Flujo electrones\n[$(cm^2 sr s)^{-1}$]', fontsize = 20)
g1.axes.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
g1.axes.grid(axis = 'both', which = 'major', alpha = 0.8, linewidth = 2, linestyle = '--')
g1.legend(loc = 0, fontsize = 15)


#plots de espectros swia

#curvas flujos
#for i in range(len(indices_energias_swia_shock)):
#    g2.semilogy(t_swia_spec, flujosenergia_swia_shock[:,i], linewidth = 2, label = r'${} eV$'.format(int(nivelesenergia_swia_shock[i])))
#
#g2.axvline(x = t_mag[C], linewidth = 2, color = 'k', label = 'Shock center')
#g2.axes.axvspan(xmin = t_mag[i_u], xmax = t_mag[f_u], facecolor = 'r', alpha = 0.3, label = 'Upstream')
#g2.axes.axvspan(xmin = t_mag[i_d], xmax = t_mag[f_d], facecolor = 'y', alpha = 0.3, label = 'Downstream')
#g2.set_ylabel('Flujo iones\n[$(cm^2 sr s)^{-1}$]', fontsize = 20)
#g2.axes.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
#g2.axes.grid(axis = 'both', which = 'major', alpha = 0.8, linewidth = 2, linestyle = '--')
#g2.legend(loc = 0, fontsize = 15, ncol = 2)

#espectros continuos
spec_2 = g2.contourf(t_swea, nivelesenergia_swea, flujosenergia_swea.T, locator=ticker.LogLocator(), cmap='jet')
g2.axvline(x = t_mag[C], linewidth = 2, color = 'k', label = 'Shock center')
g2.axes.axvspan(xmin = t_mag[i_u], xmax = t_mag[f_u], facecolor = 'r', alpha = 0.5, label = 'Upstream')
g2.axes.axvspan(xmin = t_mag[i_d], xmax = t_mag[f_d], facecolor = 'y', alpha = 0.5, label = 'Downstream')
divider = make_axes_locatable(g2)
cax = divider.append_axes('top', size='5%', pad=0.3)
f1.colorbar(spec_2, cax=cax, orientation='horizontal')
g2.axes.set_yscale('log')
g2.set_ylabel('Flujo iones\n[$(cm^2 sr s)^{-1}$]', fontsize = 20)
g2.axes.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
g2.axes.grid(axis = 'both', which = 'major', alpha = 0.8, linewidth = 2, linestyle = '--')
g2.legend(loc = 0, fontsize = 15)


#plot temperatura swia
g3.plot(t_swia_mom, temperatura_swia[:,0], linewidth = 2, label = r'$T_x$')
g3.plot(t_swia_mom, temperatura_swia[:,1], linewidth = 2, label = r'$T_y$')
g3.plot(t_swia_mom, temperatura_swia[:,2], linewidth = 2, label = r'$T_z$')
g3.plot(t_swia_mom, temperatura_swia_norm, linewidth = 2, label = r'T')
g3.axvline(x = t_mag[C], linewidth = 2, color = 'k', label = 'Shock center')
g3.axes.axvspan(xmin = t_mag[i_u], xmax = t_mag[f_u], facecolor = 'r', alpha = 0.3, label = 'Upstream')
g3.axes.axvspan(xmin = t_mag[i_d], xmax = t_mag[f_d], facecolor = 'y', alpha = 0.3, label = 'Downstream')
g3.set_ylabel('Temperatura iones\n[eV]', fontsize = 20)
g3.set_xlabel('Tiempo\n[hora decimal]', fontsize = 20)
g3.axes.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
g3.axes.grid(axis = 'both', which = 'both', alpha = 0.8, linewidth = 2, linestyle = '--')
g3.legend(loc = 0, fontsize = 15)


#plot densidad swia
g4.plot(t_swia_mom, densidad_swia, linewidth = 2)
g4.axvline(x = t_mag[C], linewidth = 2, color = 'k', label = 'Shock center')
g4.axes.axvspan(xmin = t_mag[i_u], xmax = t_mag[f_u], facecolor = 'r', alpha = 0.3, label = 'Upstream')
g4.axes.axvspan(xmin = t_mag[i_d], xmax = t_mag[f_d], facecolor = 'y', alpha = 0.3, label = 'Downstream')
g4.axes.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
g4.axes.grid(axis = 'both', which = 'both', alpha = 0.8, linewidth = 2, linestyle = '--')
g4.set_ylabel('$n_p$\n[$cm^{-3}$]', fontsize = 20)
g4.legend(loc = 0, fontsize = 15)


#plot velocidad MSO, modulo y componentes
g5.plot(t_swia_mom, vel[:,0], linewidth = 2, label = r'$V_x$')
g5.plot(t_swia_mom, vel[:,1], linewidth = 2, label = r'$V_y$')
g5.plot(t_swia_mom, vel[:,2], linewidth = 2, label = r'$V_z$')
g5.plot(t_swia_mom, norm_vel, linewidth = 2, label = r'V')
g5.axvline(x = t_mag[C], linewidth = 2, color = 'k', label = 'Shock center')
g5.axes.axvspan(xmin = t_mag[i_u], xmax = t_mag[f_u], facecolor = 'r', alpha = 0.3, label = 'Upstream')
g5.axes.axvspan(xmin = t_mag[i_d], xmax = t_mag[f_d], facecolor = 'y', alpha = 0.3, label = 'Downstream')
g5.set_ylabel('Velocidad MSO\nen referencial shock\n[km/s]', fontsize = 20)
g5.axes.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
g5.axes.grid(axis = 'both', which = 'both', alpha = 0.8, linewidth = 2, linestyle = '--')
g5.legend(loc = 0, fontsize = 15)


#plots de modulo de B y de sus componentes
g6.plot(t_mag, Bx, linewidth = 2, label = r'$B_x$')
g6.plot(t_mag, By, linewidth = 2, label = r'$B_y$')
g6.plot(t_mag, Bz, linewidth = 2, label = r'$B_z$')
g6.plot(t_mag, B, linewidth = 2, label = r'B')
g6.axvline(x = t_mag[C], linewidth = 2, color = 'k', label = 'Shock center')
g6.axes.axvspan(xmin = t_mag[i_u], xmax = t_mag[f_u], facecolor = 'r', alpha = 0.3, label = 'Upstream')
g6.axes.axvspan(xmin = t_mag[i_d], xmax = t_mag[f_d], facecolor = 'y', alpha = 0.3, label = 'Downstream')
g6.set_xlabel('Tiempo\n[hora decimal]', fontsize = 20)
g6.set_ylabel('Campo magnético\n[nT]', fontsize = 20)
g6.axes.tick_params(axis = 'both', which = 'both', length = 4, width = 2, labelsize = 20)
g6.axes.grid(axis = 'both', which = 'both', alpha = 0.8, linewidth = 2, linestyle = '--')
g6.legend(loc = 0, fontsize = 15)

#f1.savefig(path_analisis+'datos_MAVEN_sombreados_{}'.format(shock_date))

#%%#####################################################################################################
########################################################################################################
########################################################################################################
########################################################################################################
#%%---------------------------------------- TESTEO HIPOTESIS MHD ---------------------------------------

#plot de densidad y B para ver que es un shock rapido

f2, plot1 = plt.subplots()

gr1, = plot1.plot(t_mag, B, linewidth = 1, marker ='o', markersize = '2', color = 'C0', label = '$B$')
plot1.set_xlabel('Tiempo\n[hora decimal]', fontsize = 30)
plot1.set_ylabel('B\n[nT]', fontsize = 30)
plt.xlim(t_mag[3020], t_mag[4244])
plot1.axes.tick_params(axis = 'both', which = 'both', length = 6, width = 3, labelsize = 30)
plot1.axes.grid(axis = 'both', which = 'both', alpha = 0.8, linewidth = 2, linestyle = '--')

plot2 = plt.twinx(plot1)
gr2, = plot2.plot(t_swia_mom, densidad_swia, linewidth = 1, marker ='o', markersize = '2',  color = 'C2', label = '$n_p$')
plt.xlim(t_swia_mom[755], t_swia_mom[1060])
plot2.set_ylabel('$n_p$\n[$cm^{-3}$]', fontsize = 30)
plot2.axes.tick_params(axis = 'y', which = 'both', length = 6, width = 3, labelsize = 30)

plot1.legend(handles = [gr1,gr2], loc = 0, fontsize = 20)

#f2.savefig(path_analisis+'fast_shock_{}'.format(shock_date))

#%%

#paso todo lo que use en esta seccion a SI

#elijo respecto a que normal calculo las conservaciones
norm = np.copy(N) #*     ahora elegi la del fit


#chequeo conservaciones en valores medios

#defino campos tang y normales
# vel en m/s
U_u = Vu*(1e3)
U_d = Vd*(1e3)
U_un = np.dot(norm,U_u)
U_dn = np.dot(norm,U_d)
U_ut = (U_u - U_un*norm)
U_dt = (U_d - U_dn*norm)
#B en T
B_u = Bu*(1e-4)
B_d = Bd*(1e-4)
B_un = np.dot(norm,B_u)
B_dn = np.dot(norm,B_d)
B_ut = (B_u - B_un*norm)
B_dt = (B_d - B_dn*norm)

#densidad en kg/m^3
mp = 1.67e-27 #masa del proton en kg
densnum_u = np.mean(densidad_swia[iu_v:fu_v])*(1e-6) #1/m^3
densnum_d = np.mean(densidad_swia[id_v:fd_v])*(1e-6) #1/m^3
rho_u = mp*densnum_u
rho_d = mp*densnum_d

#presion suponiendo gas ideal (en Pa=J/m^3)
kB = 1.38e-23 #cte de Boltzmann en J/K
#por ahora supongo T = 2*Ti, tendria que ser T=Ti+Te
Tu = 2*np.mean(temperatura_swia_norm[iu_v:fu_v])*(8.62e5) #en K
Td = 2*np.mean(temperatura_swia_norm[id_v:fd_v])*(8.62e5) #en K
Pu = densnum_u*kB*Tu
Pd = densnum_d*kB*Td


#numeros de Mach

mu = np.pi*4e-7 #permeabilidad mag del vacio en Wb/Am=mT/A
v_alfv = np.linalg.norm(B_u/np.sqrt(mu*rho_u)) # m/s
v_cs = np.sqrt((Pu/rho_u)*5/3) # m/s

M_A = np.linalg.norm(U_u)/v_alfv
M_cs = np.linalg.norm(U_u)/v_cs
M_f = np.linalg.norm(U_u)/np.sqrt(v_alfv**2 + v_cs**2)

M_c = 2.7 #M_A critico para theta_Bun = 90, para angulos menores decrese

'''
chequeo si con la presion y densidad anterior
se cumple la hipotesis de evolucion adiabatica
(chequeo si gamma da 5/3)
'''
G = Symbol('G')
eq_adiab = Pu*rho_u**G - (Pd*rho_d**G)
gam = solve(eq_adiab, G)

#relaciones RH en porcentaje (100 = se cumple perfectamente)

#conservacion de la masa
cons_masa_u = np.abs(rho_u*U_un)
cons_masa_d = np.abs(rho_d*U_dn)

if cons_masa_u > cons_masa_d:
    cons_masa = cons_masa_d/cons_masa_u*100
else:
    cons_masa = cons_masa_u/cons_masa_d*100    

#consevacion del impulso normal al shock
cons_impul_n_u = np.abs(rho_u*U_un**2 + Pu + B_u**2/(2*mu))
cons_impul_n_d = np.abs(rho_d*U_dn**2 + Pd + B_d**2/(2*mu))

cons_impul_n = np.empty_like(cons_impul_n_u)
for i in range(len(cons_impul_n_u)):
    if cons_impul_n_u[i] > cons_impul_n_d[i]:
        cons_impul_n[i] = cons_impul_n_d[i]/cons_impul_n_u[i]*100
    else:
        cons_impul_n[i] = cons_impul_n_u[i]/cons_impul_n_d[i]*100

#conservacion del impulso tangencial al shock
cons_impul_t_u = np.abs(rho_u*U_un*U_ut - B_un/mu*B_ut)
cons_impul_t_d = np.abs(rho_d*U_dn*U_dt - B_dn/mu*B_dt)

cons_impul_t = np.empty_like(cons_impul_t_u)
for i in range(len(cons_impul_t_u)):
    if cons_impul_t_u[i] > cons_impul_t_d[i]:
        cons_impul_t[i] = cons_impul_t_d[i]/cons_impul_t_u[i]*100
    else:
        cons_impul_t[i] = cons_impul_t_u[i]/cons_impul_t_d[i]*100

#consevacion de la energia
gamma = 5/3
cons_energ_u = np.abs(rho_u*U_un*(1/2*U_u**2 + gamma/(gamma-1)*Pu/rho_u) + U_un*B_u**2/mu - np.dot(U_u,B_u)*B_un/mu)
cons_energ_d = np.abs(rho_d*U_dn*(1/2*U_d**2 + gamma/(gamma-1)*Pd/rho_d) + U_dn*B_d**2/mu - np.dot(U_d,B_d)*B_dn/mu)

cons_energ = np.empty_like(cons_energ_u)
for i in range(len(cons_energ)):
    if cons_energ_u[i] > cons_energ_d[i]:
        cons_energ[i] = cons_energ_d[i]/cons_energ_u[i]*100
    else:
        cons_energ[i] = cons_energ_u[i]/cons_energ_d[i]*100

#conservacion de componente normal de B
cons_Bn_u = np.abs(B_un)
cons_Bn_d = np.abs(B_dn)

if cons_Bn_u > cons_Bn_d:
    cons_Bn = cons_Bn_d/cons_Bn_u*100
else:
    cons_Bn = cons_Bn_u/cons_Bn_d*100

#conservacion de campo electrico tang
cons_Et_u = np.abs(U_un*B_ut - B_un*U_ut)
cons_Et_d = np.abs(U_dn*B_dt - B_dn*U_dt)

cons_Et = np.empty_like(cons_Et_u)
for i in range(len(cons_Et)):
    if cons_Et_u[i] > cons_Et_d[i]:
        cons_Et[i] = cons_Et_d[i]/cons_Et_u[i]*100
    else:
        cons_Et[i] = cons_Et_u[i]/cons_Et_d[i]*100

#hipotesis de coplanaridad
hipt_copl_B = np.dot(norm,np.cross(B_u,B_d))


#%%#####################################################################################################
########################################################################################################
########################################################################################################
########################################################################################################
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

#%%#####################################################################################################
########################################################################################################
########################################################################################################
########################################################################################################
#%%------------------------------- COMPARACION CON FIT MACRO DEL BOWSHOCK ------------------------------

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


#%%#####################################################################################################
########################################################################################################
########################################################################################################
########################################################################################################
#%%------------------------------- GUARDO RESULTADOS ------------------------------


# parametros que cambio a mano, delimitacion del shock

datos1 = np.zeros([6,4])

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
#limites t_mag regiones up/down de 5min para variar intervalos
datos1[5,0] = t_id5
datos1[5,1] = t_fd5
datos1[5,2] = t_iu5

np.savetxt(path_analisis+'parametros_shock_amano_{}'.format(shock_date), datos1, delimiter = '\t',
           header = '\n'.join(['{}'.format(shock_date),'limites apoapsides',
                                                 'limites regiones up/dowstream',
                                                 'limites ancho temporal shock',
                                                 'tiempo centro shock',
                                                 'extremos regiones up/downstream',
                                                 'limites regiones up/downstream de 5min para variar int']))


# caracteristicas generales del shock

datos2 = np.zeros([5,4])

#posisicon de la nave en el centro del shock
datos2[0,0:3] = Rc

#vel de la nave (xyz) y su norma
datos2[1,0:3] = v_nave
datos2[1,3] = norm_v_nave

#ancho temporal del shock
datos2[2,0] = ancho_shock_temp

#ancho espacial del shock y su modulo
datos2[3,0:3] = ancho_shock
datos2[3,3] = norm_ancho_shock

#ancho intervalos down/upstream
datos2[4,0] = ancho_updown

np.savetxt(path_analisis+'caracteristicas_generales_shock_{}'.format(shock_date), datos2, delimiter = '\t',
           header = '\n'.join(['{}'.format(shock_date),'(x,y,z) nave en el centro del shock [RM]',
                                                 'vel nave (x,y,z) y su norma [km/s]',
                                                 'ancho temporal shock [s]',
                                                 'ancho espacial shock (x,y,z) y su norma [km]',
                                                 'ancho intervalo up/downstream [min]']))


# coplanaridad para un sample

datos3 = np.zeros([15,5])

#Bu y su devstd
datos3[0,0:3] = Bu
datos3[1,0:3] = std_Bu
#Bd y su desvstd
datos3[2,0:3] = Bd
datos3[3,0:3] = std_Bd
#Vu y su desvstd
datos3[4,0:3] = Vu
#datos3[5,:] = std_Vu
#Vd y su desvstd
datos3[6,0:3] = Vd
#datos3[7,:] = std_Vd

#normales nB, nBuV, nBdV, nBduV, nV
datos3[8,0:3] = nB 
datos3[9,0:3] = nBuV
datos3[10,0:3] = nBdV
datos3[11,0:3] = nBduV
datos3[12,0:3] = nV

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
           header = '\n'.join(['{}'.format(shock_date),'Bu [nT]',
                                                 'desvstd Bu [nT]',
                                                 'Bd [nT]',
                                                 'desvstd Bd [nT]',
                                                 'Vu [km/s]',
                                                 'desvstd Vu [km/s]',
                                                 'Vd [km/s]',
                                                 'desvstd Vd [km/s]',
                                                 'nB',
                                                 'nBuV',
                                                 'nBdV',
                                                 'nBduV',
                                                 'nV',
                                                 'angulos entre normales y Bu [grados]',
                                                 'angulos entre normales y Rc [grados]']))


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


#testeo hipotesis MHD

datos6 = np.zeros([10,5])

#normal de referencia
datos6[0,0:3] = norm

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
datos6[4,0:3] = cons_impul_n
datos6[5,0:3] = cons_impul_t
datos6[6,0:3] = cons_energ
datos6[7,0] = cons_Bn
datos6[8,0:3] = cons_Et

#hipotesis teo coplanaridad
datos6[9,0] = hipt_copl_B

np.savetxt(path_analisis+'hipotesis_MHD_shock_{}'.format(shock_date), datos6, delimiter = '\t',
           header = '\n'.join(['{}'.format(shock_date),'normal de ref usada en calculos',
                                                 'Tu Td [K] Pu Pd [Pa]',
                                                 'M_Alfv M_sonico M_rapido M_critico',
                                                 'conservacion masa',
                                                 'conservacion impulso norm',
                                                 'conservacion impulso tang',
                                                 'conservacion energia',
                                                 'conservacion Bn',
                                                 'conservacion campo electrico tang',
                                                 'hipotesis teo coplanaridad [nT]']))


#comparacion con fir bowshock macro

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
