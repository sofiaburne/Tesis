
from mag import B, Bx, By, Bz, x, y, z, t_mag, shock_date
import swea_swia as ss
from swea_swia import t_swea, flujosenergia_swea, nivelesenergia_swea, t_swia_mom, t_swia_spec, densidad_swia, velocidad_swia, velocidad_swia_norm, temperatura_swia, temperatura_swia_norm, flujosenergia_swia, nivelesenergia_swia
import funciones_fit_bowshock as fbow
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
t_id = 9.89378 #*
t_fd = 9.99986 #*
t_iu = 9.7333 #*
t_fu = 9.8116 #*

i_d = (np.abs(t_mag-t_id)).argmin()
f_d = (np.abs(t_mag-t_fd)).argmin()
i_u = (np.abs(t_mag-t_iu)).argmin()
f_u = (np.abs(t_mag-t_fu)).argmin()
#f_u = i_u + np.abs(i_d-f_d) #esto es si quiero que tengan en el mismo ancho up y down

#ancho en minutos de los intervalos up/downstream
ancho_up = (t_mag[f_u]-t_mag[i_u])*60
ancho_down = (t_mag[f_d]-t_mag[i_d])*60
print(ancho_up, ancho_down)

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
#std_norm_Vu = st.stdev(norm_V1)

norm_V2 = np.empty_like(V2[:,0])
for i in range(len(V2)):
    norm_V2[i] = np.linalg.norm([V2[i,0], V2[i,1], V2[i,2]])
norm_Vd = np.mean(norm_V2)
#std_norm_Vd = st.stdev(norm_V2)


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
g5.legend(loc = 4, fontsize = 15)


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
g6.legend(loc = 4, fontsize = 15)

#f1.savefig(path_analisis+'datos_MAVEN_sombreados_{}'.format(shock_date))


#%%------------------------------- GUARDO RESULTADOS ------------------------------

#
## parametros que cambio a mano para la delimitacion del shock
#
#datos1 = np.zeros([6,4])
#
##limites t_mag apoapsides
#datos1[0,0] = t_apo11
#datos1[0,1] = t_apo12
#datos1[0,2] = t_apo21
#datos1[0,3] = t_apo22
#
##limites t_mag regiones up/downstream
#datos1[1,0] = t_id
#datos1[1,1] = t_fd
#datos1[1,2] = t_iu
#
##limites t_mag ancho temporal del shock
#datos1[2,0] = t_ancho_temp1
#datos1[2,1] = t_ancho_temp2
##t_mag centro del shock
#datos1[3,0] = tc
#
##limites t_mag extremos para encontrar regiones up/downstream
#datos1[4,0] = lim_t1u
#datos1[4,1] = lim_t2u
#datos1[4,2] = lim_t1d
#datos1[4,3] = lim_t2d
##limites t_mag regiones up/down de 5min para variar intervalos
#datos1[5,0] = t_id5
#datos1[5,1] = t_fd5
#datos1[5,2] = t_iu5
#
##np.savetxt(path_analisis+'parametros_shock_amano_{}'.format(shock_date), datos1, delimiter = '\t',
##header = '\n'.join(['{}'.format(shock_date),'limites apoapsides',
##                    'limites regiones up/dowstream',
##                    'limites ancho temporal shock',
##                    'tiempo centro shock',
##                    'extremos regiones up/downstream',
##                    'limites regiones up/downstream de 5min para variar int']))
#
#
## caracteristicas generales del shock
#
#datos2 = np.zeros([18,4])
#
##tiempos en t_mag del inicio y fin de la orbita del shock
#datos2[0,0] = Tapo1
#datos2[0,1] = Tapo2
#
##posisicon de la nave en el centro del shock
#datos2[1,0:3] = Rc
#
##vel de la nave (xyz) y su norma
#datos2[2,0:3] = v_nave
#datos2[2,3] = norm_v_nave
#
##ancho temporal del shock
#datos2[3,0] = ancho_shock_temp
#
##ancho espacial del shock y su modulo
#datos2[4,0:3] = ancho_shock
#datos2[4,3] = norm_ancho_shock
#
##ancho intervalos down/upstream
#datos2[5,0] = ancho_updown
#
##Bu y su devstd
#datos2[6,0:3] = Bu
#datos2[7,0:3] = std_Bu
##modulo de Bu y su devstd
#datos2[8,0] = norm_Bu
#datos2[8,1] = std_norm_Bu
#
##Bd y su desvstd
#datos2[9,0:3] = Bd
#datos2[10,0:3] = std_Bd
##modulo de Bd y su devstd
#datos2[11,0] = norm_Bd
#datos2[11,1] = std_norm_Bd
#
##Vu y su desvstd
#datos2[12,0:3] = Vu
##datos3[13,:] = std_Vu
##modulo de Vu y su devstd
#datos2[14,0] = norm_Vu
##datos3[14,1] = std_norm_Vu
#
##Vd y su desvstd
#datos2[15,0:3] = Vd
##datos3[16,:] = std_Vd
##modulo de Vd y su devstd
#datos2[17,0] = norm_Vd
##datos3[17,1] = std_norm_Vd

#np.savetxt(path_analisis+'caracteristicas_generales_shock_{}'.format(shock_date), datos2, delimiter = '\t',
#           header = '\n'.join(['{}'.format(shock_date), 't_mag inicio orbita y fin',
#                                                 '(x,y,z) nave en el centro del shock [RM]',
#                                                 'vel nave (x,y,z) y su norma [km/s]',
#                                                 'ancho temporal shock [s]',
#                                                 'ancho espacial shock (x,y,z) y su norma [km]',
#                                                 'ancho intervalo up/downstream [min]',
#                                                 'Bu [nT]',
#                                                 'desvstd Bu [nT]',
#                                                 'modulo Bu y su desvstd',
#                                                 'Bd [nT]',
#                                                 'desvstd Bd [nT]',
#                                                 'modulo Bd y su desvstd',
#                                                 'Vu [km/s]',
#                                                 'desvstd Vu [km/s]',
#                                                 'modulo Vu y su desvstd',
#                                                 'Vd [km/s]',
#                                                 'desvstd Vd [km/s]',
#                                                 'modulo Vd y su desvstd']))

