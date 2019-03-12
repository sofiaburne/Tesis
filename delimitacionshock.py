# 0 uso modulo desde otro modulo
# 1 uso modulo y quiero que me haga plots y los guarde
MODO_delimitacion = 0


from mag import B, Bx, By, Bz, x, y, z, t_mag, shock_date
import sweaswia as ss
from sweaswia import t_swea, flujosenergia_swea, counts_swea, nivelesenergia_swea, t_swia_mom, t_swia_spec, densidad_swia, velocidad_swia, velocidad_swia_norm, temperatura_swia, temperatura_swia_norm, flujosenergia_swia, nivelesenergia_swia
import bowshock_funciones as fbow
import coplanaridad_funciones as fcop


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

if MODO_delimitacion == 1:
    
    #subselecciono datos en la orbita de interes
    
    #plot de |B| para determinar orbita del shock de interes
    plt.figure(0, tight_layout = True, figsize = (40,20))
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
counts_swea = counts_swea[iapo1_swea:iapo2_swea,:]
t_swia_mom = t_swia_mom[iapo1_swia_mom:iapo2_swia_mom]
densidad_swia = densidad_swia[iapo1_swia_mom:iapo2_swia_mom]
velocidad_swia = velocidad_swia[iapo1_swia_mom:iapo2_swia_mom,:]
velocidad_swia_norm = velocidad_swia_norm[iapo1_swia_mom:iapo2_swia_mom]
temperatura_swia = temperatura_swia[iapo1_swia_mom:iapo2_swia_mom,:]
temperatura_swia_norm = temperatura_swia_norm[iapo1_swia_mom:iapo2_swia_mom]
t_swia_spec = t_swia_spec[iapo1_swia_spec:iapo2_swia_spec]
flujosenergia_swia = flujosenergia_swia[iapo1_swia_spec:iapo2_swia_spec,:]

#%%

if MODO_delimitacion == 1:
    
    #ploteo magnitudes importantes
    
    figsize = (60,30)
    lw = 1
    font_title = 30
    font_label = 30
    font_leg = 15
    ticks_l = 6
    ticks_w = 3
    grid_alpha = 0.8
    
    
    #selecciono algunas curvas de energia para plotear
    indices_energias_swea_shock, nivelesenergia_swea_shock, flujosenergia_swea_shock = ss.rango_energias_shock(10,1000,100,nivelesenergia_swea,flujosenergia_swea)       
    indices_energias_swia_shock, nivelesenergia_swia_shock, flujosenergia_swia_shock = ss.rango_energias_shock(100,10000,1000,nivelesenergia_swia,flujosenergia_swia)       
    
    
    
    f0, ((p1,p4), (p2,p5), (p3,p6)) = plt.subplots(3,2, sharex = True, figsize = figsize) 
    f0.suptitle('Datos MAVEN {}'.format(shock_date), fontsize = font_title)
    plt.subplots_adjust(top=0.92, bottom=0.10, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
    #f0.tight_layout()
    
    
    #plots de espectros swea
    
    #curvas de flujos 
    #for i in range(len(indices_energias_swea_shock)):
    #    p1.semilogy(t_swea, flujosenergia_swea_shock[:,i], linewidth = lw, label = r'${} eV$'.format(int(nivelesenergia_swea_shock[i])))
    #
    #p1.set_ylabel('Flujo electrones\n[$(cm^2 sr s)^{-1}$]', fontsize = font_label)
    #p1.axes.tick_params(axis = 'both', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
    #p1.axes.grid(axis = 'both', which = 'major', alpha = grid_alpha, linewidth = lw, linestyle = '--')
    #p1.legend(loc = 0, fontsize = font_leg, ncol = 2)
    
    #espectro continuo
    spec1 = p1.contourf(t_swea, nivelesenergia_swea, flujosenergia_swea.T, locator=ticker.LogLocator(), cmap='jet')
    divider = make_axes_locatable(p1)
    cax = divider.append_axes('top', size='5%', pad=0.3)
    f0.colorbar(spec1, cax=cax, orientation='horizontal')
    p1.axes.set_yscale('log')
    p1.set_ylabel('Flujo electrones\n[$(cm^2 sr s)^{-1}$]', fontsize = font_label)
    p1.axes.tick_params(axis = 'both', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
    p1.axes.grid(axis = 'both', which = 'major', alpha = grid_alpha, linewidth = lw, linestyle = '--')
    
    
    
    #plots de espectros swia
    
    #curvas de flujos
    #for i in range(len(indices_energias_swia_shock)):
    #    p2.semilogy(t_swia_spec, flujosenergia_swia_shock[:,i], linewidth = lw, label = r'${} eV$'.format(int(nivelesenergia_swia_shock[i])))
    #
    #p2.set_ylabel('Flujo iones\n[$(cm^2 sr s)^{-1}$]', fontsize = font_label)
    #p2.axes.tick_params(axis = 'both', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
    #p2.axes.grid(axis = 'both', which = 'major', alpha = grid_alpha, linewidth = lw, linestyle = '--')
    #p2.legend(loc = 0, fontsize = font_leg, ncol = 2)
    
    #espectros continuos
    spec2 = p2.contourf(t_swia_spec, nivelesenergia_swia, flujosenergia_swia.T, locator=ticker.LogLocator(), cmap='jet')
    divider = make_axes_locatable(p2)
    cax = divider.append_axes('top', size='5%', pad=0.3)
    f0.colorbar(spec2, cax=cax, orientation='horizontal')
    p2.axes.set_yscale('log')
    p2.set_ylabel('Flujo iones\n[$(cm^2 sr s)^{-1}$]', fontsize = font_label)
    p2.axes.tick_params(axis = 'both', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
    p2.axes.grid(axis = 'both', which = 'major', alpha = grid_alpha, linewidth = lw, linestyle = '--')
    
    
    #plot temperatura swia
    p3.plot(t_swia_mom, temperatura_swia[:,0], linewidth = lw, label = r'$T_x$')
    p3.plot(t_swia_mom, temperatura_swia[:,1], linewidth = lw, label = r'$T_y$')
    p3.plot(t_swia_mom, temperatura_swia[:,2], linewidth = lw, label = r'$T_z$')
    p3.plot(t_swia_mom, temperatura_swia_norm, linewidth = lw, label = r'T')
    p3.set_ylabel('Temperatura iones\n[eV]', fontsize = font_label)
    p3.set_xlabel('Tiempo\n[hora decimal]', fontsize = font_label)
    p3.axes.tick_params(axis = 'both', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
    p3.axes.grid(axis = 'both', which = 'both', alpha = grid_alpha, linewidth = lw, linestyle = '--')
    p3.legend(loc = 0, fontsize = font_leg)
    
    
    #plot densidad swia
    p4.plot(t_swia_mom, densidad_swia, linewidth = lw)
    p4.axes.tick_params(axis = 'both', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
    p4.axes.grid(axis = 'both', which = 'both', alpha = grid_alpha, linewidth = lw, linestyle = '--')
    p4.set_ylabel('$n_p$\n[$cm^{-3}$]', fontsize = font_label)
    
    
    #plot velocidad MSO, modulo y componentes
    p5.plot(t_swia_mom, velocidad_swia[:,0], linewidth = lw, label = r'$V_x$')
    p5.plot(t_swia_mom, velocidad_swia[:,1], linewidth = lw, label = r'$V_y$')
    p5.plot(t_swia_mom, velocidad_swia[:,2], linewidth = lw, label = r'$V_z$')
    p5.plot(t_swia_mom, velocidad_swia_norm, linewidth = lw, label = r'V')
    p5.set_ylabel('Velocidad MSO\n[km/s]', fontsize = font_label)
    p5.axes.tick_params(axis = 'both', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
    p5.axes.grid(axis = 'both', which = 'both', alpha = grid_alpha, linewidth = lw, linestyle = '--')
    p5.legend(loc = 0, fontsize = font_leg)
    
    
    #plots de modulo de B y de sus componentes
    p6.plot(t_mag, Bx, linewidth = lw, label = r'$B_x$')
    p6.plot(t_mag, By, linewidth = lw, label = r'$B_y$')
    p6.plot(t_mag, Bz, linewidth = lw, label = r'$B_z$')
    p6.plot(t_mag, B, linewidth = lw, label = r'B')
    p6.set_xlabel('Tiempo\n[hora decimal]', fontsize = font_label)
    p6.set_ylabel('Campo magnético\n[nT]', fontsize = font_label)
    p6.axes.tick_params(axis = 'both', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
    p6.axes.grid(axis = 'both', which = 'both', alpha = grid_alpha, linewidth = lw, linestyle = '--')
    p6.legend(loc = 5, fontsize = font_leg)
    
    f0.savefig(path_analisis+'datos_MAVEN_{}'.format(shock_date))
    f0.savefig(path_analisis+'datos_MAVEN_{}.pdf'.format(shock_date))
    
    
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

if MODO_delimitacion == 1:
    
    '''
    delimito las regiones up/downsteam y centro del shock
    a partir del perfil del modulo de B, V y la densidad
    '''
    
    figsize = (25,15)
    lw = 1.5
    font_title = 40
    font_label = 35
    font_leg = 25
    ticks_l = 6
    ticks_w = 3
    grid_alpha = 0.8
    updown_alpha = 0.3
    
    
    fig = plt.figure(2, figsize = figsize, tight_layout = True)
    plt.title(r'$\bf{MAVEN,}$ $\bf{MAG,}$ $\bf{SWIA}$  $\bf{Dec}$ $\bf{25,}$ $\bf{2014,}$ $\bf{9:42:00}$ $\bf{-}$ $\bf{10:06:00}$ $\bf{UTC}$', fontsize = font_title)
    host = fig.add_subplot(111)
    par1 = host.twinx()
    #par2 = host.twinx()
    
    host.set_xlabel('Time [hs]', fontsize = font_label)
    host.set_xlim(xmin = 9.7, xmax = 10.1) #*
    host.set_ylabel('$|B_{SW,MSO}|$ [nT]', fontsize = font_label)
    host.set_ylim(ymin = 0, ymax = 70)
    par1.set_ylabel('$|V_{SW,MSO}|$ [km/s]', fontsize = font_label)
    par1.set_ylim(ymin = 0, ymax = 400)
    #par2.set_ylabel('$n_p$\n[$cm^{-3}$]',color = 'C2', fontsize = font_label)
    
    par1.axes.tick_params(axis = 'y', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
    #par2.axes.tick_params(axis = 'y', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
    host.axes.tick_params(axis = 'both', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
    host.axes.grid(axis = 'both', which = 'both', alpha = grid_alpha, linewidth = lw, linestyle = '--')
    
    p1, = host.plot(t_mag, B, linewidth = lw, color = 'C0', label = '$|B_{SW,MSO}|$')
    p2, = par1.plot(t_swia_mom, velocidad_swia_norm, linewidth = lw, color = 'C1', label = '$|V_{SW,MSO}|$')
    p11 = host.axes.axvspan(xmin = t_mag[min(i_u,f_u)], xmax = t_mag[max(i_u,f_u)], facecolor = 'r', alpha = updown_alpha, label = 'Upstream')
    p22 =host.axes.axvspan(xmin = t_mag[min(i_d,f_d)], xmax = t_mag[max(i_d,f_d)], facecolor = 'y', alpha = updown_alpha, label = 'Downstream')
    #p3, = par2.plot(t_swia_mom, densidad_swia, linewidth = lw, color = 'C2', label = '$n_p$')
    
    #lns = [p1, p2, p3]
    lns = [p1, p2, p11, p22]
    host.legend(handles = lns, loc = 6, fontsize = font_leg)
    
    # alineacion tercer eje Y
    #par2.spines['left'].set_position(('outward', 120))      
    #par2.yaxis.set_label_position('left')
    #par2.yaxis.set_ticks_position('left')
    
    fig.savefig(path_analisis+'B_V_densidad_{}'.format(shock_date))
    fig.savefig(path_analisis+'B_V_densidad_{}.pdf'.format(shock_date))

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
#f_u = i_u + f_d - i_d #esto es si quiero que tengan en el mismo ancho up y down (sirve para in/outbound)

#ancho en minutos de los intervalos up/downstream
ancho_up = abs(t_mag[f_u]-t_mag[i_u])*60
ancho_down = abs(t_mag[f_d]-t_mag[i_d])*60
print(ancho_up, ancho_down)

#busco los indices correspondientes para el campo de vel
iu_v = (np.abs(t_mag[i_u]-t_swia_mom)).argmin()
fu_v = (np.abs(t_mag[f_u]-t_swia_mom)).argmin()
id_v = (np.abs(t_mag[i_d]-t_swia_mom)).argmin()
fd_v = (np.abs(t_mag[f_d]-t_swia_mom)).argmin()


#centro del shock (primera estimacion, despues lo refino cuando analizo subestructuras)
tc0 = 9.81947 #*
C0 = (np.abs(t_mag-tc0)).argmin()
#posicion de la nave en el centro del shock
Rc0 = np.array([x[C0], y[C0], z[C0]])


#para variar intervalos up/down

#limites extremos donde encontrar posibles regiones up/down (1 = lim a izq, 2 = lim a der)
lim_t1u = 9.40 #*
lim_t2u = 9.80 #*
lim_t1d = 9.9 #*
lim_t2d = 10.1 #*

#%%

#velocidad de la nave en km/s
v_nave =  vel_nave(x,y,z,t_mag,i_u,f_d)
norm_v_nave = np.linalg.norm(v_nave) #tiene que ser menor a 6 km/s (vel escape de Marte)

#resto velocidad de la nave a mediciones de velocidad
vel = np.empty_like(velocidad_swia)
for i in range(3):
    vel[:,i] = velocidad_swia[:,i] - v_nave[i]

norm_vel = np.sqrt(vel[:,0]**2 + vel[:,1]**2 + vel[:,2]**2)

#%%

#campos en regiones upstream y downstream

B1 = np.array([Bx[min(i_u,f_u):max(i_u,f_u)], By[min(i_u,f_u):max(i_u,f_u)], Bz[min(i_u,f_u):max(i_u,f_u)]]).T
B2 = np.array([Bx[min(i_d,f_d):max(i_d,f_d)], By[min(i_d,f_d):max(i_d,f_d)], Bz[min(i_d,f_d):max(i_d,f_d)]]).T

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


V1 = np.array([vel[min(iu_v,fu_v):max(iu_v,fu_v),0], vel[min(iu_v,fu_v):max(iu_v,fu_v),1], vel[min(iu_v,fu_v):max(iu_v,fu_v),2]]).T
V2 = np.array([vel[min(id_v,fd_v):max(id_v,fd_v),0], vel[min(id_v,fd_v):max(id_v,fd_v),1], vel[min(id_v,fd_v):max(id_v,fd_v),2]]).T

##para el caso de up/down de mismo ancho:
##ojo: por como selecciono los indices para V1 y V2 puede que no queden del mismo tamaño
#if (len(V1)) != (len(V2)): raise ValueError('V1 y V2 tienen distinta cant de elementos')

#vectores Vu Vd
Vu = np.mean(V1, axis = 0)
Vd = np.mean(V2, axis = 0)
std_Vu = np.array([st.stdev(np.float64(V1[:,0])), st.stdev(np.float64(V1[:,1])), st.stdev(np.float64(V1[:,2]))])
std_Vd = np.array([st.stdev(np.float64(V2[:,0])), st.stdev(np.float64(V2[:,1])), st.stdev(np.float64(V2[:,2]))])

#modulos de Vu y Vd
norm_V1 = np.empty_like(V1[:,0])
for i in range(len(V1)):
    norm_V1[i] = np.linalg.norm([V1[i,0], V1[i,1], V1[i,2]])
norm_Vu = np.mean(norm_V1)
std_norm_Vu = st.stdev(np.float64(norm_V1))


norm_V2 = np.empty_like(V2[:,0])
for i in range(len(V2)):
    norm_V2[i] = np.linalg.norm([V2[i,0], V2[i,1], V2[i,2]])
norm_Vd = np.mean(norm_V2)
std_norm_Vd = st.stdev(np.float64(norm_V2))



#pitch angle  (Vu_par a Bu es Vu*cos(pithc) y Vu_per es Vu*sin(pitch), y pitch = arctan(Vu_per/Vu_par))
pitch = fcop.alpha(Vu,Bu)
err_pitch = fcop.err_alpha(Vu, Bu, std_Vu, std_Bu)

#zenith angle (lo vuelvo a calcular cuando analizo subestructuras y mejoro Rc)
cenit0 = fcop.alpha(Rc0,np.array([1,0,0])) #el segundo vector es el versor x_MSO
err_cenit0 = fcop.err_alpha(Rc0, np.array([1,0,0]), np.array([1e-8,1e-8,1e-8]), np.array([1e-8,1e-8,1e-8]))


#%%

if MODO_delimitacion == 1:
    
    '''
    Vuelvo a graficar los datos del shock marcando regiones up/downstram y centro del shock.
    Ahora las velocidades estan medidas en SR shock MSO (reste vel de la nave).
    '''
    
    figsize = (60,30)
    lw = 1.5
    font_title = 40
    font_label = 35
    font_leg = 25
    ticks_l = 6
    ticks_w = 3
    grid_alpha = 0.8
    updown_alpha = 0.3
    xmin = 9.7 #*
    xmax = 10.1 #*
    
    
    
    f1, ((g1,g4), (g2,g5), (g3,g6)) = plt.subplots(3,2, sharex = True, figsize = figsize) #ojo con sharex y los distintos inst
    f1.suptitle(r'$\bf{MAVEN,}$ $\bf{MAG,}$ $\bf{SWEA,}$ $\bf{SWIA}$ $\bf{Dec}$ $\bf{25,}$ $\bf{2014,}$ $\bf{9:42:00}$ $\bf{-}$ $\bf{10:06:00}$ $\bf{UTC}$', fontsize = font_title)
    #plt.subplots_adjust(top=0.92, bottom=0.10, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
    plt.subplots_adjust(top=0.92, bottom=0.10, left=0.10, right=0.95, hspace=0.1, wspace=0.3)
    #f1.tight_layout()
    
    #plots de espectros swea
    
#    #curvas flujos
#    for i in range(len(indices_energias_swea_shock)):
#        g1.semilogy(t_swea, flujosenergia_swea_shock[:,i], linewidth = lw, label = r'${} eV$'.format(int(nivelesenergia_swea_shock[i])))
#    
#    g1.axes.axvspan(xmin = t_mag[i_u], xmax = t_mag[f_u], facecolor = 'r', alpha = updown_alpha, label = 'Upstream')
#    g1.axes.axvspan(xmin = t_mag[i_d], xmax = t_mag[f_d], facecolor = 'y', alpha = updown_alpha, label = 'Downstream')
#    g1.axvline(x = t_mag[C0], linewidth = lw, label = 'Centro choque')
#    g1.set_ylabel('Flujo electrones\n[$(cm^2 sr s)^{-1}$]', fontsize = font_label)
#    g1.axes.tick_params(axis = 'both', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
#    g1.axes.grid(axis = 'both', which = 'major', alpha = grid_alpha, linewidth = lw, linestyle = '--')
#    g1.legend(loc = 0, fontsize = font_leg, ncol = 2)
    
    #espectro continuo
    spec_1 = g1.contourf(t_swea, nivelesenergia_swea, flujosenergia_swea.T, locator=ticker.LogLocator(), cmap='jet')
    #g1.axes.axvspan(xmin = t_mag[min(i_u,f_u)], xmax = t_mag[max(i_u,f_u)], facecolor = 'r', alpha = updown_alpha, label = 'Upstream')
    #g1.axes.axvspan(xmin = t_mag[min(i_d,f_d)], xmax = t_mag[max(i_d,f_d)], facecolor = 'y', alpha = updown_alpha, label = 'Downstream')
    g1.axvline(x = t_mag[C0], linewidth = lw, color = 'k', label = 'Shock center')
    divider = make_axes_locatable(g1)
    cax = divider.append_axes('top', size='5%', pad=1.2)
    cax.tick_params(labelsize = 30)
    cbar = f1.colorbar(spec_1, cax=cax, orientation='horizontal')
    cbar.ax.set_xlabel('differential flux [$(cm^2 sr s)^{-1}$]', fontsize = 30)
    g1.axes.set_yscale('log')
    g1.set_ylabel('$E_{e,SW}$ [eV]', fontsize = font_label)
    g1.axes.tick_params(axis = 'both', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
    g1.axes.grid(axis = 'both', which = 'major', alpha = grid_alpha, linewidth = lw, linestyle = '--')
    g1.set_xlim(xmin = xmin, xmax = xmax)
    g1.legend(loc = 0, fontsize = font_leg)
    
    
    #plots de espectros swia
    
#    #curvas flujos
#    for i in range(len(indices_energias_swia_shock)):
#        g2.semilogy(t_swia_spec, flujosenergia_swia_shock[:,i], linewidth = lw, label = r'${} eV$'.format(int(nivelesenergia_swia_shock[i])))
#    
#    g2.axes.axvspan(xmin = t_mag[i_u], xmax = t_mag[f_u], facecolor = 'r', alpha = updown_alpha, label = 'Upstream')
#    g2.axes.axvspan(xmin = t_mag[i_d], xmax = t_mag[f_d], facecolor = 'y', alpha = updown_alpha, label = 'Downstream')
#    g2.axvline(x = t_mag[C0], linewidth = lw, label = 'Centro choque')
#    g2.set_ylabel('Flujo iones\n[$(cm^2 sr s)^{-1}$]', fontsize = font_label)
#    g2.axes.tick_params(axis = 'both', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
#    g2.axes.grid(axis = 'both', which = 'major', alpha = grid_alpha, linewidth = lw, linestyle = '--')
#    g2.legend(loc = 0, fontsize = font_leg, ncol = 2)
    
    #espectros continuos
    spec_2 = g2.contourf(t_swia_spec, nivelesenergia_swia, flujosenergia_swia.T, locator=ticker.LogLocator(), cmap='jet')
    #g2.axes.axvspan(xmin = t_mag[min(i_u,f_u)], xmax = t_mag[max(i_u,f_u)], facecolor = 'r', alpha = updown_alpha, label = 'Upstream')
    #g2.axes.axvspan(xmin = t_mag[min(i_d,f_d)], xmax = t_mag[max(i_d,f_d)], facecolor = 'y', alpha = updown_alpha, label = 'Downstream')
    g2.axvline(x = t_mag[C0], linewidth = lw, color = 'k', label = 'Shock center')
    divider = make_axes_locatable(g2)
    cax = divider.append_axes('top', size='5%', pad=1.2)
    cax.tick_params(labelsize=font_leg)
    cax.tick_params(labelsize = 30)
    cbar = f1.colorbar(spec_2, cax=cax, orientation='horizontal')
    cbar.ax.set_xlabel('differential flux [$(cm^2 sr s)^{-1}$]', fontsize = 30)
    g2.axes.set_yscale('log')
    g2.set_ylabel('$E_{i,SW}$ [eV]', fontsize = font_label)
    g2.axes.tick_params(axis = 'both', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
    g2.axes.grid(axis = 'both', which = 'major', alpha = grid_alpha, linewidth = lw, linestyle = '--')
    g2.set_xlim(xmin = xmin, xmax = xmax)
    g2.legend(loc = 0, fontsize = font_leg)
    
    
    #plot temperatura swia
    g3.semilogy(t_swia_mom, temperatura_swia[:,0], linewidth = lw, label = r'$T_x$')
    g3.semilogy(t_swia_mom, temperatura_swia[:,1], linewidth = lw, label = r'$T_y$')
    g3.semilogy(t_swia_mom, temperatura_swia[:,2], linewidth = lw, label = r'$T_z$')
    g3.semilogy(t_swia_mom, temperatura_swia_norm, linewidth = lw, label = r'$|T|$')
    g3.axes.axvspan(xmin = t_mag[min(i_u,f_u)], xmax = t_mag[max(i_u,f_u)], facecolor = 'r', alpha = updown_alpha, label = 'Upstream')
    g3.axes.axvspan(xmin = t_mag[min(i_d,f_d)], xmax = t_mag[max(i_d,f_d)], facecolor = 'y', alpha = updown_alpha, label = 'Downstream')
    g3.axvline(x = t_mag[C0], linewidth = lw, color = 'k', label = 'Shock center')
    g3.set_ylabel('$T_{i,SW}$ [eV]', fontsize = font_label)
    #g3.set_xlabel('Time [hs]', fontsize = font_label)
    g3.axes.tick_params(axis = 'both', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
    g3.axes.grid(axis = 'both', which = 'major', alpha = grid_alpha, linewidth = lw, linestyle = '--')
    g3.set_xlim(xmin = xmin, xmax = xmax)
    g3.legend(loc = 4, fontsize = font_leg)
    
    
    #plot densidad swia
    g4.plot(t_swia_mom, densidad_swia, linewidth = lw)
    g4.axes.axvspan(xmin = t_mag[min(i_u,f_u)], xmax = t_mag[max(i_u,f_u)], facecolor = 'r', alpha = updown_alpha, label = 'Upstream')
    g4.axes.axvspan(xmin = t_mag[min(i_d,f_d)], xmax = t_mag[max(i_d,f_d)], facecolor = 'y', alpha = updown_alpha, label = 'Downstream')
    g4.axvline(x = t_mag[C0], linewidth = lw, color = 'k', label = 'Shock center')
    g4.axes.tick_params(axis = 'both', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
    g4.axes.grid(axis = 'both', which = 'both', alpha = grid_alpha, linewidth = lw, linestyle = '--')
    g4.set_ylabel('$n_{i,SW}$ [$cm^{-3}$]', fontsize = font_label)
    g4.set_ylim(0,60)
    g4.set_xlim(xmin = xmin, xmax = xmax)
    g4.legend(loc = 0, fontsize = font_leg)
    
    
    #plot velocidad MSO, modulo y componentes
    g5.plot(t_swia_mom, vel[:,0], linewidth = lw, label = r'$V_x$')
    g5.plot(t_swia_mom, vel[:,1], linewidth = lw, label = r'$V_y$')
    g5.plot(t_swia_mom, vel[:,2], linewidth = lw, label = r'$V_z$')
    g5.plot(t_swia_mom, norm_vel, linewidth = lw, label = r'$|V|$')
    g5.axes.axvspan(xmin = t_mag[min(i_u,f_u)], xmax = t_mag[max(i_u,f_u)], facecolor = 'r', alpha = updown_alpha, label = 'Upstream')
    g5.axes.axvspan(xmin = t_mag[min(i_d,f_d)], xmax = t_mag[max(i_d,f_d)], facecolor = 'y', alpha = updown_alpha, label = 'Downstream')
    g5.axvline(x = t_mag[C0], linewidth = lw, color = 'k', label = 'Shock center')
    g5.set_ylabel('$V_{SW,MSO}$ [km/s]', fontsize = font_label)
    g5.axes.tick_params(axis = 'both', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
    g5.axes.grid(axis = 'both', which = 'both', alpha = grid_alpha, linewidth = lw, linestyle = '--')
    g5.set_xlim(xmin = xmin, xmax = xmax)
    g5.legend(loc = 4, fontsize = font_leg)
    
    
    #plots de modulo de B y de sus componentes
    g6.plot(t_mag, Bx, linewidth = lw, label = r'$B_x$')
    g6.plot(t_mag, By, linewidth = lw, label = r'$B_y$')
    g6.plot(t_mag, Bz, linewidth = lw, label = r'$B_z$')
    g6.plot(t_mag, B, linewidth = lw, label = r'$|B|$')
    g6.axes.axvspan(xmin = t_mag[min(i_u,f_u)], xmax = t_mag[max(i_u,f_u)], facecolor = 'r', alpha = updown_alpha, label = 'Upstream')
    g6.axes.axvspan(xmin = t_mag[min(i_d,f_d)], xmax = t_mag[max(i_d,f_d)], facecolor = 'y', alpha = updown_alpha, label = 'Downstream')
    g6.axvline(x = t_mag[C0], linewidth = lw, color = 'k', label = 'Shock center')
    g6.set_xlabel('Time [hs]', fontsize = font_label)
    g6.set_ylabel('$B_{SW,MSO}$ [nT]', fontsize = font_label)
    g6.axes.tick_params(axis = 'both', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
    g6.axes.grid(axis = 'both', which = 'major', alpha = grid_alpha, linewidth = lw, linestyle = '--')
    g6.set_xlim(xmin = xmin, xmax = xmax)
    g6.set_ylim(ymin = -5, ymax = 45)
    g6.legend(loc = 4, fontsize = font_leg)
    
    f1.savefig(path_analisis+'datos_MAVEN_sombreados_{}'.format(shock_date))
    f1.savefig(path_analisis+'datos_MAVEN_sombreados_{}.pdf'.format(shock_date))


#%%------------------------------- GUARDO RESULTADOS ------------------------------

if MODO_delimitacion == 1:
    
    # parametros que cambio a mano para la delimitacion del shock
    
    datos1 = np.zeros([4,4])
    
    #limites t_mag apoapsides
    datos1[0,0] = t_apo11
    datos1[0,1] = t_apo12
    datos1[0,2] = t_apo21
    datos1[0,3] = t_apo22
    
    #limites t_mag regiones up/downstream
    datos1[1,0] = t_id
    datos1[1,1] = t_fd
    datos1[1,2] = t_iu
    
    #limites t_mag extremos para encontrar regiones up/downstream
    datos1[2,0] = lim_t1u
    datos1[2,1] = lim_t2u
    datos1[2,2] = lim_t1d
    datos1[2,3] = lim_t2d
    #limites t_mag regiones up/down de 5min para variar intervalos
    datos1[3,0] = t_id5
    datos1[3,1] = t_fd5
    datos1[3,2] = t_iu5
    
    #np.savetxt(path_analisis+'parametros_shock_amano_{}'.format(shock_date), datos1, delimiter = '\t',
    #header = '\n'.join(['{}'.format(shock_date),'limites apoapsides',
    #                    'limites regiones up/dowstream',
    #                    'extremos regiones up/downstream',
    #                    'limites regiones up/downstream de 5min para variar int']))
    
    
    # caracteristicas generales del shock
    
    datos2 = np.zeros([15,4])
    
    #tiempos en t_mag del inicio y fin de la orbita del shock
    datos2[0,0] = Tapo1
    datos2[0,1] = Tapo2
    
    #vel de la nave (xyz) y su norma
    datos2[1,0:3] = v_nave
    datos2[1,3] = norm_v_nave
    
    #ancho intervalos down/upstream
    datos2[2,0] = ancho_up
    datos2[2,1] = ancho_down
    
    #Bu y su devstd
    datos2[3,0:3] = Bu
    datos2[4,0:3] = std_Bu
    #modulo de Bu y su devstd
    datos2[5,0] = norm_Bu
    datos2[5,1] = std_norm_Bu
    
    #Bd y su desvstd
    datos2[6,0:3] = Bd
    datos2[7,0:3] = std_Bd
    #modulo de Bd y su devstd
    datos2[8,0] = norm_Bd
    datos2[8,1] = std_norm_Bd
    
    #Vu y su desvstd
    datos2[9,0:3] = Vu
    #datos3[10,:] = std_Vu
    #modulo de Vu y su devstd
    datos2[11,0] = norm_Vu
    #datos3[11,1] = std_norm_Vu
    
    #Vd y su desvstd
    datos2[12,0:3] = Vd
    #datos3[13,:] = std_Vd
    #modulo de Vd y su devstd
    datos2[14,0] = norm_Vd
    #datos3[14,1] = std_norm_Vd
    
    np.savetxt(path_analisis+'caracteristicas_generales_shock_{}'.format(shock_date), datos2, delimiter = '\t',
               header = '\n'.join(['{}'.format(shock_date), 't_mag inicio orbita y fin',
                                                     'vel nave (x,y,z) y su norma [km/s]',
                                                     'ancho intervalo up/downstream [min]',
                                                     'Bu [nT]',
                                                     'desvstd Bu [nT]',
                                                     'modulo Bu y su desvstd',
                                                     'Bd [nT]',
                                                     'desvstd Bd [nT]',
                                                     'modulo Bd y su desvstd',
                                                     'Vu [km/s]',
                                                     'desvstd Vu [km/s]',
                                                     'modulo Vu y su desvstd',
                                                     'Vd [km/s]',
                                                     'desvstd Vd [km/s]',
                                                     'modulo Vd y su desvstd']))
    
