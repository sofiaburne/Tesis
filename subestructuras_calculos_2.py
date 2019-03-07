# 0 uso modulo desde otro modulo
# 1 uso modulo y quiero que me haga plots y los guarde
MODO_subestructuras = 1


from mag import shock_date
import delimitacionshock as ds
from delimitacionshock import v_nave, t_swia_mom, densidad_swia, t_swea, nivelesenergia_swea, flujosenergia_swea, t_swia_spec, nivelesenergia_swia, flujosenergia_swia, iu_v, fu_v, id_v, fd_v, Vu, std_Vu, norm_Vu
#from delimitacionshock import t_apo11, t_apo12, t_apo21, t_apo22, x, y, z, B, Bx, By, Bz, t_mag, i_u, f_u, i_d, f_d, t_iu, t_fu, t_id, t_fd, Bu, Bd, norm_Bu, norm_Bd, std_Bu, std_Bd, std_norm_Bu, std_norm_Bd
from delimitacionshock import t_apo11, t_apo12, t_apo21, t_apo22, B, t_mag, x, y, z, t_iu, t_fu, t_id, t_fd, Bu, std_Bu
import bowshock_funciones as fbow
import coplanaridad_funciones as fcop


from importlib import reload
import numpy as np
import statistics as st
import matplotlib.pyplot as plt
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os


#carpeta para guardar resultados
path_analisis = r'C:\Users\sofia\Documents\Facultad\Tesis\Analisis/{}/'.format(shock_date)
if not os.path.exists(path_analisis):
    os.makedirs(path_analisis)
    
    
#%% renombro variables para distinguir alta y baja frecuencia

x_low = x
y_low = y
z_low = z

B_low = B
#Bx_low = Bx
#By_low = By
#Bz_low = Bz

t_mag_low = t_mag

t_apo11_low = t_apo11
t_apo12_low = t_apo12
t_apo21_low = t_apo21
t_apo22_low = t_apo22


#i_u_low = i_u
#f_u_low = f_u
#i_d_low = i_d
#f_d_low = f_d

t_id_low = t_id
t_fd_low = t_fd
t_iu_low = t_iu
t_fu_low = t_fu

Bu_low = Bu
#Bd_low = Bd
#norm_Bu_low = norm_Bu
#norm_Bd_low = norm_Bd
std_Bu_low = std_Bu
#std_Bd_low = std_Bd
#std_norm_Bu_low = std_norm_Bu
#std_norm_Bd_low = std_norm_Bd

#%% datos de alta frecuencia

#importo Bx, By, Bz de alta frecuencia
path_mag_high = r'C:\Users\sofia\Documents\Facultad\Tesis\Datos Maven\MAG\alta_frec/'
d_high, day_frac_high, Bx, By, Bz, X_high, Y_high, Z_high = np.loadtxt(path_mag_high+'2014/12/mvn_mag_l2_2014359pc_20141225_v01_r01.sts', skiprows = 147, usecols = (1,6,7,8,9,11,12,13),unpack = True)

#en radios marcianos
x, y, z = X_high/3390, Y_high/3390, Z_high/3390
#vector de tiempos en hora decimal
t_mag = (day_frac_high-d_high[5])*24
#modulo de B en coordenadas MSO
B = np.sqrt(Bx**2 + By**2 + Bz**2)

#me quedo con los datos de la orbita
R_apo1_high, iapo1_high = ds.orbita(t_apo11_low,t_apo12_low,t_mag,x,y,z)
R_apo2_high, iapo2_high = ds.orbita(t_apo21_low,t_apo22_low,t_mag,x,y,z)

x = x[iapo1_high:iapo2_high]
y = y[iapo1_high:iapo2_high]
z = z[iapo1_high:iapo2_high]

B = B[iapo1_high:iapo2_high]
Bx = Bx[iapo1_high:iapo2_high]
By = By[iapo1_high:iapo2_high]
Bz = Bz[iapo1_high:iapo2_high]

t_mag = t_mag[iapo1_high:iapo2_high]

i_d = (np.abs(t_mag-t_id_low)).argmin()
f_d = (np.abs(t_mag-t_fd_low)).argmin()
i_u = (np.abs(t_mag-t_iu_low)).argmin()
f_u = (np.abs(t_mag-t_fu_low)).argmin()


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




#%% FUNCIONES

def lim_int_ext(lim_izq, lim_der, factor_sigma, sgn = 1,
                Bu = Bu, norm_Bu = norm_Bu, std_Bu = std_Bu, std_norm_Bu = std_norm_Bu,
                t_mag = t_mag, Bx = Bx, By = By, Bz = Bz, B = B):
    
    '''
    Busca los tiempos (y sus indices) en los que Bx, By, Bz y B superan el valor de Bu (o Bd) en
    factor_sigma veces su dispersion. Luego toma como limite a izquierda el menor de esos tiempos,
    y como limite a derecha el mayor.
    '''

    index_Bx = (np.abs(Bx[lim_izq:lim_der] - (Bu[0] + sgn*factor_sigma*abs(std_Bu[0])))).argmin()
    index_By = (np.abs( By[lim_izq:lim_der] - (Bu[1] + sgn*factor_sigma*abs(std_Bu[1])))).argmin()
    index_Bz = (np.abs(Bz[lim_izq:lim_der] - (Bu[2] + sgn*factor_sigma*abs(std_Bu[2])))).argmin()
    index_B = (np.abs(B[lim_izq:lim_der] - (norm_Bu + sgn*factor_sigma*abs(std_norm_Bu)))).argmin()
    
    t_izq = np.min([t_mag[lim_izq+index_Bx], t_mag[lim_izq+index_By], t_mag[lim_izq+index_Bz], t_mag[lim_izq+index_B]])
    t_der = np.max([t_mag[lim_izq+index_Bx], t_mag[lim_izq+index_By], t_mag[lim_izq+index_Bz], t_mag[lim_izq+index_B]])
    index_izq = list(t_mag).index(t_izq)
    index_der = list(t_mag).index(t_der)
    
    return t_izq, t_der, index_izq, index_der




def lim_int_ext_over(lim_izq, lim_der, factor_sigma = 1, factor_Bd = 0.05,
                     t_mag = t_mag, B = B, Bd = Bd, norm_Bd = norm_Bd, std_Bd = std_Bd,
                     std_norm_Bd = std_norm_Bd):
    
    '''
    Busca los tiempos (y sus indices) en los B supera el valor de Bd en factor_sigma veces
    su dispersion y en factor_Bd veces el valor Bd.
    Luego toma como limite a izquierda el menor de esos tiempos y como limite a derecha el mayor.
    '''

    index_sigmaBd = (np.abs(B[lim_izq:lim_der] - (norm_Bd + factor_sigma*abs(std_norm_Bd)))).argmin()
    index_percBd = (np.abs(B[lim_izq:lim_der] - (norm_Bd + factor_Bd*norm_Bd))).argmin()
    
    t_izq = np.min([t_mag[lim_izq+index_sigmaBd], t_mag[lim_izq+index_percBd]])
    t_der = np.max([t_mag[lim_izq+index_sigmaBd], t_mag[lim_izq+index_percBd]])
    index_izq = list(t_mag).index(t_izq)
    index_der = list(t_mag).index(t_der)
    
    return t_izq, t_der, index_izq, index_der




def chired(ydata, ymod, param_fit):
    
    '''
    Calcula el chi cuadrado reducido de un ajuste lineal respecto a los parámetros
    (ej: un ajuste lineal, que tiene param_fit = 2)
    '''
    sigma_sq = np.mean(ydata**2) - (np.mean(ydata))**2
    
    chi = np.sum(((ydata - ymod)**2)/sigma_sq)
    
    chi_red = chi/(len(ydata) - param_fit)
    
    return chi_red
 



def fit_ramp(t_mag, B, i1_ramp_eye, i2_ramp_eye, i1_over_eye, i2_over_eye):
    
    '''
    Para determinar el limite a izquierda del incio de la rampa hago un fit lineal entre
    todos los puntos de la primera mitad del rango elegido a ojo para el incio de la rampa
    y todos los puntos del rango para el inicio del overshoot. Para un dado punto de la región
    "inicio overshoot", barro todos los puntos de la primer mitad de "inicio rampa a ojo" y me
    quedo con el fit que mejor chi cuadrado tenga. Después, elijo el mejor entre todos los fits
    de cada punto de la región "inicio overshoot".
    
    Hago lo mismo para determinar el limite a derecha del inicio de la rampa, pero barriendo la
    segunda mitad de la región "inicio rampa a ojo".
    '''
    
    #t_mag = t_mag
    #B = B
    #i1_ramp_eye = i1_ramp_eye
    #i2_ramp_eye = i2_ramp_eye
    #i1_over_eye = i1_over
    #i2_over_eye = i2_over
    
    i_half = int(abs(i2_ramp_eye - i1_ramp_eye)/2) + i1_ramp_eye
    
    M = len(B[i1_ramp_eye:i_half+1])   
    print('# puntos inicio media rampa: ', M)
    K = len(B[i1_over_eye:i2_over_eye+1])
    print('# puntos fin rampa: ', K)
    
    index1 = np.empty([K])
    params1 = np.empty([K,2])
    err_fit1 = np.empty([K])
    
    index2 = np.empty([K])
    params2 = np.empty([K,2])
    err_fit2 = np.empty([K])
    
    #plt.plot(t_mag, B, linewidth = 2, marker = 'o', markersize = 5)
    ##inicio ramp
    #plt.axvline(x = t_mag[i1_ramp_eye], linewidth = 2, linestyle = '--', color = 'k')
    #plt.axvline(x = t_mag[i2_ramp_eye], linewidth = 2, linestyle = '--', color = 'k')
    ##inicio overshoot
    #plt.axvline(x = ti1_over, linewidth = 2, linestyle = '-', color = 'g')
    #plt.axvline(x = ti2_over, linewidth = 2, linestyle = '-', color = 'g')
    #plt.xlim(9.8194, 9.81957)
    
    for k in range(K):
        
        params1_m = np.empty([M,2])
        err_fit1_m = np.empty([M])
        
        params2_m = np.empty([M,2])
        err_fit2_m = np.empty([M])
        
        for m in range(M):
    
            #si tengo dos puntos para fitear no tiene sentido hacer fit lineal
            if len(B[min(i1_ramp_eye,i1_over_eye)+m:max(i1_ramp_eye,i1_over_eye)+k+1]) < 3:
                print(k,m)
                continue
            else:
                params1_m[m,:] = np.polyfit(t_mag[min(i1_ramp_eye,i1_over_eye)+m:max(i1_ramp_eye,i1_over_eye)+k+1], B[min(i1_ramp_eye,i1_over_eye)+m:max(i1_ramp_eye,i1_over_eye)+k+1], 1, cov = False)
                #defino error del fit a partir del chi cuadrado
                err_fit1_m[m] = chired(B[min(i1_ramp_eye,i1_over_eye)+m:max(i1_ramp_eye,i1_over_eye)+k+1], params1_m[m,0]*t_mag[min(i1_ramp_eye,i1_over_eye)+m:max(i1_ramp_eye,i1_over_eye)+k+1] + params1_m[m,1], 2)
                
                #plt.plot(t_mag[i1_ramp_eye+m:i1_over_eye+k+1], params1_m[m,0]*t_mag[i1_ramp_eye+m:i1_over_eye+k+1]+params1_m[m,1], linewidth = 2, color = 'C{}'.format(m))
                
                
        for m in range(M):
            
            if len(B[min(i_half,i1_over_eye)+m:max(i_half,i1_over_eye)+k+1]) < 3:
                continue
            else:
                params2_m[m,:] = np.polyfit(t_mag[min(i_half,i1_over_eye)+m:max(i_half,i1_over_eye)+k+1], B[min(i_half,i1_over_eye)+m:max(i_half,i1_over_eye)+k+1], 1, cov = False)
                #defino error del fit a partir del chi cuadrado
                err_fit2_m[m] = chired(B[min(i_half,i1_over_eye)+m:max(i_half,i1_over_eye)+k+1], params2_m[m,0]*t_mag[min(i_half,i1_over_eye)+m:max(i_half,i1_over_eye)+k+1] + params2_m[m,1], 2)
                
                #plt.plot(t_mag[i_half+m:i1_over_eye+k+1], params2_m[m,0]*t_mag[i_half+m:i1_over_eye+k+1]+params2_m[m,1], linewidth = 2, color = 'C{}'.format(m+3))
    
        
        #me quedo con el mejor fit para un dado k en cada mitad
       
        index1[k] = (np.abs(err_fit1_m - 1)).argmin()
        params1[k,:] = params1_m[int(index1[k]),:]
        err_fit1[k] = err_fit1_m[int(index1[k])]
        
        index2[k] = (np.abs(err_fit2_m - 1)).argmin()
        params2[k,:] = params2_m[int(index2[k]),:]
        err_fit2[k] = err_fit2_m[int(index2[k])]
    
    
    #me quedo con el mejor fit entre todos los k para cada mitad
    
    index_1 = (np.abs(err_fit1 - 1)).argmin()
    indicei1 =  int(index1[int(index_1)]) + i1_ramp_eye
    
    index_2 = (np.abs(err_fit2 - 1)).argmin()
    indicei2 =  int(index2[int(index_2)]) + i_half
    
    indicef1 = index_1 + i1_over_eye #indice del tiempo (perteneciente a la region inicio overshoot) con el que hice el fit para encontrar el mejor ti1_ramp
    indicef2 = index_2 + i1_over_eye #indice del tiempo (perteneciente a la region inicio overshoot) con el que hice el fit para encontrar el mejor ti2_ramp
        
    
    return indicei1, params1[index_1,:], err_fit1[index_1], indicei2, params2[index_2,:], err_fit2[index_2], indicef1, indicef2




def filter_data(data, fs_new = 8, fs_data = 32):
    
    '''
    Toma data y promedia cada j_av puntos tal que se pasa de tener
    una frecuencia de muestreo de fs_data mediciones por seg a fs_new
    mediciones por seg. Si el tamaño de data no es divisible en j_av
    puntos, entonces elimino los primeros m puntos necesarios para que
    lo sea.
    '''
  
    j_av = int(fs_data/fs_new)  #para tener frec de sampleo: fs_new Hz, tengo que promediar mis datos cada j_av puntos 
    
    if len(data)%j_av == 0:
        data_new = np.mean(data.reshape(-1, j_av), axis=1)
    
    else:
        m = len(data) - j_av*int(len(data)/j_av)
        data_new_ = data[m:]
        data_new = np.mean(data_new_.reshape(-1, j_av), axis=1)
    
    return data_new




def vel_shock(sgn, N, theta_N, theta_NVu, norm_Vu, W_ci, delta_t_foot, percent):
    
    '''
    Velocidad del shock a lo largo de la normal (ref.: Giagkiozis S. 2017).
    Metodo A: Moses 1985
    Metodo B: Gosling & Thomsen 1985
    
    sgn = +1 si outbound
    sgn = -1 si inbound
    
    Las unidades de Vsh serán las mismas que las de Vu.
    delta_t_foot y W_ci deben tener las mismas unidades temporales.
    
    
    Como estas formulas suponen que las estructuras son estacionarias, hago los
    calculos suponiendo que el ancho temporal observado del foot es un X% del
    ancho total (ie, que el foot esta X% desarrollado cuando la nave lo mide).
    '''
    
    delta_t_ft = percent*delta_t_foot
#    #VuN = np.dot(Vu,N)
#    Vu = np.linalg.norm(Vu)
#    theta_N = theta_N*np.pi/180
#    theta_NVu = theta_NVu*np.pi/180
    
    
    #A
    X_L = 0.68 * np.sin(np.deg2rad( theta_N ))**2 / (W_ci * delta_t_ft)
    #Vsh_A = np.linalg.norm(Vu * np.cos(np.deg2rad(theta_NVu)) * X_L / (1 + sgn*X_L))
    Vsh_A = norm_Vu * np.cos(np.deg2rad( theta_NVu )) * X_L / (1 + sgn*X_L)
    
    #B
    t_ot = (1/W_ci)*np.arccos((1 - 2*(np.cos(np.deg2rad( theta_N )))**2) / (2*(np.sin(np.deg2rad( theta_N ))**2)) )
    f = W_ci*t_ot*(2*np.cos(np.deg2rad( theta_N ))**2 - 1) + 2*np.sin(np.deg2rad( theta_N ))**2 * np.sin(np.deg2rad( W_ci*t_ot))
    X_G = f/ (W_ci * delta_t_ft)
    #Vsh_B = np.linalg.norm(Vu * np.cos(theta_NVu) * X_G / (1 + sgn*X_G))
    Vsh_B = norm_Vu * np.cos(np.deg2rad(theta_NVu)) * X_G / (1 + sgn*X_G)
    
    if Vsh_A < 0: print('Vsh_A negativa, el shock se aleja de la nave')
    if Vsh_B < 0: print('Vsh_B negativa, el shock se aleja de la nave')
    
    return abs(Vsh_A), abs(Vsh_B)

#%%

#comparación de sigmas de Bd

std_BdBd = abs(2*np.dot(Bd, std_Bd) + np.dot(std_Bd, std_Bd))

ratio_sig = min(std_BdBd, std_norm_Bd**2)/max(std_BdBd, std_norm_Bd**2)
print('cociente entre sigmas Bd = ', round(ratio_sig,2))

#%%

tipo = 'in' #*

#%%

if MODO_subestructuras == 1:
    
    
    #ploteo para elegir limites a ojo
    
    
    # me genero un set de datos de 8 mediciones/s (una frec intermedia entre 32 y 1)
    B_mid = filter_data(B)
    t_mag_mid = filter_data(t_mag)
    
    
    figsize = (60,30)
    lw = 3
    msize = 8
    font_title = 30
    font_label = 30
    font_leg = 15
    ticks_l = 6
    ticks_w = 3
    grid_alpha = 0.8
    updown_alpha = 0.5
    
    
    f0, ((p3,p1), (p5,p2), (p6,p4)) = plt.subplots(3,2, sharex = True, figsize = figsize) 
    
    f0.suptitle('Datos MAVEN {}'.format(shock_date), fontsize = font_title)
    f0.subplots_adjust(top=0.92, bottom=0.10, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
    
    #B alta frec
    p3.plot(t_mag, B, linewidth = lw, marker = 'o', markersize = msize)
    #regiones up/down
    #p3.axvspan(xmin = t_mag[i_u], xmax = t_mag[f_u], facecolor = 'r', alpha = updown_alpha, label = 'Upstream')
    #p3.axvspan(xmin = t_mag[i_d], xmax = t_mag[f_d], facecolor = 'y', alpha = updown_alpha, label = 'Downstream')
    #asintotas de Bu y Bd
    p3.axhline(y = norm_Bu, linewidth = lw, color = 'r')
    p3.axhline(y = norm_Bd, linewidth = lw, color = 'r')
    p3.axhspan(ymin = norm_Bu - std_norm_Bu, ymax = norm_Bu + std_norm_Bu, facecolor = 'g', alpha = updown_alpha)
    p3.axhspan(ymin = norm_Bd - std_norm_Bd, ymax = norm_Bd + std_norm_Bd, facecolor = 'g', alpha = updown_alpha)
    p3.set_ylabel('$B$ [nT]\n32 Hz', fontsize = font_label)
    p3.axes.tick_params(axis = 'both', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
    p3.axes.grid(axis = 'both', which = 'both', alpha = grid_alpha, linewidth = lw, linestyle = '--')
    p3.legend(loc = 0, fontsize = font_leg)
    
    #B media frec
    p5.plot(t_mag_mid, B_mid, linewidth = lw, marker = 'o', markersize = msize)
    #regiones up/down
    #p5.axvspan(xmin = t_mag[i_u], xmax = t_mag[f_u], facecolor = 'r', alpha = updown_alpha, label = 'Upstream')
    #p5.axvspan(xmin = t_mag[i_d], xmax = t_mag[f_d], facecolor = 'y', alpha = updown_alpha, label = 'Downstream')
    #asintotas de Bu y Bd
    p5.axhline(y = norm_Bu, linewidth = lw, color = 'r')
    p5.axhline(y = norm_Bd, linewidth = lw, color = 'r')
    p5.axhspan(ymin = norm_Bu - std_norm_Bu, ymax = norm_Bu + std_norm_Bu, facecolor = 'g', alpha = updown_alpha)
    p5.axhspan(ymin = norm_Bd - std_norm_Bd, ymax = norm_Bd + std_norm_Bd, facecolor = 'g', alpha = updown_alpha)
    p5.set_ylabel('$B$ [nT]\n8 Hz', fontsize = font_label)
    #p5.set_xlim(9.5,10.5)
    #p5.set_ylim(0,50)
    p5.tick_params(axis = 'both', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
    p5.grid(axis = 'both', which = 'both', alpha = grid_alpha, linewidth = lw, linestyle = '--')
    p5.legend(loc = 0, fontsize = font_leg)
    
    #B baja frec
    p6.plot(t_mag_low, B_low, linewidth = lw, marker = 'o', markersize = msize)
    #regiones up/down
    #p6.axvspan(xmin = t_mag[i_u], xmax = t_mag[f_u], facecolor = 'r', alpha = updown_alpha, label = 'Upstream')
    #p6.axvspan(xmin = t_mag[i_d], xmax = t_mag[f_d], facecolor = 'y', alpha = updown_alpha, label = 'Downstream')
    #asintotas de Bu y Bd
    p6.axhline(y = norm_Bu, linewidth = lw, color = 'r')
    p6.axhline(y = norm_Bd, linewidth = lw, color = 'r')
    p6.set_ylabel('$B$ [nT]\n1 Hz', fontsize = font_label)
    p6.set_xlabel('Tiempo [hora decimal]', fontsize = font_label)
    p6.axes.tick_params(axis = 'both', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
    p6.axes.grid(axis = 'both', which = 'both', alpha = grid_alpha, linewidth = lw, linestyle = '--')
    p6.legend(loc = 0, fontsize = font_leg)
    
    
    #espectros electrones
    spec1 = p1.contourf(t_swea, nivelesenergia_swea, flujosenergia_swea.T, locator=ticker.LogLocator(), cmap='jet')
    divider = make_axes_locatable(p1)
    cax = divider.append_axes('top', size='5%', pad=0.3)
    f0.colorbar(spec1, cax=cax, orientation='horizontal')
    p1.axes.set_yscale('log')
    #regiones up/down
    #p1.axvspan(xmin = t_mag[i_u], xmax = t_mag[f_u], facecolor = 'r', alpha = updown_alpha, label = 'Upstream')
    #p1.axvspan(xmin = t_mag[i_d], xmax = t_mag[f_d], facecolor = 'y', alpha = updown_alpha, label = 'Downstream')
    p1.set_ylabel('Flujo electrones\n[$(cm^2 sr s)^{-1}$]', fontsize = font_label)
    p1.axes.tick_params(axis = 'both', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
    p1.axes.grid(axis = 'both', which = 'major', alpha = grid_alpha, linewidth = lw, linestyle = '--')
    
    
    
    #espectros iones
    spec2 = p2.contourf(t_swia_spec, nivelesenergia_swia, flujosenergia_swia.T, locator=ticker.LogLocator(), cmap='jet')
    divider = make_axes_locatable(p2)
    cax = divider.append_axes('top', size='5%', pad=0.3)
    f0.colorbar(spec2, cax=cax, orientation='horizontal')
    p2.axes.set_yscale('log')
    #regiones up/down
    #p2.axvspan(xmin = t_mag[i_u], xmax = t_mag[f_u], facecolor = 'r', alpha = updown_alpha, label = 'Upstream')
    #p2.axvspan(xmin = t_mag[i_d], xmax = t_mag[f_d], facecolor = 'y', alpha = updown_alpha, label = 'Downstream')
    p2.set_ylabel('Flujo iones\n[$(cm^2 sr s)^{-1}$]', fontsize = font_label)
    p2.axes.tick_params(axis = 'both', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
    p2.axes.grid(axis = 'both', which = 'major', alpha = grid_alpha, linewidth = lw, linestyle = '--')
    
    
    
    #plot densidad swia
    p4.plot(t_swia_mom, densidad_swia, linewidth = lw)
    #regiones up/down
    #p4.axvspan(xmin = t_mag[i_u], xmax = t_mag[f_u], facecolor = 'r', alpha = updown_alpha, label = 'Upstream')
    #p4.axvspan(xmin = t_mag[i_d], xmax = t_mag[f_d], facecolor = 'y', alpha = updown_alpha, label = 'Downstream')
    p4.set_ylabel('$n_p$\n[$cm^{-3}$]', fontsize = font_label)
    p4.set_xlabel('Tiempo [hora decimal]', fontsize = font_label)
    p4.axes.tick_params(axis = 'both', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
    p4.axes.grid(axis = 'both', which = 'both', alpha = grid_alpha, linewidth = lw, linestyle = '--')


#%% limites a ojo


#FOOT      Solo marco el incio del foot (el final lo marco al delimitar el incio de la rampa)

#A izq (der) de Ti1 para shock inbound (outbound) seguro no estoy en el foot.
#A der (izq) de Ti2 para shock inbound (outbound) seguro estoy adentro del foot.
#Entonces el foot empieza en algun lado en entre estos dos limites.
    
#Ojo con los paquetes de onditas sueltos
#ver que las mediciones ya estén por arriba de Bu
#Esto también se puede ver en los perfiles de baja frec

Ti1_foot_eye = 9.81436 #*   lim izq
Ti2_foot_eye = 9.81596 #*   lim der

i1_foot_eye = (np.abs(t_mag - Ti1_foot_eye)).argmin()
i2_foot_eye = (np.abs(t_mag - Ti2_foot_eye)).argmin()


#RAMP      Solo marco el incio de la ramp (el final lo marco al delimitar el incio del overshoot)

#tomar un intervalo con mas de 4 puntos (para elegir entre 2 a izq y 2 a der por lo menos)

Ti1_ramp_eye = 9.819427 #*   lim izq
Ti2_ramp_eye = 9.819458 #*   lim der

i1_ramp_eye = (np.abs(t_mag - Ti1_ramp_eye)).argmin()
i2_ramp_eye = (np.abs(t_mag - Ti2_ramp_eye)).argmin()


#OVERSHOOT

#inicio

#cuando ya estoy por arriba de Bd

Ti1_over_eye = 9.81947 #*   lim izq
Ti2_over_eye = 9.8195 #*    lim der
i1_over_eye = (np.abs(t_mag - Ti1_over_eye)).argmin()
i2_over_eye = (np.abs(t_mag - Ti2_over_eye)).argmin()

#fin

#cuando hay muchos puntos consecutivos por debajo de Bd ya estoy en el undershoot
#Como limite fino del fin del overshoot tratar de seleccionar el punto de inflexión
#entre el ultimo punto del overshoot y el primero del undershoot

Tf1_over_eye = 9.841666 #*   lim izq
Tf2_over_eye = 9.84786 #*    lim der
f1_over_eye = (np.abs(t_mag - Tf1_over_eye)).argmin()
f2_over_eye = (np.abs(t_mag - Tf2_over_eye)).argmin()



#%%

if MODO_subestructuras == 1:
    
    #ploteo para ver si elegi bien los limites a ojo
    
    
    figsize = (60,30)
    lw = 3
    msize = 8
    font_title = 30
    font_label = 30
    font_leg = 15
    ticks_l = 6
    ticks_w = 3
    grid_alpha = 0.8
    updown_alpha = 0.5
    
    
    f1, ((p3,p1), (p5,p2), (p6,p4)) = plt.subplots(3,2, sharex = True, figsize = figsize)
    
    f1.suptitle('Datos MAVEN {}'.format(shock_date), fontsize = font_label)
    f1.subplots_adjust(top=0.92, bottom=0.10, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
    
    #B alta frec
    p3.plot(t_mag, B, linewidth = lw, marker = 'o', markersize = msize)
    #regiones up/down
    #p3.axvspan(xmin = t_mag[i_u], xmax = t_mag[f_u], facecolor = 'r', alpha = updown_alpha, label = 'Upstream')
    #p3.axvspan(xmin = t_mag[i_d], xmax = t_mag[f_d], facecolor = 'y', alpha = updown_alpha, label = 'Downstream')
    #asintotas de Bu y Bd
    p3.axhline(y = norm_Bu, linewidth = lw, color = 'r')
    p3.axhline(y = norm_Bd, linewidth = lw, color = 'r')
    p3.axhspan(ymin = norm_Bu - std_norm_Bu, ymax = norm_Bu + std_norm_Bu, facecolor = 'g', alpha = updown_alpha)
    p3.axhspan(ymin = norm_Bd - std_norm_Bd, ymax = norm_Bd + std_norm_Bd, facecolor = 'g', alpha = updown_alpha)
    #inicio foot
    p3.axvline(x = Ti1_foot_eye, linewidth = lw, linestyle = '--', color = 'k')
    p3.axvline(x = Ti2_foot_eye, linewidth = lw, linestyle = '--', color = 'k')
    #inicio ramp
    p3.axvline(x = Ti1_ramp_eye, linewidth = lw, linestyle = '--', color = 'k')
    p3.axvline(x = Ti2_ramp_eye, linewidth = lw, linestyle = '--', color = 'k')
    #inicio overshoot
    p3.axvline(x = Ti1_over_eye, linewidth = lw, linestyle = '--', color = 'k')
    p3.axvline(x = Ti2_over_eye, linewidth = lw, linestyle = '--', color = 'k')
    #final overshoot
    p3.axvline(x = Tf1_over_eye, linewidth = lw, linestyle = '--', color = 'k')
    p3.axvline(x = Tf2_over_eye, linewidth = lw, linestyle = '--', color = 'k')
    p3.set_ylabel('$B$ [nT]\n32 Hz', fontsize = font_label)
    p3.axes.tick_params(axis = 'both', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
    p3.axes.grid(axis = 'both', which = 'both', alpha = grid_alpha, linewidth = lw, linestyle = '--')
    p3.legend(loc = 0, fontsize = font_leg)
    
    #B media frec
    p5.plot(t_mag_mid, B_mid, linewidth = lw, marker = 'o', markersize = msize)
    #regiones up/down
    #p5.axvspan(xmin = t_mag[i_u], xmax = t_mag[f_u], facecolor = 'r', alpha = updown_alpha, label = 'Upstream')
    #p5.axvspan(xmin = t_mag[i_d], xmax = t_mag[f_d], facecolor = 'y', alpha = updown_alpha, label = 'Downstream')
    #asintotas de Bu y Bd
    p5.axhline(y = norm_Bu, linewidth = lw, color = 'r')
    p5.axhline(y = norm_Bd, linewidth = lw, color = 'r')
    p5.axhspan(ymin = norm_Bu - std_norm_Bu, ymax = norm_Bu + std_norm_Bu, facecolor = 'g', alpha = updown_alpha)
    p5.axhspan(ymin = norm_Bd - std_norm_Bd, ymax = norm_Bd + std_norm_Bd, facecolor = 'g', alpha = updown_alpha)
    #inicio foot
    p5.axvline(x = Ti1_foot_eye, linewidth = lw, linestyle = '--', color = 'k')
    p5.axvline(x = Ti2_foot_eye, linewidth = lw, linestyle = '--', color = 'k')
    #inicio ramp
    p5.axvline(x = Ti1_ramp_eye, linewidth = lw, linestyle = '--', color = 'k')
    p5.axvline(x = Ti2_ramp_eye, linewidth = lw, linestyle = '--', color = 'k')
    #inicio overshoot
    p5.axvline(x = Ti1_over_eye, linewidth = lw, linestyle = '--', color = 'k')
    p5.axvline(x = Ti2_over_eye, linewidth = lw, linestyle = '--', color = 'k')
    #final overshoot
    p5.axvline(x = Tf1_over_eye, linewidth = lw, linestyle = '--', color = 'k')
    p5.axvline(x = Tf2_over_eye, linewidth = lw, linestyle = '--', color = 'k')
    p5.set_ylabel('$B$ [nT]\n8 Hz', fontsize = font_label)
    #p5.set_xlim(9.5,10.5)
    #p5.set_ylim(0,50)
    p5.tick_params(axis = 'both', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
    p5.grid(axis = 'both', which = 'both', alpha = grid_alpha, linewidth = lw, linestyle = '--')
    p5.legend(loc = 0, fontsize = font_leg)
    
    #B baja frec
    p6.plot(t_mag_low, B_low, linewidth = lw, marker = 'o', markersize = msize)
    #regiones up/down
    #p6.axvspan(xmin = t_mag[i_u], xmax = t_mag[f_u], facecolor = 'r', alpha = updown_alpha, label = 'Upstream')
    #p6.axvspan(xmin = t_mag[i_d], xmax = t_mag[f_d], facecolor = 'y', alpha = updown_alpha, label = 'Downstream')
    #inicio foot
    p6.axvline(x = Ti1_foot_eye, linewidth = lw, linestyle = '--', color = 'k')
    p6.axvline(x = Ti2_foot_eye, linewidth = lw, linestyle = '--', color = 'k')
    #inicio ramp
    p6.axvline(x = Ti1_ramp_eye, linewidth = lw, linestyle = '--', color = 'k')
    p6.axvline(x = Ti2_ramp_eye, linewidth = lw, linestyle = '--', color = 'k')
    #inicio overshoot
    p6.axvline(x = Ti1_over_eye, linewidth = lw, linestyle = '--', color = 'k')
    p6.axvline(x = Ti2_over_eye, linewidth = lw, linestyle = '--', color = 'k')
    #final overshoot
    p6.axvline(x = Tf1_over_eye, linewidth = lw, linestyle = '--', color = 'k')
    p6.axvline(x = Tf2_over_eye, linewidth = lw, linestyle = '--', color = 'k')
    #asintotas de Bu y Bd
    p6.axhline(y = norm_Bu, linewidth = lw, color = 'r')
    p6.axhline(y = norm_Bd, linewidth = lw, color = 'r')
    p6.set_ylabel('$B$ [nT]\n1 Hz', fontsize = font_label)
    p6.set_xlabel('Tiempo [hora decimal]', fontsize = font_label)
    p6.axes.tick_params(axis = 'both', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
    p6.axes.grid(axis = 'both', which = 'both', alpha = updown_alpha, linewidth = lw, linestyle = '--')
    p6.legend(loc = 0, fontsize = font_leg)
    
    
    #espectros electrones
    spec1 = p1.contourf(t_swea, nivelesenergia_swea, flujosenergia_swea.T, locator=ticker.LogLocator(), cmap='jet')
    divider = make_axes_locatable(p1)
    cax = divider.append_axes('top', size='5%', pad=0.3)
    f1.colorbar(spec1, cax=cax, orientation='horizontal')
    p1.axes.set_yscale('log')
    #regiones up/down
    #p1.axvspan(xmin = t_mag[i_u], xmax = t_mag[f_u], facecolor = 'r', alpha = updown_alpha, label = 'Upstream')
    #p1.axvspan(xmin = t_mag[i_d], xmax = t_mag[f_d], facecolor = 'y', alpha = updown_alpha, label = 'Downstream')
    #inicio foot
    p1.axvline(x = Ti1_foot_eye, linewidth = lw, linestyle = '--', color = 'k')
    p1.axvline(x = Ti2_foot_eye, linewidth = lw, linestyle = '--', color = 'k')
    #inicio ramp
    p1.axvline(x = Ti1_ramp_eye, linewidth = lw, linestyle = '--', color = 'k')
    p1.axvline(x = Ti2_ramp_eye, linewidth = lw, linestyle = '--', color = 'k')
    #inicio overshoot
    p1.axvline(x = Ti1_over_eye, linewidth = lw, linestyle = '--', color = 'k')
    p1.axvline(x = Ti2_over_eye, linewidth = lw, linestyle = '--', color = 'k')
    #final overshoot
    p1.axvline(x = Tf1_over_eye, linewidth = lw, linestyle = '--', color = 'k')
    p1.axvline(x = Tf2_over_eye, linewidth = lw, linestyle = '--', color = 'k')
    p1.set_ylabel('Flujo electrones\n[$(cm^2 sr s)^{-1}$]', fontsize = font_label)
    p1.axes.tick_params(axis = 'both', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
    p1.axes.grid(axis = 'both', which = 'major', alpha = grid_alpha, linewidth = lw, linestyle = '--')
    
    
    
    #espectros iones
    spec2 = p2.contourf(t_swia_spec, nivelesenergia_swia, flujosenergia_swia.T, locator=ticker.LogLocator(), cmap='jet')
    divider = make_axes_locatable(p2)
    cax = divider.append_axes('top', size='5%', pad=0.3)
    f1.colorbar(spec2, cax=cax, orientation='horizontal')
    p2.axes.set_yscale('log')
    #regiones up/down
    #p2.axvspan(xmin = t_mag[i_u], xmax = t_mag[f_u], facecolor = 'r', alpha = updown_alpha, label = 'Upstream')
    #p2.axvspan(xmin = t_mag[i_d], xmax = t_mag[f_d], facecolor = 'y', alpha = updown_alpha, label = 'Downstream')
    #inicio foot
    p2.axvline(x = Ti1_foot_eye, linewidth = lw, linestyle = '--', color = 'k')
    p2.axvline(x = Ti2_foot_eye, linewidth = lw, linestyle = '--', color = 'k')
    #inicio ramp
    p2.axvline(x = Ti1_ramp_eye, linewidth = lw, linestyle = '--', color = 'k')
    p2.axvline(x = Ti2_ramp_eye, linewidth = lw, linestyle = '--', color = 'k')
    #inicio overshoot
    p2.axvline(x = Ti1_over_eye, linewidth = lw, linestyle = '--', color = 'k')
    p2.axvline(x = Ti2_over_eye, linewidth = lw, linestyle = '--', color = 'k')
    #final overshoot
    p2.axvline(x = Tf1_over_eye, linewidth = lw, linestyle = '--', color = 'k')
    p2.axvline(x = Tf2_over_eye, linewidth = lw, linestyle = '--', color = 'k')
    p2.set_ylabel('Flujo iones\n[$(cm^2 sr s)^{-1}$]', fontsize = font_label)
    p2.axes.tick_params(axis = 'both', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
    p2.axes.grid(axis = 'both', which = 'major', alpha = grid_alpha, linewidth = lw, linestyle = '--')
    
    
    
    #plot densidad swia
    p4.plot(t_swia_mom, densidad_swia, linewidth = lw)
    #regiones up/down
    #p4.axvspan(xmin = t_mag[i_u], xmax = t_mag[f_u], facecolor = 'r', alpha = updown_alpha, label = 'Upstream')
    #p4.axvspan(xmin = t_mag[i_d], xmax = t_mag[f_d], facecolor = 'y', alpha = updown_alpha, label = 'Downstream')
    #inicio foot
    p4.axvline(x = Ti1_foot_eye, linewidth = lw, linestyle = '--', color = 'k')
    p4.axvline(x = Ti2_foot_eye, linewidth = lw, linestyle = '--', color = 'k')
    #inicio ramp
    p4.axvline(x = Ti1_ramp_eye, linewidth = lw, linestyle = '--', color = 'k')
    p4.axvline(x = Ti2_ramp_eye, linewidth = lw, linestyle = '--', color = 'k')
    #inicio overshoot
    p4.axvline(x = Ti1_over_eye, linewidth = lw, linestyle = '--', color = 'k')
    p4.axvline(x = Ti2_over_eye, linewidth = lw, linestyle = '--', color = 'k')
    #final overshoot
    p4.axvline(x = Tf1_over_eye, linewidth = lw, linestyle = '--', color = 'k')
    p4.axvline(x = Tf2_over_eye, linewidth = lw, linestyle = '--', color = 'k')
    p4.set_ylabel('$n_p$\n[$cm^{-3}$]', fontsize = font_label)
    p4.set_xlabel('Tiempo [hora decimal]', fontsize = font_label)
    p4.axes.tick_params(axis = 'both', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
    p4.axes.grid(axis = 'both', which = 'both', alpha = grid_alpha, linewidth = lw, linestyle = '--')


#%% automatizo delimitaciones


#FOOT inicio

#como Bu varia poco, miro variaciones de 3 sigmas

fact_sigma_foot = 3
ti1_foot, ti2_foot, i1_foot, i2_foot = lim_int_ext(lim_izq = i1_foot_eye, lim_der = i2_foot_eye, factor_sigma = fact_sigma_foot)
i_foot = int(np.abs(i2_foot - i1_foot)/2) + i1_foot
ti_foot = t_mag[i_foot]


#OVERSHOOT

#inicio

#como Bd varia bastante, considero variaciones de 1 sigma

#haciendolo asi sobre-estimo:
#fact_sigma_iover = 1
#ti1_over, ti2_over, i1_over, i2_over = lim_int_ext(Bd, norm_Bd, std_Bd, std_norm_Bd, i1_over_eye, i2_over_eye, fact_sigma_iover)
#i_over = int(np.abs(i2_over - i1_over)/2) + i1_over
#ti_over = t_mag[i_over]

ti1_over, ti2_over, i1_over, i2_over = lim_int_ext_over(lim_izq = i1_over_eye, lim_der = i2_over_eye)
i_over = int(np.abs(i2_over - i1_over)/2) + i1_over
ti_over = t_mag[i_over]

#final

fact_sigma_fover = 1
#pongo sgn -1 para ver cuando estoy por debajo de cierto umbral de Bd
tf1_over, tf2_over, f1_over, f2_over = lim_int_ext(f1_over_eye, f2_over_eye, fact_sigma_fover, - 1, Bd, norm_Bd, std_Bd, std_norm_Bd)
f_over = int(np.abs(f2_over - f1_over)/2) + f1_over
tf_over = t_mag[f_over]

#amplitud

ind_max_over = (abs(B - max(B[min(i1_over_eye,f1_over_eye):max(i1_over_eye,f1_over_eye)]))).argmin()
Bmax_over = B[ind_max_over]
amp_over = (Bmax_over - norm_Bd)/norm_Bd


#inicio RAMP

i1_ramp, i1_params, i1_chired, i2_ramp, i2_params, i2_chired, f1_rampfit, f2_rampfit = fit_ramp(t_mag, B, i1_ramp_eye, i2_ramp_eye, i1_over, i2_over)
ti1_ramp = t_mag[i1_ramp]
ti2_ramp = t_mag[i2_ramp]
i_ramp = int(np.abs(i2_ramp - i1_ramp)/2) + i1_ramp
ti_ramp = t_mag[i_ramp]

#centro del shock (centro de la rampa)
tc = abs(ti_over - ti_ramp)/2 + np.min([ti_ramp,ti_over])
C_high = (np.abs(t_mag-tc)).argmin()


#anchos temporales (en s)

#foot
ancho_foot_temp = 3600*abs(ti_foot - ti_ramp)
#ramp
ancho_ramp_temp = 3600*abs(ti_ramp - ti_over)
#overshoot
ancho_over_temp = 3600*abs(ti_over - tf_over)
#shock (foot + ramp)
ancho_shock_temp = 3600*abs(ti_foot - ti_over)

#%%

if MODO_subestructuras == 1:
    
    #ploteo todos los limites determinados (a ojo y automatizados)
    
    figsize = (30,15)
    lw = 1
    msize = 8
    font_title = 30
    font_label = 30
    font_leg = 15
    ticks_l = 6
    ticks_w = 3
    grid_alpha = 0.8
    updown_alpha = 0.5
    
    
    plt.figure(5, figsize = figsize)
    plt.title('Delimitacion de subestructuras', fontsize = font_label)
    
    plt.plot(t_mag, B, linewidth = lw, marker = 'o', markersize = msize)
    #regiones up/down
    #plt.axvspan(xmin = t_mag[i_u], xmax = t_mag[f_u], facecolor = 'r', alpha = updown_alpha, label = 'Upstream')
    #plt.axvspan(xmin = t_mag[i_d], xmax = t_mag[f_d], facecolor = 'y', alpha = updown_alpha, label = 'Downstream')
    #asintotas de Bu y Bd
    plt.axhline(y = norm_Bu, linewidth = lw, color = 'r')
    plt.axhline(y = norm_Bd, linewidth = lw, color = 'r')
    plt.axhspan(ymin = norm_Bu - std_norm_Bu, ymax = norm_Bu + std_norm_Bu, facecolor = 'g', alpha = updown_alpha)
    plt.axhspan(ymin = norm_Bd - std_norm_Bd, ymax = norm_Bd + std_norm_Bd, facecolor = 'g', alpha = updown_alpha)
    #inicio foot
    plt.axvline(x = Ti1_foot_eye, linewidth = lw, linestyle = '--', color = 'k')
    plt.axvline(x = Ti2_foot_eye, linewidth = lw, linestyle = '--', color = 'k')
    plt.axvline(x = ti1_foot, linewidth = lw, linestyle = '-', color = 'm')
    plt.axvline(x = ti2_foot, linewidth = lw, linestyle = '-', color = 'm')
    #inicio ramp
    plt.axvline(x = Ti1_ramp_eye, linewidth = lw, linestyle = '--', color = 'k')
    plt.axvline(x = Ti2_ramp_eye, linewidth = lw, linestyle = '--', color = 'k')
    plt.axvline(x = ti1_ramp, linewidth = lw, linestyle = '-', color = 'c')
    plt.axvline(x = ti2_ramp, linewidth = lw, linestyle = '-', color = 'c')
    #ajustes ramp
    plt.plot(t_mag[min(i1_ramp,f1_rampfit):max(i1_ramp,f1_rampfit)+1], i1_params[0]*t_mag[min(i1_ramp,f1_rampfit):max(i1_ramp,f1_rampfit)+1] + i1_params[1], linewidth = lw, color = 'y', linestyle = '-.')
    plt.plot(t_mag[min(i2_ramp,f2_rampfit):max(i2_ramp,f2_rampfit)+1], i2_params[0]*t_mag[min(i2_ramp,f2_rampfit):max(i2_ramp,f2_rampfit)+1] + i2_params[1], linewidth = lw, color = 'r', linestyle = '-.')
    plt.axvline(x = t_mag[C_high], linewidth = lw, linestyle = 'dotted', color = 'orange', label = 'Centro del choque')
    #inicio overshoot
    plt.axvline(x = Ti1_over_eye, linewidth = lw, linestyle = '--', color = 'k')
    plt.axvline(x = Ti2_over_eye, linewidth = lw, linestyle = '--', color = 'k')
    plt.axvline(x = ti1_over, linewidth = lw, linestyle = '-', color = 'g')
    plt.axvline(x = ti2_over, linewidth = lw, linestyle = '-', color = 'g')
    plt.axhline(y = Bmax_over, linewidth = lw, linestyle = '-', color = 'purple')
    #final overshoot
    plt.axvline(x = Tf1_over_eye, linewidth = lw, linestyle = '--', color = 'k')
    plt.axvline(x = Tf2_over_eye, linewidth = lw, linestyle = '--', color = 'k')
    plt.axvline(x = tf1_over, linewidth = lw, linestyle = '-', color = 'y')
    plt.axvline(x = tf2_over, linewidth = lw, linestyle = '-', color = 'y')
    
    plt.xlabel('Tiempo [hora decimal]', fontsize = font_label)
    plt.ylabel(r'$|\vec{B}|$ [nT]', fontsize = font_label)
#    plt.xlim(9.5,10.5)
#    plt.ylim(0,50)
    plt.tick_params(axis = 'both', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
#    plt.grid(axis = 'both', which = 'both', alpha = grid_alpha, linewidth = lw, linestyle = '--')
    plt.legend(loc = 0, fontsize = font_leg)
    
    

#%%

#hago subplots para que se vea mejor cada subestructura    
    
    
if MODO_subestructuras == 1:
    
    
    if tipo == 'in':
        
        #ploteo todos los limites determinados (a ojo y automatizados)
        
        figsize = (30,15)
        lw = 1
        msize = 8
        font_title = 30
        font_label = 30
        font_leg = 20
        ticks_l = 6
        ticks_w = 3
        grid_alpha = 0.8
        updown_alpha = 0.5
        
        ylim = [0,50]
        
        xlim_foot = [t_mag[114071], 9.81999] #*
        xlim_ramp = [t_mag[115023],t_mag[115035]] #*
        xlim_over = [t_mag[114550],t_mag[119341]] #*
        
        
        
        plt.figure(55, figsize = figsize)
        plt.suptitle(r'$\bf{Delimitación}$ $\bf{temporal}$ $\bf{de}$ $\bf{subestructuras}$', fontsize = font_title)
        plt.subplots_adjust(top=0.9, bottom=0.10, left=0.10, right=0.95, hspace=0.3, wspace=0.25)

        
        p0 = plt.subplot(131)
        plt.title('foot', fontsize = font_label)
        plt.plot(t_mag, B, linewidth = lw)
        #asintotas de Bu y Bd
        plt.axhline(y = norm_Bu, linewidth = lw, color = 'r', label = r'$B_u$')
        plt.axhspan(ymin = norm_Bu - std_norm_Bu, ymax = norm_Bu + std_norm_Bu, facecolor = 'g', alpha = updown_alpha)
        #inicio foot
        plt.axvline(x = Ti1_foot_eye, linewidth = lw, linestyle = '--', color = 'k', label = 'límites a ojo')
        plt.axvline(x = Ti2_foot_eye, linewidth = lw, linestyle = '--', color = 'k')
        plt.axvline(x = ti1_foot, linewidth = lw, linestyle = '-', color = 'm', label = 'inicio foot')
        plt.axvline(x = ti2_foot, linewidth = lw, linestyle = '-', color = 'm')
        #inicio ramp
        plt.axvline(x = Ti1_ramp_eye, linewidth = lw, linestyle = '--', color = 'k')
        plt.axvline(x = Ti2_ramp_eye, linewidth = lw, linestyle = '--', color = 'k')
        plt.axvline(x = ti1_ramp, linewidth = lw, linestyle = '-', color = 'c', label = 'inicio ramp')
        plt.axvline(x = ti2_ramp, linewidth = lw, linestyle = '-', color = 'c')
        
#        class CustomScalarFormatter(ticker.ScalarFormatter):
#            def format_data(self, value):
#                if self._useLocale:
#                    s = ticker.locale.format_string('%1.3g', (value,))
#                else:
#                    s = '%1.3g' % value
#                s = self._formatSciNotation(s)
#                return self.fix_minus(s)
#        
#        xmajorformatter = CustomScalarFormatter(9.81)  # default useOffset=True
#        p0.get_xaxis().set_major_formatter(xmajorformatter)
        
        
        plt.xlim(xlim_foot[0], xlim_foot[1])
        plt.ylim(ylim[0], ylim[1])
        
        ft_ticks = np.round(np.linspace(xlim_foot[0], xlim_foot[1], 4, endpoint = True),6)
        false_ticks = (ft_ticks - 9.81)*(1e5)
        ft_ticks_label = []
        for i in range(len(false_ticks)):
            string = '{:04f}'.format(false_ticks[i])
            string = string[:-7]
            ft_ticks_label.append(string)

        
        plt.xticks(ft_ticks, ft_ticks_label)
    
        plt.tick_params(axis = 'both', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
        plt.ylabel(r'$|B_{SW,MSO}|$ [nT]', fontsize = font_label)
    #    plt.grid(axis = 'both', which = 'both', alpha = grid_alpha, linewidth = lw, linestyle = '--')
        plt.legend(loc = 0, fontsize = font_leg)  
        
        
        p1 = plt.subplot(132)
        plt.title('ramp', fontsize = font_label)
        plt.setp(p1.get_yticklabels(), visible = False)
        plt.plot(t_mag, B, linewidth = lw)
        #asintotas de Bu y Bd
        plt.axhline(y = norm_Bd, linewidth = lw, color = 'r', label = r'$B_d$')
        plt.axhspan(ymin = norm_Bd - std_norm_Bd, ymax = norm_Bd + std_norm_Bd, facecolor = 'g', alpha = updown_alpha)
        #inicio ramp
        plt.axvline(x = Ti1_ramp_eye, linewidth = lw, linestyle = '--', color = 'k', label = 'límites a ojo')
        plt.axvline(x = Ti2_ramp_eye, linewidth = lw, linestyle = '--', color = 'k')
        plt.axvline(x = ti1_ramp, linewidth = lw, linestyle = '-', color = 'c', label = 'inicio ramp')
        plt.axvline(x = ti2_ramp, linewidth = lw, linestyle = '-', color = 'c')
        #ajustes ramp
        plt.plot(t_mag[min(i1_ramp,f1_rampfit):max(i1_ramp,f1_rampfit)+1], i1_params[0]*t_mag[min(i1_ramp,f1_rampfit):max(i1_ramp,f1_rampfit)+1] + i1_params[1], linewidth = lw, color = 'y', linestyle = '-.')
        plt.plot(t_mag[min(i2_ramp,f2_rampfit):max(i2_ramp,f2_rampfit)+1], i2_params[0]*t_mag[min(i2_ramp,f2_rampfit):max(i2_ramp,f2_rampfit)+1] + i2_params[1], linewidth = lw, color = 'r', linestyle = '-.')
        plt.axvline(x = t_mag[C_high], linewidth = lw, linestyle = 'dotted', color = 'orange', label = 'Centro del choque')
        #inicio overshoot
        plt.axvline(x = Ti1_over_eye, linewidth = lw, linestyle = '--', color = 'k')
        plt.axvline(x = Ti2_over_eye, linewidth = lw, linestyle = '--', color = 'k')
        plt.axvline(x = ti1_over, linewidth = lw, linestyle = '-', color = 'g', label = 'inicio overshoot')
        plt.axvline(x = ti2_over, linewidth = lw, linestyle = '-', color = 'g')        
        plt.xlim(xlim_ramp[0], xlim_ramp[1])
        plt.ylim(ylim[0], ylim[1])
        
        r_ticks = np.round(np.linspace(xlim_ramp[0], xlim_ramp[1], 4, endpoint = True),6)
        r_false_ticks = (r_ticks - 9.819)*(1e7)
        r_ticks_label = []
        for i in range(len(r_false_ticks)):
            string = '{:04f}'.format(r_false_ticks[i])
            string = string[:-8]
            #print(string)
            r_ticks_label.append(string)
        
        plt.xticks(r_ticks, r_ticks_label)
        
        plt.tick_params(axis = 'both', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
        plt.xlabel('Tiempo [hs]', fontsize = font_label)
    #    plt.grid(axis = 'both', which = 'both', alpha = grid_alpha, linewidth = lw, linestyle = '--')
        plt.legend(loc = 0, fontsize = font_leg)
        
        p2 = plt.subplot(133)
        plt.title('overshoot', fontsize = font_label)
        plt.setp(p2.get_yticklabels(), visible = False)
        plt.plot(t_mag, B, linewidth = lw)
        #asintotas de Bu y Bd
        plt.axhline(y = norm_Bd, linewidth = lw, color = 'r', label = r'$B_d$')
        plt.axhspan(ymin = norm_Bd - std_norm_Bd, ymax = norm_Bd + std_norm_Bd, facecolor = 'g', alpha = updown_alpha)
        #inicio overshoot
        plt.axvline(x = Ti1_over_eye, linewidth = lw, linestyle = '--', color = 'k', label = 'límites a ojo')
        plt.axvline(x = Ti2_over_eye, linewidth = lw, linestyle = '--', color = 'k')
        plt.axvline(x = ti1_over, linewidth = lw, linestyle = '-', color = 'g', label = 'inicio overshoot')
        plt.axvline(x = ti2_over, linewidth = lw, linestyle = '-', color = 'g')
        plt.axhline(y = Bmax_over, linewidth = lw, linestyle = '-', color = 'purple', label = 'máximo overshoot')
        #final overshoot
        plt.axvline(x = Tf1_over_eye, linewidth = lw, linestyle = '--', color = 'k')
        plt.axvline(x = Tf2_over_eye, linewidth = lw, linestyle = '--', color = 'k')
        plt.axvline(x = tf1_over, linewidth = lw, linestyle = '-', color = 'y', label = 'final overshoot')
        plt.axvline(x = tf2_over, linewidth = lw, linestyle = '-', color = 'y')
        plt.xlim(xlim_over[0], xlim_over[1])
        plt.ylim(ylim[0], ylim[1])
        plt.xticks(np.round(np.linspace(xlim_over[0], xlim_over[1], 5, endpoint = True),2))
        plt.tick_params(axis = 'both', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
    #    plt.grid(axis = 'both', which = 'both', alpha = grid_alpha, linewidth = lw, linestyle = '--')
        plt.legend(loc = 0, fontsize = font_leg)
        
        
        
        plt.savefig(path_analisis+'subestructuras_temporal_{}'.format(shock_date))
        plt.savefig(path_analisis+'subestructuras_temporal_{}.pdf'.format(shock_date))
    
#%%    
    
    if tipo == 'out':
        
        
        #ploteo todos los limites determinados (a ojo y automatizados)
        
        figsize = (30,15)
        lw = 1.5
        msize = 8
        font_title = 30
        font_label = 30
        font_leg = 15
        ticks_l = 6
        ticks_w = 3
        grid_alpha = 0.8
        updown_alpha = 0.5
        
        xlim_foot = [9.81159, 9.82212] #*
        xlim_ramp = [9.81938,9.81954] #*
        xlim_over = [9.81777,9.86354] #*
        
        
        
        plt.figure(55, figsize = figsize)
        plt.title('Delimitacion de subestructuras', fontsize = font_label)
        plt.subplots_adjust(top=0.92, bottom=0.10, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
        
        plt.subplot(133)
        plt.plot(t_mag, B, linewidth = lw)
        #asintotas de Bu y Bd
        plt.axhline(y = norm_Bu, linewidth = lw, color = 'r', label = r'$B_u$')
        plt.axhspan(ymin = norm_Bu - std_norm_Bu, ymax = norm_Bu + std_norm_Bu, facecolor = 'g', alpha = updown_alpha)
        #inicio foot
        plt.axvline(x = Ti1_foot_eye, linewidth = lw, linestyle = '--', color = 'k', label = 'límites a ojo')
        plt.axvline(x = Ti2_foot_eye, linewidth = lw, linestyle = '--', color = 'k')
        plt.axvline(x = ti1_foot, linewidth = lw, linestyle = '-', color = 'm', label = 'inicio foot')
        plt.axvline(x = ti2_foot, linewidth = lw, linestyle = '-', color = 'm')
        #inicio ramp
        plt.axvline(x = Ti1_ramp_eye, linewidth = lw, linestyle = '--', color = 'k')
        plt.axvline(x = Ti2_ramp_eye, linewidth = lw, linestyle = '--', color = 'k')
        plt.axvline(x = ti1_ramp, linewidth = lw, linestyle = '-', color = 'c', label = 'inicio ramp')
        plt.axvline(x = ti2_ramp, linewidth = lw, linestyle = '-', color = 'c')
        plt.xlim(xlim_foot[0], xlim_foot[1])
        plt.xticks(np.linspace(xlim_foot[0], xlim_foot[1], 4, endpoint = True), rotation = 45)
        plt.tick_params(axis = 'both', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
        plt.ylabel(r'$|\vec{B}|$ [nT]', fontsize = font_label)
    #    plt.grid(axis = 'both', which = 'both', alpha = grid_alpha, linewidth = lw, linestyle = '--')
        plt.legend(loc = 0, fontsize = font_leg)  
        
        
        plt.subplot(132)
        plt.plot(t_mag, B, linewidth = lw)
        #asintotas de Bu y Bd
        plt.axhline(y = norm_Bd, linewidth = lw, color = 'r', label = r'$B_d$')
        plt.axhspan(ymin = norm_Bd - std_norm_Bd, ymax = norm_Bd + std_norm_Bd, facecolor = 'g', alpha = updown_alpha)
        #inicio ramp
        plt.axvline(x = Ti1_ramp_eye, linewidth = lw, linestyle = '--', color = 'k', label = 'límites a ojo')
        plt.axvline(x = Ti2_ramp_eye, linewidth = lw, linestyle = '--', color = 'k')
        plt.axvline(x = ti1_ramp, linewidth = lw, linestyle = '-', color = 'c', label = 'inicio ramp')
        plt.axvline(x = ti2_ramp, linewidth = lw, linestyle = '-', color = 'c')
        #ajustes ramp
        plt.plot(t_mag[min(i1_ramp,f1_rampfit):max(i1_ramp,f1_rampfit)+1], i1_params[0]*t_mag[min(i1_ramp,f1_rampfit):max(i1_ramp,f1_rampfit)+1] + i1_params[1], linewidth = lw, color = 'y', linestyle = '-.')
        plt.plot(t_mag[min(i2_ramp,f2_rampfit):max(i2_ramp,f2_rampfit)+1], i2_params[0]*t_mag[min(i2_ramp,f2_rampfit):max(i2_ramp,f2_rampfit)+1] + i2_params[1], linewidth = lw, color = 'r', linestyle = '-.')
        plt.axvline(x = t_mag[C_high], linewidth = lw, linestyle = 'dotted', color = 'orange', label = 'Centro del choque')
        #inicio overshoot
        plt.axvline(x = Ti1_over_eye, linewidth = lw, linestyle = '--', color = 'k')
        plt.axvline(x = Ti2_over_eye, linewidth = lw, linestyle = '--', color = 'k')
        plt.axvline(x = ti1_over, linewidth = lw, linestyle = '-', color = 'g', label = 'inicio overshoot')
        plt.axvline(x = ti2_over, linewidth = lw, linestyle = '-', color = 'g')
        plt.xlim(xlim_ramp[0], xlim_ramp[1])
        plt.xticks(np.linspace(xlim_ramp[0], xlim_ramp[1], 3, endpoint = True))
        plt.tick_params(axis = 'both', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
        plt.xlabel('Tiempo [hora decimal]', fontsize = font_label)
    #    plt.grid(axis = 'both', which = 'both', alpha = grid_alpha, linewidth = lw, linestyle = '--')
        plt.legend(loc = 0, fontsize = font_leg)
        
        plt.subplot(131)
        plt.plot(t_mag, B, linewidth = lw)
        #asintotas de Bu y Bd
        plt.axhline(y = norm_Bd, linewidth = lw, color = 'r', label = r'$B_d$')
        plt.axhspan(ymin = norm_Bd - std_norm_Bd, ymax = norm_Bd + std_norm_Bd, facecolor = 'g', alpha = updown_alpha)
        #inicio overshoot
        plt.axvline(x = Ti1_over_eye, linewidth = lw, linestyle = '--', color = 'k', label = 'límites a ojo')
        plt.axvline(x = Ti2_over_eye, linewidth = lw, linestyle = '--', color = 'k')
        plt.axvline(x = ti1_over, linewidth = lw, linestyle = '-', color = 'g', label = 'inicio overshoot')
        plt.axvline(x = ti2_over, linewidth = lw, linestyle = '-', color = 'g')
        plt.axhline(y = Bmax_over, linewidth = lw, linestyle = '-', color = 'purple', label = 'máximo overshoot')
        #final overshoot
        plt.axvline(x = Tf1_over_eye, linewidth = lw, linestyle = '--', color = 'k')
        plt.axvline(x = Tf2_over_eye, linewidth = lw, linestyle = '--', color = 'k')
        plt.axvline(x = tf1_over, linewidth = lw, linestyle = '-', color = 'y', label = 'final overshoot')
        plt.axvline(x = tf2_over, linewidth = lw, linestyle = '-', color = 'y')
        plt.xlim(xlim_over[0], xlim_over[1])
        plt.xticks(np.linspace(xlim_over[0], xlim_over[1], 3, endpoint = True))
        plt.tick_params(axis = 'both', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
    #    plt.grid(axis = 'both', which = 'both', alpha = grid_alpha, linewidth = lw, linestyle = '--')
        plt.legend(loc = 0, fontsize = font_leg)
        
        
        
        plt.savefig(path_analisis+'subestructuras_temporal_{}'.format(shock_date))
        plt.savefig(path_analisis+'subestructuras_temporal_{}.pdf'.format(shock_date))
    
    

    

#%%
    
# Calculo angulo cenital y normal del fit global del bs en baja frec
    

#centro del shock (centro de la rampa) en variables de baja frec
C = (np.abs(t_mag_low-tc)).argmin()
#posicion de la nave en el centro del shock
Rc = np.array([x_low[C], y_low[C], z_low[C]])
err_Rc = np.array([1e-8, 1e-8, 1e-8])
    

#zenith angle
cenit = fcop.alpha(Rc,np.array([1,0,0])) #el segundo vector es el versor x_MSO
err_cenit = fcop.err_alpha(Rc,np.array([1,0,0]), err_Rc, err_Rc) # el error del versor x es igual al de Rc


#normal del shock reescalando el fit macro del bowshock

L = fbow.L_fit(Rc) #redefino L para que el fit contenga el centro de mi shock y calculo normal del fit
err_L = fbow.error_L_vignes(Rc, err_Rc)

N = fbow.norm_fit_MGS(Rc[0], Rc[1], Rc[2], L)
err_N = fbow.err_N_fit(Rc, err_Rc, L, err_L)
err_perp_N = err_N - np.dot(np.dot(err_N,N),N)
#cono_err_N = fcop.alpha(N, (N + err_perp_N))

#angulo entre campo upstream y normal del fit
theta_N = fcop.alpha(Bu_low,N)
err_theta_N = fcop.err_alpha(Bu_low, N, std_Bu_low, err_N)

#angulo entre vel_SW upstream y normal del fit
theta_NVu = fcop.alpha(Vu,N)
err_theta_NVu = fcop.err_alpha(Vu, N, std_Vu, err_N)

#angulo entre posicion de la nave en el centro del shock y normal del fit
theta_NRc = fcop.alpha(Rc,N)
err_theta_NRc = fcop.err_alpha(Rc, N, err_Rc, err_N)

#angulo entre vel nave y normal del fit
err_v_nave = np.array([1e-8,1e-8,1e-8])
theta_Nvnave = fcop.alpha(v_nave,N)
err_theta_Nvnave = fcop.err_alpha(v_nave, N, err_v_nave, err_N)

#%%

#paso del espacio temporal al espacial

#ion cyclotron angular frequency (unidades de rad/s en SI)

qp = 1.6e-19 #Coulombs (q electrica del proton)
mp = 1.67e-27 #kg (masa del proton)
W_ci = qp*norm_Bu*(1e-9)/mp #rad/s (pase B de nT a T)

if tipo == 'in': sgn = -1 #*  (-1 inbound, +1 outbound)
elif tipo == 'out': sgn = 1

#suponiendo foot desarrollado al 100%
Vsh_A_100, Vsh_B_100 = vel_shock(sgn, N, theta_N, theta_NVu, norm_Vu, W_ci, ancho_foot_temp, 1)

#suponiendo foot desarrollado al perX% y perY%
perX = 0.85 #*
perY = 0.75 #*
Vsh_A_perX, Vsh_B_perX = vel_shock(sgn, N, theta_N, theta_NVu, norm_Vu, W_ci, ancho_foot_temp, perX)
Vsh_A_perY, Vsh_B_perY = vel_shock(sgn, N, theta_N, theta_NVu, norm_Vu, W_ci, ancho_foot_temp, perY)



#%%
#anchos espaciales en km (pues Vsh esta en km/s igual que Vu)

#con metodo A

#para 100%
ancho_foot_km_A100 = Vsh_A_100*ancho_foot_temp
ancho_ramp_km_A100 = Vsh_A_100*ancho_ramp_temp
ancho_over_km_A100 = Vsh_A_100*ancho_over_temp
ancho_shock_km_A100 = Vsh_A_100*ancho_shock_temp

#para perX%
ancho_foot_km_AperX = Vsh_A_perX*ancho_foot_temp
ancho_ramp_km_AperX = Vsh_A_perX*ancho_ramp_temp
ancho_over_km_AperX = Vsh_A_perX*ancho_over_temp
ancho_shock_km_AperX = Vsh_A_perX*ancho_shock_temp

#para perY%
ancho_foot_km_AperY = Vsh_A_perY*ancho_foot_temp
ancho_ramp_km_AperY = Vsh_A_perY*ancho_ramp_temp
ancho_over_km_AperY = Vsh_A_perY*ancho_over_temp
ancho_shock_km_AperY = Vsh_A_perY*ancho_shock_temp


#con metodo B

#para 100%
ancho_foot_km_B100 = Vsh_B_100*ancho_foot_temp
ancho_ramp_km_B100 = Vsh_B_100*ancho_ramp_temp
ancho_over_km_B100 = Vsh_B_100*ancho_over_temp
ancho_shock_km_B100 = Vsh_B_100*ancho_shock_temp

#para perX%
ancho_foot_km_BperX = Vsh_B_perX*ancho_foot_temp
ancho_ramp_km_BperX = Vsh_B_perX*ancho_ramp_temp
ancho_over_km_BperX = Vsh_B_perX*ancho_over_temp
ancho_shock_km_BperX = Vsh_B_perX*ancho_shock_temp

#para perY%
ancho_foot_km_BperY = Vsh_B_perY*ancho_foot_temp
ancho_ramp_km_BperY = Vsh_B_perY*ancho_ramp_temp
ancho_over_km_BperY = Vsh_B_perY*ancho_over_temp
ancho_shock_km_BperY = Vsh_B_perY*ancho_shock_temp

#%%

#anchos espaciales en unidades de:
#                                  "ion inertial lengths" = "di" ("c/Wpi") para overshoot
#                                  "ion gyroradii" = "ri" para overshoot y foot
#                                  "electron inertial length" = "de" para la ramp



densnum_u = np.mean(densidad_swia[min(iu_v,fu_v):max(iu_v,fu_v)]) #en 1/cm^3
di = 2.28e2 / np.sqrt(densnum_u) # en km 
de = 5.31 / np.sqrt(densnum_u)  # en km
ri = norm_Vu / W_ci            #en km


#con metodo A

#para 100%
ancho_foot_ri_A100 = ancho_foot_km_A100 / ri
ancho_ramp_de_A100 = ancho_ramp_km_A100 / de
ancho_over_di_A100 = ancho_over_km_A100 / di
ancho_over_ri_A100 = ancho_over_km_A100 / ri
ancho_shock_ri_A100 = ancho_shock_km_A100 / ri

#para perX
ancho_foot_ri_AperX = ancho_foot_km_AperX / ri
ancho_ramp_de_AperX = ancho_ramp_km_AperX / de
ancho_over_di_AperX = ancho_over_km_AperX / di
ancho_over_ri_AperX = ancho_over_km_AperX / ri
ancho_shock_ri_AperX = ancho_shock_km_AperX / ri

#para perY
ancho_foot_ri_AperY = ancho_foot_km_AperY / ri
ancho_ramp_de_AperY = ancho_ramp_km_AperY / de
ancho_over_di_AperY = ancho_over_km_AperY / di
ancho_over_ri_AperY = ancho_over_km_AperY / ri
ancho_shock_ri_AperY = ancho_shock_km_AperY / ri


#con metodo B

#para 100%
ancho_foot_ri_B100 = ancho_foot_km_B100 / ri
ancho_ramp_de_B100 = ancho_ramp_km_B100 / de
ancho_over_di_B100 = ancho_over_km_B100 / di
ancho_over_ri_B100 = ancho_over_km_B100 / ri
ancho_shock_ri_B100 = ancho_shock_km_B100 / ri

#para perX
ancho_foot_ri_BperX = ancho_foot_km_BperX / ri
ancho_ramp_de_BperX = ancho_ramp_km_BperX / de
ancho_over_di_BperX = ancho_over_km_BperX / di
ancho_over_ri_BperX = ancho_over_km_BperX / ri
ancho_shock_ri_BperX = ancho_shock_km_BperX / ri

#para perY
ancho_foot_ri_BperY = ancho_foot_km_BperY / ri
ancho_ramp_de_BperY = ancho_ramp_km_BperY / de
ancho_over_di_BperY = ancho_over_km_BperY / di
ancho_over_ri_BperY = ancho_over_km_BperY / ri
ancho_shock_ri_BperY = ancho_shock_km_BperY / ri


#%%

#ploteo perfiles espaciales para el caso foot desarrollado al 100% (metodos A y B)
#ploteo todos los limites determinados (a ojo y automatizados)


xdiA = (t_mag-t_mag[C_high])*3600*Vsh_A_100/di
xdiB = (t_mag-t_mag[C_high])*3600*Vsh_B_100/di
xdeA = (t_mag-t_mag[C_high])*3600*Vsh_A_100/de
xdeB = (t_mag-t_mag[C_high])*3600*Vsh_B_100/de


#%%

#plots completos



if MODO_subestructuras == 1:

    #metodo A

    
    figsize = (30,15)
    lw = 3
    msize = 8
    font_title = 30
    font_label = 30
    font_leg = 15
    ticks_l = 6
    ticks_w = 3
    grid_alpha = 0.8
    updown_alpha = 0.5
    xlim = [-15,45]
    ymax = 50
    
    
    plt.figure(6, figsize = figsize)
    plt.title('Delimitacion espacial de subestructuras - Método A', fontsize = font_label)
    
    plt.plot(xdiA, B, linewidth = lw, marker = 'o', markersize = msize)
    #asintotas de Bu y Bd
    plt.axhline(y = norm_Bu, linewidth = lw, color = 'r')
    plt.axhline(y = norm_Bd, linewidth = lw, color = 'r')
    plt.axhspan(ymin = norm_Bu - std_norm_Bu, ymax = norm_Bu + std_norm_Bu, facecolor = 'g', alpha = updown_alpha)
    plt.axhspan(ymin = norm_Bd - std_norm_Bd, ymax = norm_Bd + std_norm_Bd, facecolor = 'g', alpha = updown_alpha)
    #inicio foot
    plt.axvline(x = xdiA[i1_foot_eye], linewidth = lw, linestyle = '--', color = 'k')
    plt.axvline(x = xdiA[i2_foot_eye], linewidth = lw, linestyle = '--', color = 'k')
    plt.axvline(x = xdiA[i1_foot], linewidth = lw, linestyle = '-', color = 'm')
    plt.axvline(x = xdiA[i2_foot], linewidth = lw, linestyle = '-', color = 'm')
    #inicio ramp
    plt.axvline(x = xdiA[i1_ramp_eye], linewidth = lw, linestyle = '--', color = 'k')
    plt.axvline(x = xdiA[i2_ramp_eye], linewidth = lw, linestyle = '--', color = 'k')
    plt.axvline(x = xdiA[i1_ramp], linewidth = lw, linestyle = '-', color = 'c')
    plt.axvline(x = xdiA[i2_ramp], linewidth = lw, linestyle = '-', color = 'c')
#    #ajustes ramp (os parametros estan para rectas temporales)
#    plt.plot(xdiA[i1_ramp:f1_rampfit+1], i1_params[0]*xdiA[i1_ramp:f1_rampfit+1] + i1_params[1], linewidth = lw, color = 'y', linestyle = '-.')
#    plt.plot(xdiA[i2_ramp:f2_rampfit+1], i2_params[0]*xdiA[i2_ramp:f2_rampfit+1] + i2_params[1], linewidth = lw, color = 'r', linestyle = '-.')
    plt.axvline(x = xdiA[C], linewidth = lw, linestyle = 'dotted', color = 'orange', label = 'Centro del choque')
    #inicio overshoot
    plt.axvline(x = xdiA[i1_over_eye], linewidth = lw, linestyle = '--', color = 'k')
    plt.axvline(x = xdiA[i2_over_eye], linewidth = lw, linestyle = '--', color = 'k')
    plt.axvline(x = xdiA[i1_over], linewidth = lw, linestyle = '-', color = 'g')
    plt.axvline(x = xdiA[i2_over], linewidth = lw, linestyle = '-', color = 'g')
    plt.axhline(y = Bmax_over, linewidth = lw, linestyle = '-', color = 'purple')
    #final overshoot
    plt.axvline(x = xdiA[f1_over_eye], linewidth = lw, linestyle = '--', color = 'k')
    plt.axvline(x = xdiA[f2_over_eye], linewidth = lw, linestyle = '--', color = 'k')
    plt.axvline(x = xdiA[f1_over], linewidth = lw, linestyle = '-', color = 'y')
    plt.axvline(x = xdiA[f2_over], linewidth = lw, linestyle = '-', color = 'y')
    
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ymax = ymax)
    plt.xlabel(r'$c/\omega_{pi}$', fontsize = font_label)
    plt.ylabel(r'$|\vec{B}|$ [nT]', fontsize = font_label)
    plt.tick_params(axis = 'both', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
    plt.grid(axis = 'both', which = 'both', alpha = grid_alpha, linewidth = lw, linestyle = '--')
    plt.legend(loc = 0, fontsize = font_leg)
    



    #metodo B
    
    
    plt.figure(7, figsize = figsize)
    plt.title('Delimitacion espacial de subestructuras - Método B', fontsize = font_label)
    
    plt.plot(xdiB, B, linewidth = lw, marker = 'o', markersize = msize)
#    #asintotas de Bu y Bd
    plt.axhline(y = norm_Bu, linewidth = lw, color = 'r')
    plt.axhline(y = norm_Bd, linewidth = lw, color = 'r')
    plt.axhspan(ymin = norm_Bu - std_norm_Bu, ymax = norm_Bu + std_norm_Bu, facecolor = 'g', alpha = updown_alpha)
    plt.axhspan(ymin = norm_Bd - std_norm_Bd, ymax = norm_Bd + std_norm_Bd, facecolor = 'g', alpha = updown_alpha)
#    #inicio foot
    plt.axvline(x = xdiB[i1_foot_eye], linewidth = lw, linestyle = '--', color = 'k')
    plt.axvline(x = xdiB[i2_foot_eye], linewidth = lw, linestyle = '--', color = 'k')
    plt.axvline(x = xdiB[i1_foot], linewidth = lw, linestyle = '-', color = 'm')
    plt.axvline(x = xdiB[i2_foot], linewidth = lw, linestyle = '-', color = 'm')
#    #inicio ramp
    plt.axvline(x = xdiB[i1_ramp_eye], linewidth = lw, linestyle = '--', color = 'k')
    plt.axvline(x = xdiB[i2_ramp_eye], linewidth = lw, linestyle = '--', color = 'k')
    plt.axvline(x = xdiB[i1_ramp], linewidth = lw, linestyle = '-', color = 'c')
    plt.axvline(x = xdiB[i2_ramp], linewidth = lw, linestyle = '-', color = 'c')
#    #ajustes ramp (os parametros estan para rectas temporales)
#    plt.plot(xdiB[i1_ramp:f1_rampfit+1], i1_params[0]*xdiB[i1_ramp:f1_rampfit+1] + i1_params[1], linewidth = lw, color = 'y', linestyle = '-.')
#    plt.plot(xdiB[i2_ramp:f2_rampfit+1], i2_params[0]*xdiB[i2_ramp:f2_rampfit+1] + i2_params[1], linewidth = lw, color = 'r', linestyle = '-.')
    plt.axvline(x = xdiB[C], linewidth = lw, linestyle = 'dotted', color = 'orange', label = 'Centro del choque')
#    #inicio overshoot
    plt.axvline(x = xdiB[i1_over_eye], linewidth = lw, linestyle = '--', color = 'k')
    plt.axvline(x = xdiB[i2_over_eye], linewidth = lw, linestyle = '--', color = 'k')
    plt.axvline(x = xdiB[i1_over], linewidth = lw, linestyle = '-', color = 'g')
    plt.axvline(x = xdiB[i2_over], linewidth = lw, linestyle = '-', color = 'g')
    plt.axhline(y = Bmax_over, linewidth = lw, linestyle = '-', color = 'purple')
#    #final overshoot
    plt.axvline(x = xdiB[f1_over_eye], linewidth = lw, linestyle = '--', color = 'k')
    plt.axvline(x = xdiB[f2_over_eye], linewidth = lw, linestyle = '--', color = 'k')
    plt.axvline(x = xdiB[f1_over], linewidth = lw, linestyle = '-', color = 'y')
    plt.axvline(x = xdiB[f2_over], linewidth = lw, linestyle = '-', color = 'y')
    
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ymax = ymax)
    plt.xlabel(r'$c/\omega_{pi}$', fontsize = font_label)
    plt.ylabel(r'$|\vec{B}|$ [nT]', fontsize = font_label)
    plt.tick_params(axis = 'both', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
    plt.grid(axis = 'both', which = 'both', alpha = grid_alpha, linewidth = lw, linestyle = '--')
    plt.legend(loc = 0, fontsize = font_leg)
    

#%%

#hago subplots para que se vea mejor cada subestructura    

# Metodo A    
    
if MODO_subestructuras == 1:
    
    
    #ploteo todos los limites determinados (a ojo y automatizados)
        
    figsize = (30,15)
    lw = 1.5
    msize = 8
    font_title = 30
    font_label = 30
    font_leg = 20
    ticks_l = 6
    ticks_w = 3
    grid_alpha = 0.8
    updown_alpha = 0.5
    
    ylim = [0,50]
    
    xlim_foot = [-10, 2] #*
    xlim_ramp = [-2.5,2.5] #*
    xlim_over = [-5,45] #*
    
    tipo = 'in' #*
    
    
    
    if tipo == 'in':
                
        
        plt.figure(9, figsize = figsize)
        plt.suptitle(r'$\bf{Método}$ $\bf{A}$, $\bf{desarrollo}$ $\bf{del}$ $\bf{foot}$ $\bf{100 \%}$', fontsize = font_title)
        plt.subplots_adjust(top=0.9, bottom=0.10, left=0.10, right=0.95, hspace=0.3, wspace=0.2)
        
        p0 = plt.subplot(131)
        plt.title('foot', fontsize = font_label)
        plt.plot(xdiA, B, linewidth = lw)
        #asintotas de Bu y Bd
        plt.axhline(y = norm_Bu, linewidth = lw, color = 'r', label = r'$B_u$')
        plt.axhspan(ymin = norm_Bu - std_norm_Bu, ymax = norm_Bu + std_norm_Bu, facecolor = 'g', alpha = updown_alpha)
        #inicio foot
        plt.axvline(x = xdiA[i1_foot_eye], linewidth = lw, linestyle = '--', color = 'k', label = 'límites a ojo')
        plt.axvline(x = xdiA[i2_foot_eye], linewidth = lw, linestyle = '--', color = 'k')
        plt.axvline(x = xdiA[i1_foot], linewidth = lw, linestyle = '-', color = 'm', label = 'inicio foot')
        plt.axvline(x = xdiA[i2_foot], linewidth = lw, linestyle = '-', color = 'm')
        #inicio ramp
        plt.axvline(x = xdiA[i1_ramp_eye], linewidth = lw, linestyle = '--', color = 'k')
        plt.axvline(x = xdiA[i2_ramp_eye], linewidth = lw, linestyle = '--', color = 'k')
        plt.axvline(x = xdiA[i1_ramp], linewidth = lw, linestyle = '-', color = 'c', label = 'inicio ramp')
        plt.axvline(x = xdiA[i2_ramp], linewidth = lw, linestyle = '-', color = 'c')
        plt.xlim(xlim_foot[0], xlim_foot[1])
        plt.ylim(ylim[0], ylim[1])
        plt.xticks(np.linspace(xlim_foot[0], xlim_foot[1], 4, endpoint = True))
        plt.tick_params(axis = 'both', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
        plt.xlabel(r'$c/\omega_{pi}$', fontsize = font_label)
        plt.ylabel(r'$|B_{SW,MSO}|$ [nT]', fontsize = font_label)
    #    plt.grid(axis = 'both', which = 'both', alpha = grid_alpha, linewidth = lw, linestyle = '--')
        plt.legend(loc = 0, fontsize = font_leg)  
        
        
        p1 = plt.subplot(132)
        plt.title('ramp', fontsize = font_label)
        plt.setp(p1.get_yticklabels(), visible = False)
        plt.plot(xdeA, B, linewidth = lw)
        #asintotas de Bu y Bd
        plt.axhline(y = norm_Bd, linewidth = lw, color = 'r', label = r'$B_d$')
        plt.axhspan(ymin = norm_Bd - std_norm_Bd, ymax = norm_Bd + std_norm_Bd, facecolor = 'g', alpha = updown_alpha)
        #inicio ramp
        plt.axvline(x = xdeA[i1_ramp_eye], linewidth = lw, linestyle = '--', color = 'k', label = 'límites a ojo')
        plt.axvline(x = xdeA[i2_ramp_eye], linewidth = lw, linestyle = '--', color = 'k')
        plt.axvline(x = xdeA[i1_ramp], linewidth = lw, linestyle = '-', color = 'c', label = 'inicio ramp')
        plt.axvline(x = xdeA[i2_ramp], linewidth = lw, linestyle = '-', color = 'c')
        #ajustes ramp
        plt.axvline(x = xdeA[C_high], linewidth = lw, linestyle = 'dotted', color = 'orange', label = 'Centro del choque')
        #inicio overshoot
        plt.axvline(x = xdeA[i1_over_eye], linewidth = lw, linestyle = '--', color = 'k')
        plt.axvline(x = xdeA[i2_over_eye], linewidth = lw, linestyle = '--', color = 'k')
        plt.axvline(x = xdeA[i1_over], linewidth = lw, linestyle = '-', color = 'g', label = 'inicio overshoot')
        plt.axvline(x = xdeA[i2_over], linewidth = lw, linestyle = '-', color = 'g')
        plt.xlim(xlim_ramp[0], xlim_ramp[1])
        plt.ylim(ylim[0], ylim[1])
        plt.xticks(np.linspace(xlim_ramp[0], xlim_ramp[1], 5, endpoint = True))
        plt.tick_params(axis = 'both', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
        plt.xlabel(r'$c/\omega_{pe}$', fontsize = font_label)
    #    plt.grid(axis = 'both', which = 'both', alpha = grid_alpha, linewidth = lw, linestyle = '--')
        plt.legend(loc = 0, fontsize = font_leg)
        
        p2 = plt.subplot(133)
        plt.title('overshoot', fontsize = font_label)
        plt.setp(p2.get_yticklabels(), visible = False)
        plt.plot(xdiA, B, linewidth = lw)
        #asintotas de Bu y Bd
        plt.axhline(y = norm_Bd, linewidth = lw, color = 'r', label = r'$B_d$')
        plt.axhspan(ymin = norm_Bd - std_norm_Bd, ymax = norm_Bd + std_norm_Bd, facecolor = 'g', alpha = updown_alpha)
        #inicio overshoot
        plt.axvline(x = xdiA[i1_over_eye], linewidth = lw, linestyle = '--', color = 'k', label = 'límites a ojo')
        plt.axvline(x = xdiA[i2_over_eye], linewidth = lw, linestyle = '--', color = 'k')
        plt.axvline(x = xdiA[i1_over], linewidth = lw, linestyle = '-', color = 'g', label = 'inicio overshoot')
        plt.axvline(x = xdiA[i2_over], linewidth = lw, linestyle = '-', color = 'g')
        plt.axhline(y = Bmax_over, linewidth = lw, linestyle = '-', color = 'purple', label = 'máximo overshoot')
        #final overshoot
        plt.axvline(x = xdiA[f1_over_eye], linewidth = lw, linestyle = '--', color = 'k')
        plt.axvline(x = xdiA[f2_over_eye], linewidth = lw, linestyle = '--', color = 'k')
        plt.axvline(x = xdiA[f1_over], linewidth = lw, linestyle = '-', color = 'y', label = 'final overshoot')
        plt.axvline(x = xdiA[f2_over], linewidth = lw, linestyle = '-', color = 'y')
        plt.xlim(xlim_over[0], xlim_over[1])
        plt.ylim(ylim[0], ylim[1])
        plt.xticks(np.linspace(xlim_over[0], xlim_over[1], 5, endpoint = True))
        plt.xlabel(r'$c/\omega_{pi}$', fontsize = font_label)
        plt.tick_params(axis = 'both', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
    #    plt.grid(axis = 'both', which = 'both', alpha = grid_alpha, linewidth = lw, linestyle = '--')
        plt.legend(loc = 0, fontsize = font_leg)
        
        
        
        plt.savefig(path_analisis+'subestructuras_espacial_A_{}'.format(shock_date))
        plt.savefig(path_analisis+'subestructuras_espacial_A_{}.pdf'.format(shock_date))
    
#%%    
    
    if tipo == 'out':        
        
        
        plt.figure(9, figsize = figsize)
        plt.title('Delimitacion espacial de subestructuras - Método A', fontsize = font_label)
        plt.subplots_adjust(top=0.92, bottom=0.10, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
        
        plt.subplot(133)
        plt.plot(xdiA, B, linewidth = lw)
        #asintotas de Bu y Bd
        plt.axhline(y = norm_Bu, linewidth = lw, color = 'r', label = r'$B_u$')
        plt.axhspan(ymin = norm_Bu - std_norm_Bu, ymax = norm_Bu + std_norm_Bu, facecolor = 'g', alpha = updown_alpha)
        #inicio foot
        plt.axvline(x = xdiA[i1_foot_eye], linewidth = lw, linestyle = '--', color = 'k', label = 'límites a ojo')
        plt.axvline(x = xdiA[i2_foot_eye], linewidth = lw, linestyle = '--', color = 'k')
        plt.axvline(x = xdiA[i1_foot], linewidth = lw, linestyle = '-', color = 'm', label = 'inicio foot')
        plt.axvline(x = xdiA[i2_foot], linewidth = lw, linestyle = '-', color = 'm')
        #inicio ramp
        plt.axvline(x = xdiA[i1_ramp_eye], linewidth = lw, linestyle = '--', color = 'k')
        plt.axvline(x = xdiA[i2_ramp_eye], linewidth = lw, linestyle = '--', color = 'k')
        plt.axvline(x = xdiA[i1_ramp], linewidth = lw, linestyle = '-', color = 'c', label = 'inicio ramp')
        plt.axvline(x = xdiA[i2_ramp], linewidth = lw, linestyle = '-', color = 'c')
        plt.xlim(xlim_foot[0], xlim_foot[1])
        plt.xticks(np.linspace(xlim_foot[0], xlim_foot[1], 3, endpoint = True))
        plt.tick_params(axis = 'both', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
        plt.xlabel(r'$c/\omega_{pi}$', fontsize = font_label)
    #    plt.grid(axis = 'both', which = 'both', alpha = grid_alpha, linewidth = lw, linestyle = '--')
        plt.legend(loc = 0, fontsize = font_leg)  
        
        
        plt.subplot(132)
        plt.plot(xdeA, B, linewidth = lw, marker = 'o', markersize = msize)
        #asintotas de Bu y Bd
        plt.axhline(y = norm_Bd, linewidth = lw, color = 'r', label = r'$B_d$')
        plt.axhspan(ymin = norm_Bd - std_norm_Bd, ymax = norm_Bd + std_norm_Bd, facecolor = 'g', alpha = updown_alpha)
        #inicio ramp
        plt.axvline(x = xdeA[i1_ramp_eye], linewidth = lw, linestyle = '--', color = 'k', label = 'límites a ojo')
        plt.axvline(x = xdeA[i2_ramp_eye], linewidth = lw, linestyle = '--', color = 'k')
        plt.axvline(x = xdeA[i1_ramp], linewidth = lw, linestyle = '-', color = 'c', label = 'inicio ramp')
        plt.axvline(x = xdeA[i2_ramp], linewidth = lw, linestyle = '-', color = 'c')
        #ajustes ramp
        plt.axvline(x = xdeA[C_high], linewidth = lw, linestyle = 'dotted', color = 'orange', label = 'Centro del choque')
        #inicio overshoot
        plt.axvline(x = xdeA[i1_over_eye], linewidth = lw, linestyle = '--', color = 'k')
        plt.axvline(x = xdeA[i2_over_eye], linewidth = lw, linestyle = '--', color = 'k')
        plt.axvline(x = xdeA[i1_over], linewidth = lw, linestyle = '-', color = 'g', label = 'inicio overshoot')
        plt.axvline(x = xdeA[i2_over], linewidth = lw, linestyle = '-', color = 'g')
        plt.xlim(xlim_ramp[0], xlim_ramp[1])
        plt.xticks(np.linspace(xlim_ramp[0], xlim_ramp[1], 3, endpoint = True))
        plt.tick_params(axis = 'both', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
        plt.xlabel(r'$c/\omega_{pe}$', fontsize = font_label)
    #    plt.grid(axis = 'both', which = 'both', alpha = grid_alpha, linewidth = lw, linestyle = '--')
        plt.legend(loc = 0, fontsize = font_leg)
        
        plt.subplot(131)
        plt.plot(xdiA, B, linewidth = lw, marker = 'o', markersize = msize)
        #asintotas de Bu y Bd
        plt.axhline(y = norm_Bd, linewidth = lw, color = 'r', label = r'$B_d$')
        plt.axhspan(ymin = norm_Bd - std_norm_Bd, ymax = norm_Bd + std_norm_Bd, facecolor = 'g', alpha = updown_alpha)
        #inicio overshoot
        plt.axvline(x = xdiA[i1_over_eye], linewidth = lw, linestyle = '--', color = 'k', label = 'límites a ojo')
        plt.axvline(x = xdiA[i2_over_eye], linewidth = lw, linestyle = '--', color = 'k')
        plt.axvline(x = xdiA[i1_over], linewidth = lw, linestyle = '-', color = 'g', label = 'inicio overshoot')
        plt.axvline(x = xdiA[i2_over], linewidth = lw, linestyle = '-', color = 'g')
        plt.axhline(y = Bmax_over, linewidth = lw, linestyle = '-', color = 'purple', label = 'máximo overshoot')
        #final overshoot
        plt.axvline(x = xdiA[f1_over_eye], linewidth = lw, linestyle = '--', color = 'k')
        plt.axvline(x = xdiA[f2_over_eye], linewidth = lw, linestyle = '--', color = 'k')
        plt.axvline(x = xdiA[f1_over], linewidth = lw, linestyle = '-', color = 'y', label = 'final overshoot')
        plt.axvline(x = xdiA[f2_over], linewidth = lw, linestyle = '-', color = 'y')
        plt.xlim(xlim_over[0], xlim_over[1])
        plt.xticks(np.linspace(xlim_over[0], xlim_over[1], 3, endpoint = True))
        plt.xlabel(r'$c/\omega_{pi}$', fontsize = font_label)
        plt.ylabel(r'$|\vec{B}|$ [nT]', fontsize = font_label)
        plt.tick_params(axis = 'both', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
    #    plt.grid(axis = 'both', which = 'both', alpha = grid_alpha, linewidth = lw, linestyle = '--')
        plt.legend(loc = 0, fontsize = font_leg)
        
        
        
        #plt.savefig(path_analisis+'subestructuras_espacial_A_{}'.format(shock_date))
        #plt.savefig(path_analisis+'subestructuras_espacial_A_{}.pdf'.format(shock_date))
    
#%%

# Metodo B
        
    if tipo == 'in':
                
        
        plt.figure(10, figsize = figsize)
        plt.title('Delimitacion espacial de subestructuras - Método B', fontsize = font_label)
        plt.subplots_adjust(top=0.92, bottom=0.10, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
        
        plt.subplot(131)
        plt.plot(xdiB, B, linewidth = lw)
        #asintotas de Bu y Bd
        plt.axhline(y = norm_Bu, linewidth = lw, color = 'r', label = r'$B_u$')
        plt.axhspan(ymin = norm_Bu - std_norm_Bu, ymax = norm_Bu + std_norm_Bu, facecolor = 'g', alpha = updown_alpha)
        #inicio foot
        plt.axvline(x = xdiB[i1_foot_eye], linewidth = lw, linestyle = '--', color = 'k', label = 'límites a ojo')
        plt.axvline(x = xdiB[i2_foot_eye], linewidth = lw, linestyle = '--', color = 'k')
        plt.axvline(x = xdiB[i1_foot], linewidth = lw, linestyle = '-', color = 'm', label = 'inicio foot')
        plt.axvline(x = xdiB[i2_foot], linewidth = lw, linestyle = '-', color = 'm')
        #inicio ramp
        plt.axvline(x = xdiB[i1_ramp_eye], linewidth = lw, linestyle = '--', color = 'k')
        plt.axvline(x = xdiB[i2_ramp_eye], linewidth = lw, linestyle = '--', color = 'k')
        plt.axvline(x = xdiB[i1_ramp], linewidth = lw, linestyle = '-', color = 'c', label = 'inicio ramp')
        plt.axvline(x = xdiB[i2_ramp], linewidth = lw, linestyle = '-', color = 'c')
        plt.xlim(xlim_foot[0], xlim_foot[1])
        plt.xticks(np.linspace(xlim_foot[0], xlim_foot[1], 3, endpoint = True))
        plt.tick_params(axis = 'both', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
        plt.xlabel(r'$c/\omega_{pi}$', fontsize = font_label)
        plt.ylabel(r'$|\vec{B}|$ [nT]', fontsize = font_label)
    #    plt.grid(axis = 'both', which = 'both', alpha = grid_alpha, linewidth = lw, linestyle = '--')
        plt.legend(loc = 0, fontsize = font_leg)  
        
        
        plt.subplot(132)
        plt.plot(xdeB, B, linewidth = lw)
        #asintotas de Bu y Bd
        plt.axhline(y = norm_Bd, linewidth = lw, color = 'r', label = r'$B_d$')
        plt.axhspan(ymin = norm_Bd - std_norm_Bd, ymax = norm_Bd + std_norm_Bd, facecolor = 'g', alpha = updown_alpha)
        #inicio ramp
        plt.axvline(x = xdeB[i1_ramp_eye], linewidth = lw, linestyle = '--', color = 'k', label = 'límites a ojo')
        plt.axvline(x = xdeB[i2_ramp_eye], linewidth = lw, linestyle = '--', color = 'k')
        plt.axvline(x = xdeB[i1_ramp], linewidth = lw, linestyle = '-', color = 'c', label = 'inicio ramp')
        plt.axvline(x = xdeB[i2_ramp], linewidth = lw, linestyle = '-', color = 'c')
        #ajustes ramp
        plt.axvline(x = xdeB[C_high], linewidth = lw, linestyle = 'dotted', color = 'orange', label = 'Centro del choque')
        #inicio overshoot
        plt.axvline(x = xdeB[i1_over_eye], linewidth = lw, linestyle = '--', color = 'k')
        plt.axvline(x = xdeB[i2_over_eye], linewidth = lw, linestyle = '--', color = 'k')
        plt.axvline(x = xdeB[i1_over], linewidth = lw, linestyle = '-', color = 'g', label = 'inicio overshoot')
        plt.axvline(x = xdeB[i2_over], linewidth = lw, linestyle = '-', color = 'g')
        plt.xlim(xlim_ramp[0], xlim_ramp[1])
        plt.xticks(np.linspace(xlim_ramp[0], xlim_ramp[1], 3, endpoint = True))
        plt.tick_params(axis = 'both', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
        plt.xlabel(r'$c/\omega_{pe}$', fontsize = font_label)
    #    plt.grid(axis = 'both', which = 'both', alpha = grid_alpha, linewidth = lw, linestyle = '--')
        plt.legend(loc = 0, fontsize = font_leg)
        
        plt.subplot(133)
        plt.plot(xdiB, B, linewidth = lw)
        #asintotas de Bu y Bd
        plt.axhline(y = norm_Bd, linewidth = lw, color = 'r', label = r'$B_d$')
        plt.axhspan(ymin = norm_Bd - std_norm_Bd, ymax = norm_Bd + std_norm_Bd, facecolor = 'g', alpha = updown_alpha)
        #inicio overshoot
        plt.axvline(x = xdiB[i1_over_eye], linewidth = lw, linestyle = '--', color = 'k', label = 'límites a ojo')
        plt.axvline(x = xdiB[i2_over_eye], linewidth = lw, linestyle = '--', color = 'k')
        plt.axvline(x = xdiB[i1_over], linewidth = lw, linestyle = '-', color = 'g', label = 'inicio overshoot')
        plt.axvline(x = xdiB[i2_over], linewidth = lw, linestyle = '-', color = 'g')
        plt.axhline(y = Bmax_over, linewidth = lw, linestyle = '-', color = 'purple', label = 'máximo overshoot')
        #final overshoot
        plt.axvline(x = xdiB[f1_over_eye], linewidth = lw, linestyle = '--', color = 'k')
        plt.axvline(x = xdiB[f2_over_eye], linewidth = lw, linestyle = '--', color = 'k')
        plt.axvline(x = xdiB[f1_over], linewidth = lw, linestyle = '-', color = 'y', label = 'final overshoot')
        plt.axvline(x = xdiB[f2_over], linewidth = lw, linestyle = '-', color = 'y')
        plt.xlim(xlim_over[0], xlim_over[1])
        plt.xticks(np.linspace(xlim_over[0], xlim_over[1], 3, endpoint = True))
        plt.xlabel(r'$c/\omega_{pi}$', fontsize = font_label)
        plt.tick_params(axis = 'both', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
    #    plt.grid(axis = 'both', which = 'both', alpha = grid_alpha, linewidth = lw, linestyle = '--')
        plt.legend(loc = 0, fontsize = font_leg)
        
        
        
        plt.savefig(path_analisis+'subestructuras_espacial_B_{}'.format(shock_date))
        plt.savefig(path_analisis+'subestructuras_espacial_B_{}.pdf'.format(shock_date))
    
    
    
    if tipo == 'out':        
        
        
        plt.figure(10, figsize = figsize)
        plt.title('Delimitacion espacial de subestructuras - Método B', fontsize = font_label)
        plt.subplots_adjust(top=0.92, bottom=0.10, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
        
        plt.subplot(133)
        plt.plot(xdiB, B, linewidth = lw)
        #asintotas de Bu y Bd
        plt.axhline(y = norm_Bu, linewidth = lw, color = 'r', label = r'$B_u$')
        plt.axhspan(ymin = norm_Bu - std_norm_Bu, ymax = norm_Bu + std_norm_Bu, facecolor = 'g', alpha = updown_alpha)
        #inicio foot
        plt.axvline(x = xdiB[i1_foot_eye], linewidth = lw, linestyle = '--', color = 'k', label = 'límites a ojo')
        plt.axvline(x = xdiB[i2_foot_eye], linewidth = lw, linestyle = '--', color = 'k')
        plt.axvline(x = xdiB[i1_foot], linewidth = lw, linestyle = '-', color = 'm', label = 'inicio foot')
        plt.axvline(x = xdiB[i2_foot], linewidth = lw, linestyle = '-', color = 'm')
        #inicio ramp
        plt.axvline(x = xdiB[i1_ramp_eye], linewidth = lw, linestyle = '--', color = 'k')
        plt.axvline(x = xdiB[i2_ramp_eye], linewidth = lw, linestyle = '--', color = 'k')
        plt.axvline(x = xdiB[i1_ramp], linewidth = lw, linestyle = '-', color = 'c', label = 'inicio ramp')
        plt.axvline(x = xdiB[i2_ramp], linewidth = lw, linestyle = '-', color = 'c')
        plt.xlim(xlim_foot[0], xlim_foot[1])
        plt.xticks(np.linspace(xlim_foot[0], xlim_foot[1], 3, endpoint = True))
        plt.xlabel(r'$c/\omega_{pi}$', fontsize = font_label)
        plt.tick_params(axis = 'both', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
    #    plt.grid(axis = 'both', which = 'both', alpha = grid_alpha, linewidth = lw, linestyle = '--')
        plt.legend(loc = 0, fontsize = font_leg)  
        
        
        plt.subplot(132)
        plt.plot(xdeB, B, linewidth = lw)
        #asintotas de Bu y Bd
        plt.axhline(y = norm_Bd, linewidth = lw, color = 'r', label = r'$B_d$')
        plt.axhspan(ymin = norm_Bd - std_norm_Bd, ymax = norm_Bd + std_norm_Bd, facecolor = 'g', alpha = updown_alpha)
        #inicio ramp
        plt.axvline(x = xdeB[i1_ramp_eye], linewidth = lw, linestyle = '--', color = 'k', label = 'límites a ojo')
        plt.axvline(x = xdeB[i2_ramp_eye], linewidth = lw, linestyle = '--', color = 'k')
        plt.axvline(x = xdeB[i1_ramp], linewidth = lw, linestyle = '-', color = 'c', label = 'inicio ramp')
        plt.axvline(x = xdeB[i2_ramp], linewidth = lw, linestyle = '-', color = 'c')
        #ajustes ramp
        plt.axvline(x = xdeB[C_high], linewidth = lw, linestyle = 'dotted', color = 'orange', label = 'Centro del choque')
        #inicio overshoot
        plt.axvline(x = xdeB[i1_over_eye], linewidth = lw, linestyle = '--', color = 'k')
        plt.axvline(x = xdeB[i2_over_eye], linewidth = lw, linestyle = '--', color = 'k')
        plt.axvline(x = xdeB[i1_over], linewidth = lw, linestyle = '-', color = 'g', label = 'inicio overshoot')
        plt.axvline(x = xdeB[i2_over], linewidth = lw, linestyle = '-', color = 'g')
        plt.xlim(xlim_ramp[0], xlim_ramp[1])
        plt.xticks(np.linspace(xlim_ramp[0], xlim_ramp[1], 3, endpoint = True))
        plt.tick_params(axis = 'both', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
        plt.xlabel(r'$c/\omega_{pe}$', fontsize = font_label)
    #    plt.grid(axis = 'both', which = 'both', alpha = grid_alpha, linewidth = lw, linestyle = '--')
        plt.legend(loc = 0, fontsize = font_leg)
        
        plt.subplot(131)
        plt.plot(xdiB, B, linewidth = lw)
        #asintotas de Bu y Bd
        plt.axhline(y = norm_Bd, linewidth = lw, color = 'r', label = r'$B_d$')
        plt.axhspan(ymin = norm_Bd - std_norm_Bd, ymax = norm_Bd + std_norm_Bd, facecolor = 'g', alpha = updown_alpha)
        #inicio overshoot
        plt.axvline(x = xdiB[i1_over_eye], linewidth = lw, linestyle = '--', color = 'k', label = 'límites a ojo')
        plt.axvline(x = xdiB[i2_over_eye], linewidth = lw, linestyle = '--', color = 'k')
        plt.axvline(x = xdiB[i1_over], linewidth = lw, linestyle = '-', color = 'g', label = 'inicio overshoot')
        plt.axvline(x = xdiB[i2_over], linewidth = lw, linestyle = '-', color = 'g')
        plt.axhline(y = Bmax_over, linewidth = lw, linestyle = '-', color = 'purple', label = 'máximo overshoot')
        #final overshoot
        plt.axvline(x = xdiB[f1_over_eye], linewidth = lw, linestyle = '--', color = 'k')
        plt.axvline(x = xdiB[f2_over_eye], linewidth = lw, linestyle = '--', color = 'k')
        plt.axvline(x = xdiB[f1_over], linewidth = lw, linestyle = '-', color = 'y', label = 'final overshoot')
        plt.axvline(x = xdiB[f2_over], linewidth = lw, linestyle = '-', color = 'y')
        plt.xlim(xlim_over[0], xlim_over[1])
        plt.xticks(np.linspace(xlim_over[0], xlim_over[1], 3, endpoint = True))
        plt.xlabel(r'$c/\omega_{pi}$', fontsize = font_label)
        plt.ylabel(r'$|\vec{B}|$ [nT]', fontsize = font_label)
        plt.tick_params(axis = 'both', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
    #    plt.grid(axis = 'both', which = 'both', alpha = grid_alpha, linewidth = lw, linestyle = '--')
        plt.legend(loc = 0, fontsize = font_leg)
        
        
        
        plt.savefig(path_analisis+'subestructuras_espacial_B_{}'.format(shock_date))
        plt.savefig(path_analisis+'subestructuras_espacial_B_{}.pdf'.format(shock_date))


#%%------------------------------- GUARDO RESULTADOS ------------------------------

if MODO_subestructuras == 1:
    
    #limites elegidos a ojo
    
    data0 = np.zeros([4,2])
    
    #inicio foot
    data0[0,0] = Ti1_foot_eye
    data0[0,1] = Ti2_foot_eye
    #inicio ramp
    data0[1,0] = Ti1_ramp_eye
    data0[1,1] = Ti2_ramp_eye
    #inicio overshoot
    data0[2,0] = Ti1_over_eye
    data0[2,0] = Ti2_over_eye
    #fin overshoot
    data0[3,0] = Tf1_over_eye
    data0[3,0] = Tf2_over_eye
    
    np.savetxt(path_analisis+'subest_lim_ojo_{}'.format(shock_date), data0, delimiter = '\t',
               header = '\n'.join(['{}'.format(shock_date),'limites a izq y derecha inicio foot',
                                   'limites a izq y derecha inicio ramp',
                                   'limites a izq y derecha inicio overshoot',
                                   'limites a izq y derecha fin overshoot']))
    
    
    #limites automatizados
    
    data1 = np.zeros([6,7])
    
    #incio foot
    data1[0,0] = fact_sigma_foot
    data1[0,1] = i1_foot
    data1[0,2] = ti1_foot
    data1[0,3] = i2_foot 
    data1[0,4] = ti2_foot
    data1[0,5] = i_foot
    data1[0,6] = ti_foot
    
    #inicio overshoot
    data1[1,0] = fact_sigma_iover
    data1[1,1] = i1_over
    data1[1,2] = ti1_over
    data1[1,3] = i2_over
    data1[1,4] = ti2_over
    data1[1,5] = i_over
    data1[1,6] = ti_over
    
    #fin overshoot
    data1[2,0] = fact_sigma_fover
    data1[2,1] = f1_over
    data1[2,2] = tf1_over
    data1[2,3] = f2_over 
    data1[2,4] = tf2_over
    data1[2,5] = f_over
    data1[2,6] = tf_over
    
    #inicio rampa
    data1[3,0] = i1_ramp
    data1[3,1] = f1_rampfit
    data1[3,2] = ti1_ramp
    data1[3,3] = i1_params[0]
    data1[3,4] = i1_params[1]
    data1[3,5] = i1_chired
    data1[4,0] = i2_ramp
    data1[4,1] = f2_rampfit
    data1[4,2] = ti2_ramp
    data1[4,3] = i2_params[0]
    data1[4,4] = i2_params[1]
    data1[4,5] = i2_chired
    data1[5,0] = i_ramp
    data1[5,1] = ti_ramp
    
    np.savetxt(path_analisis+'subest_lim_auto_{}'.format(shock_date), data1, delimiter = '\t',
               header = '\n'.join(['{}'.format(shock_date),'INICIO FOOT: factor de sigma de Bu considerado, indice lim a izq, t lim a izq, indice lim a der, t lim a der, indice centro, t centro',
                                   'INICIO OVERSHOOT: factor de sigma de Bd considerado, indice lim a izq, t lim a izq, indice lim a der, t lim a der, indice centro, t centro',
                                   'FIN OVERSHOOT: factor de sigma de Bd considerado, indice lim a izq, t lim a izq, indice lim a der, t lim a der, indice centro, t centro',
                                   'INICIO RAMPA: indice lim a izq, indice del intervalo fin overshoot usado para el fit, t lim a izq, parametros fit lineal para calcular lim izq, chi red del fit',
                                   'INICIO RAMPA: indice lim a der, indice del intervalo fin overshoot usado para el fit, t lim a der, parametros fit lineal para calcular lim der, chi red del fit',
                                   'INICIO RAMPA: indice centro, t centro']))
    
    
    #parametros del shock
    
    data2 = np.zeros([3,6])
    
    data2[0,0] = C
    data2[0,1:4] = Rc
    
    data2[1,0] = ancho_shock_temp
    data2[1,1:4] = ancho_shock
    data2[1,4] = norm_ancho_shock
    
    data2[2,0] = L
    data2[2,1:4] = N
    data2[2,4] = theta_N
    data2[2,5] = theta_NRc
    
    np.savetxt(path_analisis+'centroshock_anchoshock_normalfit_{}'.format(shock_date), data2, delimiter = '\t',
               header = '\n'.join(['{}'.format(shock_date),'indice centro shock, posicion nave centro shock',
                                   'ancho temp shock, ancho espacial (x,y,z), modulo ancho espacial',
                                   'L del fit, normal fit, angulo Bu y normal fit, angulo Rc y normal fit']))
    
    
