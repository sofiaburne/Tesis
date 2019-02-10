# 0 uso modulo desde otro modulo
# 1 uso modulo y quiero que me haga plots y los guarde
MODO_hipotesisMHD = 0


from mag import shock_date
from delimitacionshock import B, t_mag
from delimitacionshock import t_swia_mom, densidad_swia, temperatura_swia_norm, t_swea, flujosenergia_swea, nivelesenergia_swea
from delimitacionshock import Bu, Bd, norm_Bu, norm_Bd, Vu, Vd, iu_v, fu_v, id_v, fd_v
from subestructuras_calculos import N


from importlib import reload
import numpy as np
import matplotlib.pyplot as plt
import os
from sympy.solvers import solve
from sympy import Symbol
import scipy 
from scipy import optimize



path_analisis = r'C:\Users\sofia\Documents\Facultad\Tesis\Analisis/{}/'.format(shock_date)
if not os.path.exists(path_analisis):
    os.makedirs(path_analisis)

#%%----------------------------------- FUNCIONES GENERALES -------------------------------------------
    
#para calcular Te    
def Te(ind_t_Te, Emin_fit, Emax_fit, amp0, mu0, sigma0, dist_e, energ, t):
    
    '''
    ind_t_Te es indice del tiempo (up/down) en el que fijo la distribucion de electrones
    Emin_fit y Emax_fit son los limites de energias en donde hago el ajuste
    gaussiano
    amp0, mu0, sigma0 son los parametros iniciales del fit
    '''
    
    
    #selecciono rango de energias donde hacer el ajuste (energias van de mayor a menor)
    ind_Emin_fit = (abs(energ - Emin_fit)).argmin()
    ind_Emax_fit = (abs(energ - Emax_fit)).argmin()
    
    
    #defino funcion gaussiana
    def gaussiana(x,a,x0,sigma):
        return a*np.exp(-(x-x0)**2/(2*sigma**2))
    
    
    #fit gaussiano de los datos
    params, cov = scipy.optimize.curve_fit(gaussiana,
                                                    energ[ind_Emax_fit:ind_Emin_fit], dist_e[ind_t_Te,ind_Emax_fit:ind_Emin_fit], [amp0,mu0,sigma0])
    
    f_gauss = gaussiana(energ, params[0], params[1], params[2])
    
    #calculo ancho altura mitad
    aam = max(f_gauss)/2
    
    #estimo Te como energia donde se da el aam
    ind_aam = (abs(f_gauss - aam)).argmin()
    Te = energ[ind_aam]
    
    return Te, ind_aam, aam, f_gauss, params, ind_Emin_fit, ind_Emax_fit

#%%#####################################################################################################
########################################################################################################
########################################################################################################
########################################################################################################
#%%
#plot de densidad y B para ver que es un shock rapido

    
if MODO_hipotesisMHD == 1:
    
    
    figsize = (30,15)
    msize = 8
    lw = 3
    font_label = 30
    font_leg = 26
    ticks_l = 6
    ticks_w = 3
    xarrow_B = 9.75
    xarrow_rho = 9.70
    xlim_B = np.array([t_mag[3020], t_mag[4244]])
    xlim_rho = np.array([t_swia_mom[755], t_swia_mom[1060]])
    ylim_B = 50
    ylim_rho = 200
    
    
    den_u = np.mean(densidad_swia[iu_v:fu_v])
    den_d = np.mean(densidad_swia[id_v:fd_v])
    
    saltoB = norm_Bd/norm_Bu
    saltorho = den_d/den_u
    
    f2, plot1 = plt.subplots(figsize = figsize)
    f2.taight_layout = True
    
    gr1, = plot1.plot(t_mag, B, linewidth = lw, marker ='o', markersize = msize, color = 'C0', label = '$B$')
    plot1.axhline(y = norm_Bu, linewidth = lw, color = 'C1')
    plot1.axhline(y = norm_Bd, linewidth = lw, color = 'C1')
    plot1.annotate('', xy=(xarrow_B, norm_Bd), xycoords='data', xytext=(xarrow_B, norm_Bu), textcoords='data', arrowprops=dict(arrowstyle='<->', connectionstyle='arc3', color='C1', lw=lw))
    plot1.text(xarrow_B-0.01, (norm_Bd + norm_Bu)/2, '$\Delta$B = {}'.format(int(saltoB)), rotation = 90, verticalalignment='center', fontsize = font_leg, color = 'C1', bbox=dict(facecolor='white', edgecolor='None', alpha=1))
    plot1.set_xlabel('Tiempo\n[hora decimal]', fontsize = font_label)
    plot1.set_ylabel('B\n[nT]', fontsize = font_label)
    plt.xlim(xlim_B)
    plt.ylim(ymax = ylim_B)
    plot1.axes.tick_params(axis = 'both', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
    plot1.axes.grid(axis = 'both', which = 'both', alpha = 0.8, linewidth = lw, linestyle = '--')
    
    
    plot2 = plt.twinx(plot1)
    gr2, = plot2.plot(t_swia_mom, densidad_swia, linewidth = lw, marker ='o', markersize = msize,  color = 'C2', label = '$n_p$')
    plot2.axhline(y = den_u, linewidth = lw, color = 'C3')
    plot2.axhline(y = den_d, linewidth = lw, color = 'C3')
    plt.annotate('', xy=(xarrow_rho, den_d), xycoords='data', xytext=(xarrow_rho, den_u), textcoords='data', arrowprops=dict(arrowstyle='<->', connectionstyle='arc3', color='C3', lw=lw))
    plt.text(xarrow_rho-0.01, (den_d + den_u)/2, '$\Delta$n = {}'.format(int(saltorho)), rotation = 90, verticalalignment='center', fontsize = font_leg, color = 'C3', bbox=dict(facecolor='white', edgecolor='None', alpha=1))
    plt.xlim(xlim_rho)
    plt.ylim(ymax = ylim_rho)
    plot2.set_ylabel('$n_p$\n[$cm^{-3}$]', fontsize = font_label)
    plot2.axes.tick_params(axis = 'y', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
    
    plot1.legend(handles = [gr1,gr2], loc = 0, fontsize = font_leg)
    
    f2.savefig(path_analisis+'fast_shock_{}'.format(shock_date))
    f2.savefig(path_analisis+'fast_shock_{}.pdf'.format(shock_date))

#%%
    
#calculo Te
    
#ploteo distribucion de e vs energias a t fijo para elegir parametros y region para hacer fit

tu_Te = 9.75 #*
td_Te = 9.92 #*

ind_tu_Te = (abs(t_swea - tu_Te)).argmin()
ind_td_Te = (abs(t_swea - td_Te)).argmin()



if MODO_hipotesisMHD == 1:
    
    figsize = (30,15)
    font_title = 30
    font_label = 30
    font_leg = 26
    lw = 3
    msize = 8
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    ticks_l = 6
    ticks_w = 3
    grid_alpha = 0.8
    
    ylim_min_u = 1e6
    ylim_min_d = 1e6
    
    
    plt.figure(11, figsize = figsize)
       
    plt.subplot(121)
    plt.title(r' Upstream - t = {} hora decimal'.format(round(t_swea[ind_tu_Te],3)), fontsize = font_title)
    plt.plot(nivelesenergia_swea, flujosenergia_swea[ind_tu_Te,:], linewidth = lw, marker = 'o', markersize = msize, color = colors[0])
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'Distribución de electrones', fontsize = font_label)
    plt.xlabel('Energía [eV]', fontsize = font_label)
    plt.ylim(ymin = ylim_min_u)
    plt.tick_params(axis = 'both', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
    plt.grid(which = 'major', axis = 'both', linewidth = lw, linestyle = '--', alpha = grid_alpha)
    
    plt.subplot(122)
    plt.title(r' Downstream - t = {} hora decimal'.format(round(t_swea[ind_td_Te],3)), fontsize = font_title)
    plt.plot(nivelesenergia_swea, flujosenergia_swea[ind_td_Te,:], linewidth = lw, marker = 'o', markersize = msize, color = colors[0])
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Energía [eV]', fontsize = font_label)
    plt.ylim(ymin = ylim_min_d)
    plt.tick_params(axis = 'both', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
    plt.grid(which = 'major', axis = 'both', linewidth = lw, linestyle = '--', alpha = grid_alpha)
    
    


Te_u, ind_aam_u, aam_u, f_gauss_u, params_u, ind_Emin_fit_u, ind_Emax_fit_u = Te(ind_tu_Te, 4, 43, 2e8, 10, 1, dist_e = flujosenergia_swea, energ = nivelesenergia_swea, t = t_swea)  #* 
Te_d, ind_aam_d, aam_d, f_gauss_d, params_d, ind_Emin_fit_d, ind_Emax_fit_d = Te(ind_td_Te, 16, 88, 6e8, 30, 1, dist_e = flujosenergia_swea, energ = nivelesenergia_swea, t = t_swea)  #*

if  MODO_hipotesisMHD == 1:
    
    
    plt.figure(12, figsize = figsize)
    
    plt.subplot(121)
    plt.title(r' Upstream - t = {} hora decimal'.format(round(t_swea[ind_tu_Te],3)), fontsize = font_title)
    plt.plot(nivelesenergia_swea, flujosenergia_swea[ind_tu_Te,:], linewidth = lw, marker = 'o', markersize = msize, color = colors[0])
    plt.plot(nivelesenergia_swea, f_gauss_u, linewidth = lw, marker = 'o', markersize = msize, color = colors[1])
    
    plt.axvspan(xmin = nivelesenergia_swea[ind_Emin_fit_u], xmax = nivelesenergia_swea[ind_Emax_fit_u], facecolor = colors[2], alpha = 0.3)
    plt.axhline(y = aam_u, linewidth = lw, color = colors[3], label = 'FWHM')
    plt.axvline(x = Te_u, linewidth = lw, color = colors[4], label = r'$T_e$ = {} eV'.format(round(np.float64(Te_u),2)))
    plt.axvline(x = params_u[1], linewidth = lw, color = colors[5], label = r'$\mu$')
    
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'Distribución de electrones', fontsize = font_label)
    plt.xlabel('Energía [eV]', fontsize = font_label)
    plt.ylim(ymin = ylim_min_u)
    plt.tick_params(axis = 'both', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
    plt.legend(loc = 0, fontsize = font_leg)
    plt.grid(which = 'major', axis = 'both', linewidth = lw, linestyle = '--', alpha = grid_alpha)
    
    
    plt.subplot(122)
    plt.title(r' Downstream - t = {} hora decimal'.format(round(t_swea[ind_td_Te],3)), fontsize = font_title)
    plt.plot(nivelesenergia_swea, flujosenergia_swea[ind_td_Te,:], linewidth = lw, marker = 'o', markersize = msize, color = colors[0])
    plt.plot(nivelesenergia_swea, f_gauss_d, linewidth = lw, marker = 'o', markersize = msize, color = colors[1])
    
    plt.axvspan(xmin = nivelesenergia_swea[ind_Emin_fit_d], xmax = nivelesenergia_swea[ind_Emax_fit_d], facecolor = colors[2], alpha = 0.3)
    plt.axhline(y = aam_d, linewidth = lw, color = colors[3], label = 'FWHM')
    plt.axvline(x = Te_d, linewidth = lw, color = colors[4], label = r'$T_e$ = {} eV'.format(round(np.float64(Te_d),2)))
    plt.axvline(x = params_d[1], linewidth = lw, color = colors[5], label = r'$\mu$')
    
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Energía [eV]', fontsize = font_label)
    plt.ylim(ymin = ylim_min_d)
    plt.tick_params(axis = 'both', which = 'both', length = ticks_l, width = ticks_w, labelsize = font_label)
    plt.legend(loc = 0, fontsize = font_leg)
    plt.grid(which = 'major', axis = 'both', linewidth = lw, linestyle = '--', alpha = grid_alpha)
    
    plt.savefig(path_analisis+'Te{}'.format(shock_date))
    plt.savefig(path_analisis+'Te{}.pdf'.format(shock_date)) 
        
    
    
    

#%%
#chequeo relaciones RH



#elijo respecto a que normal calculo las conservaciones
norm = np.copy(N) #*     ahora elegi la del fit


#paso todo lo que use en esta seccion a SI


# vel en m/s (tang y normales)
U_u = Vu*(1e3)
U_d = Vd*(1e3)
U_un = np.dot(norm,U_u)
U_dn = np.dot(norm,U_d)
U_ut = (U_u - U_un*norm)
U_dt = (U_d - U_dn*norm)

#B en T (tang y normales)
B_u = Bu*(1e-9)
B_d = Bd*(1e-9)
B_un = np.dot(norm,B_u)
B_dn = np.dot(norm,B_d)
B_ut = (B_u - B_un*norm)
B_dt = (B_d - B_dn*norm)

#densidad en kg/m^3
mp = 1.67e-27 #masa del proton en kg
#mp = 1.5e-10 #masa del proton en joules/c^2
densnum_u = np.mean(densidad_swia[min(iu_v,fu_v):max(iu_v,fu_v)])*(1e6) #1/m^3
densnum_d = np.mean(densidad_swia[min(id_v,fd_v):max(id_v,fd_v)])*(1e6) #1/m^3
rho_u = mp*densnum_u
rho_d = mp*densnum_d

#presion suponiendo gas ideal (en Pa=J/m^3)
kB = 1.38e-23 #cte de Boltzmann en J/K
#por ahora supongo T = 2*Ti, tendria que ser T=Ti+Te
Tu = 2*np.mean(temperatura_swia_norm[min(iu_v,fu_v):max(iu_v,fu_v)])*(11604.5) #en K
Td = 2*np.mean(temperatura_swia_norm[min(id_v,fd_v):max(id_v,fd_v)])*(11604.5) #en K
##con Te estimadas de las distribuciones
#Tu = (np.mean(temperatura_swia_norm[min(iu_v,fu_v):max(iu_v,fu_v)])+Te_u)*(11604.5) #en K
#Td = (np.mean(temperatura_swia_norm[min(id_v,fd_v):max(id_v,fd_v)])+Te_d)*(11604.5) #en K
Pu = densnum_u*kB*Tu
Pd = densnum_d*kB*Td



#numeros de Mach

mu = (np.pi*4)*(1e-7) #permeabilidad mag del vacio en Wb/Am=mT/A

v_alfv = (np.linalg.norm(B_u)/np.sqrt(mu*rho_u))*(1e-3) # km/s
v_alfv_2 = (np.linalg.norm(B_u)/np.sqrt(mu*mp*(densnum_u + 0.05*densnum_u)))*(1e-3) # km/s   considero n=n_p+n_He, con n_He = 5%n_p

v_cs = (np.sqrt((Pu/(2*rho_u))*(5/3)))*(1e-3) # km/s
#si no les aplico sqrt dan ordenes de magnitud sin sentido:
#v_cs_2 = ((5/3)*(kB*Tu)/mp)*(1e-3) #km/s
#v_cs_3 = ((3*kB*Tu/2)/mp)*(1e-3) #km/s

M_c = 2.7 #M_A critico para theta_Bun = 90, para angulos menores decrese
M_A = np.linalg.norm(Vu)/v_alfv_2
M_cs = np.linalg.norm(Vu)/v_cs
M_f = np.linalg.norm(Vu)/np.sqrt(v_alfv_2**2 + v_cs**2)

#Mach locales
M_A_loc = np.abs(np.dot(Vu,norm))/v_alfv_2
M_cs_loc = np.abs(np.dot(Vu,norm))/v_cs
M_f_loc = np.abs(np.dot(Vu,norm))/np.sqrt(v_alfv_2**2 + v_cs**2)


#beta del plasma upstream
beta = Pu/(np.linalg.norm(B_u)**2/(2*mu))


#chequeo si se cumple la hipotesis de evolucion adiabatica (gamma da 5/3)
G = Symbol('G')
eq_adiab = Pu*rho_u**G - (Pd*rho_d**G)
gam = solve(eq_adiab, G)




#relaciones RH en porcentaje (100 = se cumple perfectamente)

#conservacion de la masa
cons_masa_u = np.abs(rho_u*U_un)
cons_masa_d = np.abs(rho_d*U_dn)
cons_masa = np.min([cons_masa_u,cons_masa_d])/np.max([cons_masa_u,cons_masa_d])*100

#consevacion del impulso normal al shock
cons_impul_n_u = np.abs(rho_u*U_un**2 + Pu + B_u**2/(2*mu))
cons_impul_n_d = np.abs(rho_d*U_dn**2 + Pd + B_d**2/(2*mu))
cons_impul_n = np.empty_like(cons_impul_n_u)
for i in range(len(cons_impul_n_u)):
    cons_impul_n[i] = np.min([cons_impul_n_d[i],cons_impul_n_u[i]])/np.max([cons_impul_n_d[i],cons_impul_n_u[i]])*100

#conservacion del impulso tangencial al shock
cons_impul_t_u = np.abs(rho_u*U_un*U_ut - B_un/mu*B_ut)
cons_impul_t_d = np.abs(rho_d*U_dn*U_dt - B_dn/mu*B_dt)
cons_impul_t = np.empty_like(cons_impul_t_u)
for i in range(len(cons_impul_t_u)):
    cons_impul_t[i] = np.min([cons_impul_t_d[i],cons_impul_t_u[i]])/np.max([cons_impul_t_d[i],cons_impul_t_u[i]])*100

#consevacion de la energia
gamma = 5/3
cons_energ_u = np.abs(rho_u*U_un*(1/2*U_u**2 + gamma/(gamma-1)*Pu/rho_u) + U_un*B_u**2/mu - np.dot(U_u,B_u)*B_un/mu)
cons_energ_d = np.abs(rho_d*U_dn*(1/2*U_d**2 + gamma/(gamma-1)*Pd/rho_d) + U_dn*B_d**2/mu - np.dot(U_d,B_d)*B_dn/mu)
cons_energ = np.empty_like(cons_energ_u)
for i in range(len(cons_energ)):
    cons_energ[i] = np.min([cons_energ_d[i],cons_energ_u[i]])/np.max([cons_energ_d[i],cons_energ_u[i]])*100

#conservacion de componente normal de B
cons_Bn_u = np.abs(B_un)
cons_Bn_d = np.abs(B_dn)
cons_Bn = np.min([cons_Bn_d,cons_Bn_u])/np.max([cons_Bn_d,cons_Bn_u])*100

#conservacion de campo electrico tang
cons_Et_u = np.abs(U_un*B_ut - B_un*U_ut)
cons_Et_d = np.abs(U_dn*B_dt - B_dn*U_dt)
cons_Et = np.empty_like(cons_Et_u)
for i in range(len(cons_Et)):
     cons_Et[i] = np.min([cons_Et_d[i],cons_Et_u[i]])/np.max([cons_Et_d[i],cons_Et_u[i]])*100

#hipotesis de coplanaridad
hipt_copl_B = np.dot(norm,np.cross(B_u,B_d))


#%%------------------------------- GUARDO RESULTADOS ------------------------------

if MODO_hipotesisMHD == 1:
    
    datos6 = np.zeros([14,5])
    
    #normal de referencia
    datos6[0,0:3] = norm
    
    #Tu Td Pu Pd rho_u rho_d
    datos6[1,0] = Tu
    datos6[1,1] = Td
    datos6[1,2] = Pu
    datos6[1,3] = Pd
    datos6[1,4] = rho_u
    datos6[1,4] = rho_d
    
    #gamma de evolución adiabatica
    datos6[2,0] = gam
    
    #velocidades
    datos6[3,0] = v_alfv
    datos6[3,1] = v_cs
    datos6[3,2] = v_cs_2
    datos6[3,3] = v_cs_3
    
    #numeros de Mach
    datos6[4,0] = M_A
    datos6[4,1] = M_cs
    datos6[4,2] = M_f
    datos6[4,3] = M_c
    
    #beta upstream
    datos6[5,0] = beta
    
    #conservaciones
    datos6[6,0] = cons_masa
    datos6[7,0:3] = cons_impul_n
    datos6[8,0:3] = cons_impul_t
    datos6[9,0:3] = cons_energ
    datos6[10,0] = cons_Bn
    datos6[11,0:3] = cons_Et
    
    #hipotesis teo coplanaridad
    datos6[12,0] = hipt_copl_B
    
    #salto B, salto rho
    datos6[13,0] = saltoB
    datos6[13,1] = saltorho
    
    np.savetxt(path_analisis+'hipotesis_MHD_shock_{}'.format(shock_date), datos6, delimiter = '\t',
               header = '\n'.join(['{}'.format(shock_date),'normal de ref usada en calculos',
                                                     'Tu Td [K] Pu Pd [Pa], rho_u rho_d [kg/m^3]',
                                                     'gamma si el sist evolucionara adiabaticamente',
                                                     'v_alf v_cs v_cs_2 v_cs_3',
                                                     'M_Alfv M_sonico M_rapido M_critico', 'beta',
                                                     'conservacion masa',
                                                     'conservacion impulso norm',
                                                     'conservacion impulso tang',
                                                     'conservacion energia',
                                                     'conservacion Bn',
                                                     'conservacion campo electrico tang',
                                                     'hipotesis teo coplanaridad [nT]',
                                                     'salto B salto rho']))
