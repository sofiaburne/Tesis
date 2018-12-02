
from mag import shock_date
from delimitacion_shock import B, t_mag
from delimitacion_shock import t_swia_mom, densidad_swia, temperatura_swia_norm
from delimitacion_shock import N, Bu, Bd, Vu, Vd, iu_v, fu_v, id_v, fd_v

from importlib import reload
import numpy as np
import matplotlib.pyplot as plt
import os
from sympy.solvers import solve
from sympy import Symbol


path_analisis = r'C:\Users\sofia\Documents\Facultad\Tesis\Analisis/{}/'.format(shock_date)
if not os.path.exists(path_analisis):
    os.makedirs(path_analisis)


#%%

#plot de densidad y B para ver que es un shock rapido

f2, plot1 = plt.subplots()

gr1, = plot1.plot(t_mag, B, linewidth = 2, marker ='o', markersize = '4', color = 'C0', label = '$B$')
plot1.set_xlabel('Tiempo\n[hora decimal]', fontsize = 30)
plot1.set_ylabel('B\n[nT]', fontsize = 30)
plt.xlim(t_mag[3020], t_mag[4244])
plot1.axes.tick_params(axis = 'both', which = 'both', length = 6, width = 3, labelsize = 30)
plot1.axes.grid(axis = 'both', which = 'both', alpha = 0.8, linewidth = 2, linestyle = '--')

plot2 = plt.twinx(plot1)
gr2, = plot2.plot(t_swia_mom, densidad_swia, linewidth = 2, marker ='o', markersize = '4',  color = 'C2', label = '$n_p$')
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
B_u = Bu*(1e-9)
B_d = Bd*(1e-9)
B_un = np.dot(norm,B_u)
B_dn = np.dot(norm,B_d)
B_ut = (B_u - B_un*norm)
B_dt = (B_d - B_dn*norm)

#densidad en kg/m^3
mp = 1.67e-27 #masa del proton en kg
#mp = 1.5e-10 #masa del proton en joules/c^2
densnum_u = np.mean(densidad_swia[iu_v:fu_v])*(1e-6) #1/m^3
densnum_d = np.mean(densidad_swia[id_v:fd_v])*(1e-6) #1/m^3
rho_u = mp*densnum_u
rho_d = mp*densnum_d

#presion suponiendo gas ideal (en Pa=J/m^3)
kB = 1.38e-23 #cte de Boltzmann en J/K
#por ahora supongo T = 2*Ti, tendria que ser T=Ti+Te
Tu = 2*np.mean(temperatura_swia_norm[iu_v:fu_v])*(11604.5) #en K
Td = 2*np.mean(temperatura_swia_norm[id_v:fd_v])*(11604.5) #en K
Pu = densnum_u*kB*Tu
Pd = densnum_d*kB*Td


#numeros de Mach

mu = (np.pi*4)*(1e-7) #permeabilidad mag del vacio en Wb/Am=mT/A
v_alfv = (np.linalg.norm(B_u)/np.sqrt(mu*rho_u))*(1e-3) # km/s
v_cs = (np.sqrt((Pu/rho_u)*(5/3)))*(1e-3) # km/s
#v_cs = ((5/3)*(kB*Tu)/mp)*(1e-3) #km/s

M_A = np.linalg.norm(Vu)/v_alfv
M_cs = np.linalg.norm(Vu)/v_cs
M_f = np.linalg.norm(Vu)/np.sqrt(v_alfv**2 + v_cs**2)

M_c = 2.7 #M_A critico para theta_Bun = 90, para angulos menores decrese

#beta del plasma upstream
beta = Pu/(B_u**2/2*mu)

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


#%%------------------------------- GUARDO RESULTADOS ------------------------------


#datos6 = np.zeros([10,5])
#
##normal de referencia
#datos6[0,0:3] = norm
#
##Tu Td Pu Pd
#datos6[1,0] = Tu
#datos6[1,1] = Td
#datos6[1,2] = Pu
#datos6[1,3] = Pd
#
##numeros de Mach
#datos6[2,0] = M_A
#datos6[2,1] = M_cs
#datos6[2,2] = M_f
#datos6[2,3] = M_c
#
##beta upstream
#datos6[3,0] = beta
#
##conservaciones
#datos6[4,0] = cons_masa
#datos6[5,0:3] = cons_impul_n
#datos6[6,0:3] = cons_impul_t
#datos6[7,0:3] = cons_energ
#datos6[8,0] = cons_Bn
#datos6[9,0:3] = cons_Et
#
##hipotesis teo coplanaridad
#datos6[10,0] = hipt_copl_B
#
#np.savetxt(path_analisis+'hipotesis_MHD_shock_{}'.format(shock_date), datos6, delimiter = '\t',
#           header = '\n'.join(['{}'.format(shock_date),'normal de ref usada en calculos',
#                                                 'Tu Td [K] Pu Pd [Pa]',
#                                                 'M_Alfv M_sonico M_rapido M_critico', 'beta',
#                                                 'conservacion masa',
#                                                 'conservacion impulso norm',
#                                                 'conservacion impulso tang',
#                                                 'conservacion energia',
#                                                 'conservacion Bn',
#                                                 'conservacion campo electrico tang',
#                                                 'hipotesis teo coplanaridad [nT]']))
