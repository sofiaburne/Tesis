import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from sympy.solvers import solve
from sympy import Symbol
#%%

'''
El fit de MAVEN no lo puedo usar correctamente porque no lo
redefiní para asegurarme de que el punto Rc (centro de la nave en el shock)
esté contenido en la superficie del fit
'''

#def MAVEN_bowshock(x,y,z):
#    #parametros fit MAVEN (Gruesbeck)
#    A, B, C, D, E, F, G, H, I = 0.049, 0.157, 0.153, 0.026, 0.012, 0.051, 0.566, -0.031, 0.019
#    #A_err, B_err, C_err, D_err, E_err, F_err, G_err, H_err, I_err = 7.8E-6, 1.1E-6, 1.0E-6, 3.8E-6, 2.2E-6, 3.1E-6, 1.0E-6, 2.6E-6, 2.1E-6
#    return A*x**2 + B*y**2 + C*z**2 + D*x*y + E*y*z + F*x*z + G*x + H*y + I*z - 1 #defino mi función tal que f(x,y,z)=0

#def norm_fit_MAVEN(x,y,z):
#    A, B, C, D, E, F, G, H, I = 0.049, 0.157, 0.153, 0.026, 0.012, 0.051, 0.566, -0.031, 0.019
#    normal = np.array([float(2*A*x + D*y + F*z + G), float(2*B*y + D*x + E*z + H), float(2*C*z + E*y + F*x + I)])
#    n_fit = normal//np.linalg.norm(normal)
#    return n_fit




'''
usando el fit de Vignes de referencia, calculo mi propio parametro L 
para que el hiperboloide contenga el punto Rc (centro de mi shock)
'''

def L_fit(Rc):
    eps, X_0 = 1.03, 0.64 #excentricidad y foco de Vignes
    L = Symbol('L', positive=True) #me quedo sólo con L positivo
    a,b,c = L/(eps**2 - 1),L/(eps**2 - 1)**(1/2), X_0 + L*eps/(eps**2 - 1)
    eq = ((Rc[0]-c)**2)/a**2 - (Rc[1]**2)/b**2 - (Rc[2]**2)/b**2 - 1    
    l = np.asarray(solve(eq, L))
    Ls = l[(np.abs(l-2.04)).argmin()] #por mas que me quede con L>0 a veces tira dos L posibles, me quedo con el mas parecido al L de Vignes
    return Ls



def error_L_vignes(Rc, err_Rc):
    
    '''
    L es función de Rc
    '''
    
    eps, X_0 = 1.03, 0.64 #excentricidad y foco de Vignes
    
    dRcx_L = (2*Rc[0] - 2*X_0)/(2*np.sqrt(X_0**2 - 2*X_0*Rc[0] + Rc[0]**2 + Rc[1]**2 + Rc[2]**2)) + eps
    dRcy_L = Rc[1]/np.sqrt(X_0**2 - 2*X_0*Rc[0] + Rc[0]**2 + Rc[1]**2 + Rc[2]**2)
    dRcz_L = Rc[2]/np.sqrt(X_0**2 - 2*X_0*Rc[0] + Rc[0]**2 + Rc[1]**2 + Rc[2]**2)
    
    err_cuad_L = (dRcx_L**2)*(err_Rc[0]**2) + (dRcy_L**2)*(err_Rc[1]**2) + (dRcz_L**2)*(err_Rc[2]**2)
    err_L = np.sqrt(err_cuad_L)
    
    return err_L




'''
fit bow shock como hiperboloide
(uso eps y X_0 de Vignes y el L a determinar)
'''

def hiperbola(x,y,z,L):
    eps, X_0 = 1.03, 0.64 #excentricidad y foco de Vignes
    a,b,c = L/(eps**2 - 1),L/(eps**1 - 1)**(1/2), X_0 + L*eps/(eps**2 - 1)
    return ((x-c)**2)/a**2 - (y**2)/b**2 - (z**2)/b**2 - 1




'''
calculo la normal del fit en el centro de mi shock
para el fit de MGS necesito tener definido un L afuera antes de llamar a norm_fit
'''

def norm_fit_MGS(x,y,z,L):
    eps, X_0 = 1.03, 0.64 #excentricidad y foco de Vignes
    a,b,c = L/(eps**2 - 1),L/(eps**1 - 1)**(1/2), X_0 + L*eps/(eps**2 - 1)
    normal = np.array([float(2*(x-c)/a**2), float(-2*y/b**2), float(- 2*z/b**2)])
    n_fit = normal/np.linalg.norm(normal)
    if n_fit[0] < 0:
        n_fit = - n_fit
    return n_fit



def err_N_fit(Rc, err_Rc, L, err_L):
    
#    Nx = (2*(x - (q + L*eps/(eps**2 - 1)))/(L/(eps**2 - 1))**2)/sqrt( (2*(x - q + L*eps/(eps**2 - 1))/(L/(eps**2 - 1))**2)**2 + (-2*y/(L/(eps**1 - 1)**(1/2))**2)**2 +  (-2*z/(L/(eps**1 - 1)**(1/2))**2)**2)
#    Ny = (-2*y/(L/(eps**1 - 1)**(1/2))**2)/sqrt( (2*(x - q + L*eps/(eps**2 - 1))/(L/(eps**2 - 1))**2)**2 + (-2*y/(L/(eps**1 - 1)**(1/2))**2)**2 +  (-2*z/(L/(eps**1 - 1)**(1/2))**2)**2)
#    Nz = (-2*z/(L/(eps**1 - 1)**(1/2))**2)/sqrt( (2*(x - q + L*eps/(eps**2 - 1))/(L/(eps**2 - 1))**2)**2 + (-2*y/(L/(eps**1 - 1)**(1/2))**2)**2 +  (-2*z/(L/(eps**1 - 1)**(1/2))**2)**2)

    eps, X_0 = 1.03, 0.64 #excentricidad y foco de Vignes    
    
    dRcx_Nx = 2*(eps**2 - 1)**2/np.sqrt(np.float64( 4 * (eps**2 -1)**4 * (L*eps/(eps**2 - 1) - X_0 + Rc[0])**2  + 4 * Rc[1] * (eps - 1)**2  + 4 * Rc[2] * (eps - 1)**2 )) - (8*(eps**2 - 1)**6 * (Rc[0] - X_0 - L*eps/(eps**2 - 1)) * (Rc[0] - X_0 + L*eps/(eps**2 - 1)) )/ (4*(eps**2 - 1)**4 * (L*eps/(eps**2 - 1) - X_0 + Rc[0])**2 + 4*Rc[1]**2 * (eps - 1)**2 + 4*Rc[2]**2 * (eps - 1)**2)**(3/2)
    dL_Nx = - ( 4 * (eps**2 - 1)**2 * (-L*eps/(eps**2 - 1) - X_0 + Rc[0]) ) / L*np.sqrt(np.float64( 4*(eps**2 - 1)**4 * (L*eps/(eps**2 - 1) - X_0 + Rc[0])**2 + 4*Rc[1]**2 * (eps - 1)**2 + 4*Rc[2]**2 * (eps - 1)**2 )) - 2*eps*(eps**2 - 1) / np.sqrt(np.float64( 4*(eps**2 - 1)**4 * (L*eps/(eps**2 - 1) - X_0 + Rc[0])**2 + 4*Rc[1]**2 * (eps - 1)**2 + 4*Rc[2]**2 * (eps - 1)**2 )) - ( (eps**2 - 1)**2 * (- L*eps/(eps**2 -1 ) - X_0 + Rc[0]) * (- 16*(eps**2 - 1)**4 * (L*eps/(eps**2 - 1) -X_0 +Rc[0])**2 / L**5 - 16*Rc[1]**2 * (eps - 1)**2 / L**5 - 16*Rc[2]**2 * (eps - 1)**2 / L**5 + 8*eps*(eps**2 - 1)**3 * (L*eps/(eps**2 -1) - X_0 + Rc[0]) / L**4 ) ) / (1/L**4 * ( 4*(eps**2 - 1)**4 * (L*eps/(eps**2 - 1) + X_0 + Rc[0])**2 + 4*Rc[1]**2 * (eps - 1)**2 + 4*Rc[2]**2 * (eps**2 - 1)**2 )**(3/2) )
    
    dRcy_Ny = 8*Rc[1]**2 * (eps**2 - 1)**3 / ( 4*(eps**2 - 1)**4 * (L*eps/(eps**2 - 1) - X_0 + Rc[0])**2 + 4*Rc[1]**2 * (eps - 1)**2 + 4*Rc[2]**2 * (eps - 1)**2 )**(3/2) - 2*(eps - 1) / np.sqrt(np.float64( 4*(eps**2 - 1)**4 * (L*eps/(eps**2 - 1) - X_0 + Rc[0])**2 + 4*Rc[1]**2 * (eps - 1)**2 + 4*Rc[2]**2 * (eps - 1)**2 ) )
    dL_Ny = 4*Rc[1]*(eps - 1) / ( 2*L * np.sqrt( np.float64( (eps**2 - 1)**4 * (Rc[0] - X_0 + L*eps/(eps**2 - 1))**2 + Rc[1]**2 * (eps-1)**2 + Rc[2]**2 * (eps-1)**2 ) ) ) + Rc[1]*(eps-1)*(16/L**5) * ( - (eps**2 - 1)**4 * (L*eps/(eps**2 - 1) - X_0 + Rc[0])**2 - Rc[1]**2 * (eps-1)**2 - Rc[2]**2 * (eps-1)**2 + (L/2)*eps*(eps**2 - 1)**3 * (L*eps/(eps**2 - 1) - X_0 + Rc[0]) ) / (1/L**4) * ( 4*( (eps**2 - 1)**4 * (L*eps/(eps**2 - 1) - X_0 + Rc[0])**2 + Rc[1]**2 * (eps-1)**2 + Rc[2]**2 * (eps-1)**2 ) )*(3/2)
    
    dRcz_Nz = 8*Rc[2]**2 * (eps**2 - 1)**3 / ( 4*(eps**2 - 1)**4 * (L*eps/(eps**2 - 1) - X_0 + Rc[0])**2 + 4*Rc[1]**2 * (eps - 1)**2 + 4*Rc[2]**2 * (eps - 1)**2 )**(3/2) - 2*(eps - 1) / np.sqrt(np.float64( 4*(eps**2 - 1)**4 * (L*eps/(eps**2 - 1) - X_0 + Rc[0])**2 + 4*Rc[1]**2 * (eps - 1)**2 + 4*Rc[2]**2 * (eps - 1)**2 ) )
    dL_Nz = 4*Rc[2]*(eps - 1) / ( 2*L * np.sqrt( np.float64( (eps**2 - 1)**4 * (Rc[0] - X_0 + L*eps/(eps**2 - 1))**2 + Rc[1]**2 * (eps-1)**2 + Rc[2]**2 * (eps-1)**2 ) ) ) + Rc[2]*(eps-1)*(16/L**5) * ( - (eps**2 - 1)**4 * (L*eps/(eps**2 - 1) - X_0 + Rc[0])**2 - Rc[1]**2 * (eps-1)**2 - Rc[2]**2 * (eps-1)**2 + (L/2)*eps*(eps**2 - 1)**3 * (L*eps/(eps**2 - 1) - X_0 + Rc[0]) ) / (1/L**4) * ( 4*( (eps**2 - 1)**4 * (L*eps/(eps**2 - 1) - X_0 + Rc[0])**2 + Rc[1]**2 * (eps-1)**2 + Rc[2]**2 * (eps-1)**2 ) )*(3/2)
    
    err_Nx = np.sqrt( np.float64( (dRcx_Nx**2)*(err_Rc[0]**2) + (dL_Nx**2)*(err_L**2) ) )
    err_Ny = np.sqrt( np.float64( (dRcy_Ny**2)*(err_Rc[1]**2) + (dL_Ny**2)*(err_L**2) ) )
    err_Nz = np.sqrt( np.float64( (dRcz_Nz**2)*(err_Rc[2]**2) + (dL_Nz**2)*(err_L**2) ) )
    
    err_N = np.array([err_Nx, err_Ny, err_Nz])
    
    return err_N




'''
para plotear cono especificando angulo de apertura (theta grados), altura (h) y eje de revolucion
d  = vector describing orientation of axis of cone
format: 2x3 vector with x,y,z coordinates of two points lying on the
axis of the cone. d(1,:) = (x1,y1,z1); d(2,:) = (x2,y2,z2).
'''

def cone(theta, d, h):
    r = h*np.tan(np.pi*theta/180)
    m = h/r
    R,A = np.meshgrid(np.linspace(0,r,11),np.linspace(0,2*np.pi,41))
    
    # Generate cone about Z axis with given aperture angle and height
    X = R*np.cos(A)
    Y = R*np.sin(A)
    Z = m*R
    
    # Cone around the z-axis, point at the origin
    # find coefficients of the axis vector xi + yj + zk
    x = d[1,0]-d[0,0]
    y = d[1,1]-d[0,1]
    z = d[1,2]-d[0,2]
    
    # find angle made by axis vector with X axis
    phix = np.arctan(y/x)
    # find angle made by axis vector with Z axis
    phiz = np.arctan(np.sqrt(x**2 + y**2)/(z))
    
    # Rotate once about Z axis 
    X1 = X*np.cos(phiz)+Z*np.sin(phiz)
    Y1 = Y
    Z1 = -X*np.sin(phiz)+Z*np.cos(phiz)

    # Rotate about X axis
    X3 = X1*np.cos(phix)-Y1*np.sin(phix) + d[0,0]
    Y3 = X1*np.sin(phix)+Y1*np.cos(phix) + d[0,1]
    Z3 = Z1 + d[0,2]
    
    return X3, Y3, Z3




#ecuacion esfera
    
def esfera():
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x_esf = np.cos(u)*np.sin(v)
    y_esf = np.sin(u)*np.sin(v)
    z_esf = np.cos(v)
    return x_esf, y_esf, z_esf




#para plotear funciones 3D del tipo f(x,y,z,param) = 0
    
def plot_implicit(fn, Rc, L, limites=(-2.5, 2.5)):
    xmin, xmax, ymin, ymax, zmin, zmax = limites*3 #limites tiene que ser una tupla (lim_min, lim_max)
    fig = plt.figure(figsize=(30,20))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect("equal")
    A = np.linspace(xmin, xmax, 100) # resolution of the contour
    B = np.linspace(zmin, zmax, 15) # number of slices
    A1,A2 = np.meshgrid(A,A) # grid on which the contour is plotted

    for z in B: # plot contours in the XY plane
        X,Y = A1,A2
        Z = fn(X,Y,z,L)
        cset = ax.contour(X, Y, Z+z, [z], zdir='z')
        # [z] defines the only level to plot for this contour for this value of z

    for y in B: # plot contours in the XZ plane
        X,Z = A1,A2
        Y = fn(X,y,Z, L)
        cset = ax.contour(X, Y+y, Z, [y], zdir='y')

    for x in B: # plot contours in the YZ plane
        Y,Z = A1,A2
        X = fn(x,Y,Z, L)
        cset = ax.contour(X+x, Y, Z, [x], zdir='x')
        
    
    # must set plot limits because the contour will likely extend
    # way beyond the displayed level.  Otherwise matplotlib extends the plot limits
    # to encompass all values in the contour.
    ax.set_zlim3d(zmin,zmax)
    ax.set_xlim3d(xmin,xmax)
    ax.set_ylim3d(ymin,ymax)
    
    ax.set_xlabel(r'$X_{MSO}$ $[R_M]$', fontsize = 20)
    ax.set_ylabel(r'$Y_{MSO}$ $[R_M]$', fontsize = 20)
    ax.set_zlabel(r'$Z_{MSO}$ $[R_M]$', fontsize = 20)
    plt.tick_params(axis='both', which = 'both', length = 4, width = 2, labelsize = 20)
    plt.legend(loc = 0, fontsize = 20)
    plt.show()
    
    return ax