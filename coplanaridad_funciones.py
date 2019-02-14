import numpy as np
import random
import statistics as st
#%%

#normales formulas teo coplanaridad
def norm_coplanar(Bd,Bu,Vd,Vu):
    '''
    Calcula todos los tipos de normales coplanares que hay (Analysis Methods Cap10).
    nB falla en 0 y 90
    nV es una formula aprox que vale para Mach muy grandes para angulos cerca de 0 y 90.
    '''
   
    nB = (np.cross(np.cross(Bd,Bu),(Bd-Bu)))/(np.linalg.norm(np.cross(np.cross(Bd,Bu),(Bd-Bu))))
    if nB[0] < 0: #en SR MSO el versor x apunta hacia el sol, entonces para tener normal externa nx>0
        nB = - nB
        
    nBuV = (np.cross(np.cross(Bu,(Vd-Vu)),(Bd-Bu)))/(np.linalg.norm(np.cross(np.cross(Bu,(Vd-Vu)),(Bd-Bu))))
    if nBuV[0] < 0:
        nBuV = - nBuV
    
    nBdV = (np.cross(np.cross(Bd,(Vd-Vu)),(Bd-Bu)))/(np.linalg.norm(np.cross(np.cross(Bd,(Vd-Vu)),(Bd-Bu))))
    if nBdV[0] < 0:
        nBdV = - nBdV
    
    nBduV = (np.cross(np.cross((Bd-Bu),(Vd-Vu)),(Bd-Bu)))/(np.linalg.norm(np.cross(np.cross((Bd-Bu),(Vd-Vu)),(Bd-Bu))))
    if nBduV[0] < 0:
        nBduV = - nBduV
    
    nV = (Vd-Vu)/np.linalg.norm(Vd-Vu)
    if nV[0] < 0:
        nV = - nV
   
    return nB, nBuV, nBdV, nBduV, nV




def err_norm_coplanar(Bd, Bu, Vd, Vu, err_Bu, err_Bd, err_Vu, err_Vd):
    
    def err_n(x, y, z, err_x, err_y, err_z):
        
        '''
        Pienso a n como n = (X x Y) x Z / | X x Y) x Z |
        nx = ((x2*y0 - x0*y2)*z2 - (x0*y1 - x1*y0)*z1) / ( ((x2*y0 - x0*y2)*z2 - (x0*y1 - x1*y0)*z1)^2 + ((x0*y1 - x1*y0)*z0 - (x1*y2 - x2*y1)*z2)^2 + ((x1*y2 - x2*y1)*z1 - (x2*y0 - x0*y2)*z0)^2 )^(1/2)
        '''
        
        den = ( (y[2]*(x[0]*z[0] + x[1]*z[1]) - x[2]*(y[0]*z[0] + y[1]*z[1]))**2 + (y[1]*(x[0]*z[0] + x[2]*z[2]) - x[1]*(y[0]*z[0] + y[2]*z[2]))**2 + (x[1]*y[0]*z[1] + x[2]*y[0]*z[2] - x[0]*(y[1]*z[1] + y[2]*z[2]))**2)**(3/2)        
            
        dx0_nx = (x[2]*y[1] - x[1]*y[2]) * (y[0]*z[0] + y[1]*z[1] + y[2]*z[2]) * (x[0]*z[0]*(-y[2]*z[1] + y[1]*z[2]) + x[2]*(y[0]*z[0]*z[1] + y[1]*(z[1]**2 + z[2]**2)) - x[1]*(y[0]*z[0]*z[2] + y[2]*(z[1]**2 + z[2]**2)) ) / (-1)*den
        dx1_nx = (x[2]*y[0] - x[0]*z[2]) * (y[0]*z[0] + y[1]*z[1] + y[2]*z[2]) * (x[0]*z[0]*(-y[2]*z[1] + y[1]*z[2]) + x[2]*(y[0]*z[0]*z[1] + y[1]*(z[1]**2 + z[2]**2)) - x[1]*(y[0]*z[0]*z[2] + y[2]*(z[1]**2 + z[2]**2)) ) / den
        dx2_nx = (x[1]*y[0] - x[0]*y[1]) * (y[0]*z[0] + y[1]*z[1] + y[2]*z[2]) * ( x[1]*y[0]*z[0]*z[2] + x[0]*z[0]*(y[2]*z[1] - y[1]*z[2]) + x[1]*y[2]*(z[1]**2 + z[2]**2) - x[2]*(y[0]*z[0]*z[1] + y[1]*(z[1]**2 + z[2]**2)) ) / den
    
        dy0_nx = (x[2]*y[1] - x[1]*y[2]) * (x[0]*z[0] + x[1]*z[1] + x[2]*z[2]) * ( x[0]*z[0]*(-y[2]*z[1] + y[1]*z[2]) + x[2]*(y[0]*z[0]*z[1] + y[1]*(z[1]**2 + z[2]**2)) - x[1]*(y[0]*z[0]*z[2] + y[2]*(z[1]**2 + z[2]**2)) ) / den
        dy1_nx = (x[2]*y[0] - x[0]*y[2]) * (x[0]*z[0] + x[1]*z[1] + x[2]*z[2]) * ( x[0]*z[0]*(-y[2]*z[1] + y[1]*z[2]) + x[2]*(y[0]*z[0]*z[1] + y[1]*(z[1]**2 + z[2]**2)) - x[1]*(y[0]*z[0]*z[2] + y[2]*(z[1]**2 + z[2]**2)) ) / (-1)*den
        dy2_nx = (-x[1]*y[0] + x[0]*y[1]) * (x[0]*z[0] + x[1]*z[1] + x[2]*z[2]) * ( x[1]*y[0]*z[0]*z[2] + x[0]*z[0]*(y[2]*z[1] - y[1]*z[2]) + x[1]*y[2]*(z[1]**2 + z[2]**2) - x[2]*(y[0]*z[0]*z[1] + y[1]*(z[1]**2 + z[2]**2)) ) / den
        
        dz0_nx = - (x[1]*y[0]*z[1] + x[2]*y[0]*z[2] - x[0]*(y[1]*z[1] + y[2]*z[2])) * (x[0]**2 * (y[1]**2 + y[2]**2)*z[0] + x[2]**2 * y[0]*(y[0]*z[0] + y[1]*z[1]) + x[1]**2 * y[0]*(y[0]*z[0] + y[2]*z[2]) + x[0]*x[2]*(- 2*y[0]*y[2]*z[0] + y[1]*(- y[2]*z[1] + y[1]*z[2])) - x[1]*(2*x[0]*y[0]*y[1]*z[0] + x[0]*y[2]*(- y[2]*z[1] + y[1]*z[2]) + x[2]*y[0]*(y[2]*z[1] + y[1]*z[2]))) / den
        dz1_nx = (- y[1]*(x[0]*z[0] + x[2]*z[2]) + x[1]*(y[0]*z[0] + y[2]*z[2])) * ( x[0]**2 * (y[1]**2 + y[2]**2)*z[0] + x[2]**2 * y[0] * (y[0]*z[0] + y[1]*z[1]) + x[1]**2 * y[0] * (y[0]*z[0] + y[2]*z[2]) + x[0]*x[2]*(- 2*y[0]*y[2]*z[0] + y[1]*(- y[2]*z[1] + y[1]*z[2])) - x[1]*(2*x[0]*y[0]*y[1]*z[0] + x[0]*y[2]*(- y[2]*z[1] + y[1]*z[2]) + x[2]*y[0]*(y[2]*z[1] + y[1]*z[2])) ) / den
        dz2_nx = (- y[2] * (x[0]*z[0] + x[1]*z[1]) + x[2]*(y[0]*z[0] + y[1]*z[1])) * ( x[0]**2 * (y[1]**2 + y[2]**2)*z[0] + x[2]**2 * y[0] * (y[0]*z[0] + y[1]*z[1]) + x[1]**2 * y[0] * (y[0]*z[0] + y[2]*z[2]) + x[0]*x[2]*(- 2*y[0]*y[2]*z[0] + y[1]*(- y[2]*z[1] + y[1]*z[2])) - x[1]*(2*x[0]*y[0]*y[1]*z[0] + x[0]*y[2]*(- y[2]*z[1] + y[1]*z[2]) + x[2]*y[0]*(y[2]*z[1] + y[1]*z[2])) ) / den
        
        dx0_ny = (x[2]*y[1] - x[1]*y[2]) * (y[0]*z[0] + y[1]*z[1] + y[2]*z[2]) * ( x[1]*z[1] * (- y[2]*z[0] + y[0]*z[2]) + x[2]*(y[1]*z[0]*z[1] + y[0]*(z[0]**2 + z[2]**2)) - x[0]*(y[1]*z[1]*z[2] + y[2]*(z[0]**2 + z[2]**2)) ) / den
        dx1_ny = (x[2]*y[0] - x[0]*y[2]) * (y[0]*z[0] + y[1]*z[1] + y[2]*z[2]) * ( x[1]*z[1] * (- y[2]*z[0] + y[0]*z[2]) + x[2]*(y[1]*z[0]*z[1] + y[0]*(z[0]**2 + z[2]**2)) - x[0]*(y[1]*z[1]*z[2] + y[2]*(z[0]**2 + z[2]**2)) ) / (-1)*den
        dx2_ny = (x[1]*y[0] - x[0]*y[1]) * (y[0]*z[0] + y[1]*z[1] + y[2]*z[2]) * ( x[1]*z[1] * (- y[2]*z[0] + y[0]*z[2]) + x[2]*(y[1]*z[0]*z[1] + y[0]*(z[0]**2 + z[2]**2)) - x[0]*(y[1]*z[1]*z[2] + y[2]*(z[0]**2 + z[2]**2)) ) / den
        
        dy0_ny = (x[2]*y[1] - x[1]*y[2]) * (x[0]*z[0] + x[1]*z[1] + x[2]*z[2]) * ( x[0]*y[1]*z[1]*z[2] + x[1]*z[1]*(y[2]*z[0] - y[0]*z[2]) + x[0]*y[2]*(z[0]**2 + z[2]**2) - x[2]*(y[1]*z[0]*z[1] + y[0]*(z[0]**2 + z[2]**2)) ) / den
        dy1_ny = (x[2]*y[0] - x[0]*y[2]) * (x[0]*z[0] + x[1]*z[1] + x[2]*z[2]) * ( x[1]*z[1] * (- y[2]*z[0] + y[0]*z[2]) + x[2] * (y[1]*z[0]*z[1] + y[0]*(z[0]**2 + z[2]**2)) - x[0]*(y[1]*z[1]*z[2] + y[2]*(z[0]**2 + z[2]**2)) ) / den
        dy2_ny = (x[1]*y[0] - x[0]*y[1]) * (x[0]*z[0] + x[1]*z[1] + x[2]*z[2]) * ( x[1]*z[1] * (- y[2]*z[0] + y[0]*z[2]) + x[2] * (y[1]*z[0]*z[1] + y[0]*(z[0]**2 + z[2]**2)) - x[0]*(y[1]*z[1]*z[2] + y[2]*(z[0]**2 + z[2]**2)) ) / (-1)*den
        
        dz0_ny = - (x[1]*y[0]*z[1] + x[2]*y[0]*z[2] - x[0]*(y[1]*z[1] + y[2]*z[2])) * ( x[1]**2 * (y[0]**2 + y[2]**2)*z[1] + x[2]**2 * y[1]*(y[0]*z[0] + y[1]*z[1]) - x[2]*y[2]*(x[1]*y[0]*z[0] + x[0]*y[1]*z[0] + 2*x[1]*y[1]*z[1]) + x[2]*y[0]*(x[1]*y[0] - x[0]*y[1])*z[2] + x[0]**2 * y[1]*(y[1]*z[1] + y[2]*z[2]) + x[0]*x[1]*(y[2]**2 * z[0] - 2*y[0]*y[1]*z[1] - y[0]*y[2]*z[2]) ) / den
        dz1_ny = (- y[1]*(x[0]*z[0] + x[2]*z[2]) + x[1]*(y[0]*z[0] + y[2]*z[2])) * ( x[1]**2 * (y[0]**2 + y[2]**2)*z[1] + x[2]**2 * y[1]*(y[0]*z[0] + y[1]*z[1]) - x[2]*y[2]*(x[1]*y[0]*z[0] + x[0]*y[1]*z[0] + 2*x[1]*y[1]*z[1]) + x[2]*y[0]*(x[1]*y[0] - x[0]*y[1])*z[2] + x[0]**2 * y[1]*(y[1]*z[1] + y[2]*z[2]) + x[0]*x[1]*(y[2]**2 * z[0] - 2*y[0]*y[1]*z[1] - y[0]*y[2]*z[2]) ) / den
        dz2_ny = (- y[2] * (x[0]*z[0] + x[1]*z[1]) + x[2]*(y[0]*z[0] + y[1]*z[1])) * ( x[1]**2 * (y[0]**2 + y[2]**2)*z[1] + x[2]**2 * y[1]*(y[0]*z[0] + y[1]*z[1]) - x[2]*y[2]*(x[1]*y[0]*z[0] + x[0]*y[1]*z[0] + 2*x[1]*y[1]*z[1]) + x[2]*y[0]*(x[1]*y[0] - x[0]*y[1])*z[2] + x[0]**2 * y[1]*(y[1]*z[1] + y[2]*z[2]) + x[0]*x[1]*(y[2]**2 * z[0] - 2*y[0]*y[1]*z[1] - y[0]*y[2]*z[2]) ) / den
        
        dx0_nz = (x[2]*y[1] - x[1]*y[2]) * (y[0]*z[0] + y[1]*z[1] + y[2]*z[2]) * ( - (x[1]*y[0] - x[0]*y[1])*(z[0]**2 + z[1]**2) + (x[2]*y[1]*z[0] - x[1]*y[2]*z[0] - x[2]*y[0]*z[1] + x[0]*y[2]*z[1])*z[2] ) / den
        dx1_nz = (x[2]*y[0] - x[0]*y[2]) * (y[0]*z[0] + y[1]*z[1] + y[2]*z[2]) * ( (x[1]*y[0] - x[0]*y[1])*(z[0]**2 + z[1]**2) + (-x[2]*y[1]*z[0] + x[1]*y[2]*z[0] + x[2]*y[0]*z[1] - x[0]*y[2]*z[1])*z[2] ) / den
        dx2_nz = - (x[1]*y[0] - x[0]*y[1]) * (y[0]*z[0] + y[1]*z[1] + y[2]*z[2]) * ( (x[1]*y[0] - x[0]*y[1])*(z[0]**2 + z[1]**2) + (-x[2]*y[1]*z[0] + x[1]*y[2]*z[0] + x[2]*y[0]*z[1] - x[0]*y[2]*z[1])*z[2] ) / den
        
        dy0_nz = (- x[2]*y[1] + x[1]*y[2]) * (x[0]*z[0] + x[1]*z[1] + x[2]*z[2]) * ( - (x[1]*y[0] - x[0]*y[1])*(z[0]**2 + z[1]**2) + (x[2]*y[1]*z[0] - x[1]*y[2]*z[0] - x[2]*y[0]*z[1] + x[0]*y[2]*z[1])*z[2] ) / den
        dy1_nz = - (x[2]*y[0] - x[0]*y[2]) * (x[0]*z[0] + x[1]*z[1] + x[2]*z[2]) * ( (x[1]*y[0] - x[0]*y[1])*(z[0]**2 + z[1]**2) + (-x[2]*y[1]*z[0] + x[1]*y[2]*z[0] + x[2]*y[0]*z[1] - x[0]*y[2]*z[1])*z[2] ) / den
        dy2_nz = (x[1]*y[0] - x[0]*y[1]) * (x[0]*z[0] + x[1]*z[1] + x[2]*z[2]) * ( (x[1]*y[0] - x[0]*y[1])*(z[0]**2 + z[1]**2) + (-x[2]*y[1]*z[0] + x[1]*y[2]*z[0] + x[2]*y[0]*z[1] - x[0]*y[2]*z[1])*z[2] ) / den
        
        dz0_nz = - ( (x[1]*y[0] - x[0]*y[1]) * (-x[2]*y[1]*z[0] + x[1]*y[2]*z[0] + x[2]*y[0]*z[1] - x[0]*y[2]*z[1]) + ( x[2]**2 * (y[0]**2 + y[1]**2) - 2*x[2]*(x[0]*y[0] + x[1]*y[1])*y[2] + (x[0]**2 + x[1]**2)*y[2]**2 )*z[2] ) * (x[1]*y[0]*z[1] + x[2]*y[0]*z[2] - x[0]*(y[1]*z[1] + y[2]*z[2])) / den
        dz1_nz = ( (x[1]*y[0] - x[0]*y[1]) * (-x[2]*y[1]*z[0] + x[1]*y[2]*z[0] + x[2]*y[0]*z[1] - x[0]*y[2]*z[1]) + ( x[2]**2 * (y[0]**2 + y[1]**2) - 2*x[2]*(x[0]*y[0] + x[1]*y[1])*y[2] + (x[0]**2 + x[1]**2)*y[2]**2 )*z[2] ) * (- y[1]* (x[0]*z[0] + x[2]*z[2]) + x[1]*(y[0]*z[0] + y[2]*z[2])) / den
        dz2_nz = (-y[2]*(x[0]*z[0] + x[1]*z[1]) + x[2]*(y[0]*z[0] + y[1]*z[1])) * ( (x[1]*y[0] - x[0]*y[1]) * (-x[2]*y[1]*z[0] + x[1]*y[2]*z[0] + x[2]*y[0]*z[1] - x[0]*y[2]*z[1]) + ( x[2]**2 * (y[0]**2 + y[1]**2) - 2*x[2]*(x[0]*y[0] + x[1]*y[1])*y[2] + (x[0]**2 + x[1]**2)*y[2]**2 )*z[2] ) / den
        
        
        err_nx = np.sqrt( dx0_nx**2 * err_x[0]**2 + dx1_nx**2 * err_x[1]**2 + dx2_nx**2 * err_x[2]**2 + 
                         dy0_nx**2 * err_y[0]**2 + dy1_nx**2 * err_y[1]**2 + dy2_nx**2 * err_y[2]**2 + 
                         dz0_nx**2 * err_z[0]**2 + dz1_nx**2 * err_z[1]**2 + dz2_nx**2 * err_z[2]**2)
        
        err_ny = np.sqrt( dx0_ny**2 * err_x[0]**2 + dx1_ny**2 * err_x[1]**2 + dx2_ny**2 * err_x[2]**2 + 
                         dy0_ny**2 * err_y[0]**2 + dy1_ny**2 * err_y[1]**2 + dy2_ny**2 * err_y[2]**2 + 
                         dz0_ny**2 * err_z[0]**2 + dz1_ny**2 * err_z[1]**2 + dz2_ny**2 * err_z[2]**2)
        
        err_nz = np.sqrt( dx0_nz**2 * err_x[0]**2 + dx1_nz**2 * err_x[1]**2 + dx2_nz**2 * err_x[2]**2 + 
                         dy0_nz**2 * err_y[0]**2 + dy1_nz**2 * err_y[1]**2 + dy2_nz**2 * err_y[2]**2 + 
                         dz0_nz**2 * err_z[0]**2 + dz1_nz**2 * err_z[1]**2 + dz2_nz**2 * err_z[2]**2)
        
        err_norm = np.array([err_nx, err_ny, err_nz])
        
        return err_norm
    
    
    
    def err_nv(x,err_x):
        
        '''
        nV = x/|x| donde x = Delta_V
        '''
        
        den = (x[0]**2 + x[1]**2 + x[2]**2)**(3/2)
        
        dx0_nx = (x[1]**2 + x[2]**2) / den
        dx1_nx = - x[0]*x[1] / den
        dx2_nx = - x[0]*x[2] / den
        
        dx0_ny = - x[0]*x[1] / den
        dx1_ny = (x[0]**2 + x[2]**2) / den
        dx2_ny = - x[2]*x[1] / den
        
        dx0_nz = - x[0]*x[2] / den
        dx1_nz = - x[1]*x[2] / den
        dx2_nz = (x[0]**2 + x[1]**2) / den
        
        err_nx = np.sqrt(dx0_nx**2 * err_x[0]**2 + dx1_nx**2 * err_x[1]**2 + dx2_nx**2 * err_x[2]**2 )
        err_ny = np.sqrt(dx0_ny**2 * err_x[0]**2 + dx1_ny**2 * err_x[1]**2 + dx2_ny**2 * err_x[2]**2 )
        err_nz = np.sqrt(dx0_nz**2 * err_x[0]**2 + dx1_nz**2 * err_x[1]**2 + dx2_nz**2 * err_x[2]**2 )
        
        err_norm = np.array([err_nx, err_ny, err_nz])
        
        return err_norm
        

    
    delta_B = Bd - Bu
    err_delta_B = np.sqrt(err_Bd**2 + err_Bu**2)
    delta_V = Vd - Vu
    err_delta_V = np.sqrt(err_Vd**2 + err_Vu**2)
    
    
    err_nB = err_n(Bd, Bu, delta_B, err_Bd, err_Bu, err_delta_B)
    err_nBuV = err_n(Bu, delta_V, delta_B, err_Bu, err_delta_V, err_delta_B)
    err_nBdV = err_n(Bd, delta_V, delta_B, err_Bd, err_delta_V, err_delta_B)
    err_nBduV = err_n(delta_B, delta_V, delta_B, err_delta_B, err_delta_V, err_delta_B)
    err_nV = err_nv(delta_V, err_delta_V)
    
    return err_nB, err_nBuV, err_nBdV, err_nBduV, err_nV

    
    

#para calcular angulos a partir de un producto interno (en grados)
def alpha(x,y):
    
   a = abs(np.arccos((np.dot(x,y))/(np.linalg.norm(x)*np.linalg.norm(y)))*180/(np.pi))
   if a > 90:
       a = abs(180 - a)
       
   return a




def err_alpha(x, y, err_x, err_y):
    
    '''
    alpha es funcion de x[0], x[1], x[2], y[0], y[1], y[2]
    '''
    
    def derxi_alpha(x, y, ai, bi, var = 'X'):
        
        if var == 'X':
            dai = - ( bi/(np.linalg.norm(x)*np.linalg.norm(y)) - ai*(np.dot(x,y))/((x[0]**2 + x[1]**2 + x[2]**2)**(3/2) * np.linalg.norm(y)) ) / np.sqrt(1 - (np.dot(x,y))**2 / (np.linalg.norm(x)**2 * np.linalg.norm(y)**2))
        
        if var == 'Y':
            dai = - ( bi/(np.linalg.norm(x)*np.linalg.norm(y)) - ai*(np.dot(x,y))/((y[0]**2 + y[1]**2 + y[2]**2)**(3/2) * np.linalg.norm(x)) ) / np.sqrt(1 - (np.dot(x,y))**2 / (np.linalg.norm(x)**2 * np.linalg.norm(y)**2))
    
        return dai
    
    
    dx0 = derxi_alpha(x, y, x[0], y[0])
    dx1 = derxi_alpha(x, y, x[1], y[1])
    dx2 = derxi_alpha(x, y, x[2], y[2])
    
    dy0 = derxi_alpha(x, y, y[0], x[0], 'Y')
    dy1 = derxi_alpha(x, y, y[1], x[1], 'Y')
    dy2 = derxi_alpha(x, y, y[2], x[2], 'Y')
    
    err_alpha = np.sqrt( dx0**2 * err_x[0]**2 + dx1**2 * err_x[1]**2 + dx2**2 * err_x[2]**2 + dy0**2 * err_y[0]**2 + dy1**2 * err_y[1]**2 + dy2**2 * err_y[2]**2 )

    return err_alpha




def campos_half(i_u, f_u, i_d, f_d, B1, B2):
    
    '''
    Para calcular los campos upstream y downstream de cada medio subintervalo
    
    En cualquier shock half1 es la mitad más a la izquierda del perfil y half2 la mitad derecha.
    O sea que, en shocks inbound half1 de upstream (por ej) será la mitad más alejada del shock, y en
    shocks outbound será la mitad más cercana al shock.
    '''

    half_u = int(np.abs(f_u - i_u)/2)
    half_d = int(np.abs(f_d - i_d)/2)
    
    half1_Bu = np.mean(B1[:half_u,:], axis = 0)
    half1_Bd = np.mean(B2[:half_d,:], axis = 0)
    std_half1_Bu = np.array([st.stdev(B1[:half_u,0]), st.stdev(B1[:half_u,1]), st.stdev(B1[:half_u,2])])
    std_half1_Bd = np.array([st.stdev(B2[:half_d,0]), st.stdev(B2[:half_d,1]), st.stdev(B2[:half_d,2])])
    
    half2_Bu = np.mean(B1[half_u:,:], axis = 0)
    half2_Bd = np.mean(B2[half_d:,:], axis = 0)
    std_half2_Bu = np.array([st.stdev(B1[half_u:,0]), st.stdev(B1[half_u:,1]), st.stdev(B1[half_u:,2])])
    std_half2_Bd = np.array([st.stdev(B2[half_d:,0]), st.stdev(B2[half_d:,1]), st.stdev(B2[half_d:,2])])
    
    half_Bu = np.array([half1_Bu, half2_Bu])
    half_Bd = np.array([half1_Bd, half2_Bd])
    std_half_Bu = np.array([std_half1_Bu, std_half2_Bu])
    std_half_Bd = np.array([std_half1_Bd, std_half2_Bd])
    
    return half_u, half_d, half_Bu, half_Bd, std_half_Bu, std_half_Bd
    
    


def half_angulo_N(half_n, N, err_half_n, err_N):
    
    '''
    Para calcular los angulos entre cada normal de la combinacion de 
    los medios subintervalos up/down con la normal del fit
    '''
    
    ang_N_half11_n = alpha(half_n[0,:],N)
    ang_N_half12_n = alpha(half_n[1,:],N)
    ang_N_half21_n = alpha(half_n[2,:],N)
    ang_N_half22_n = alpha(half_n[3,:],N)
    
    err_ang_N_half11_n = err_alpha(half_n[0,:],N, err_half_n[0,:], err_N)
    err_ang_N_half12_n = err_alpha(half_n[1,:],N, err_half_n[1,:], err_N)
    err_ang_N_half21_n = err_alpha(half_n[2,:],N, err_half_n[2,:], err_N)
    err_ang_N_half22_n = err_alpha(half_n[3,:],N, err_half_n[3,:], err_N)
    
    ang_N_n = np.array([ang_N_half11_n, ang_N_half12_n, ang_N_half21_n, ang_N_half22_n])
    err_ang_N_n = np.array([err_ang_N_half11_n, err_ang_N_half12_n, err_ang_N_half21_n, err_ang_N_half22_n])
    
    return ang_N_n, err_ang_N_n




def half_best_n(ang_N_n, half_n, zeda = 5):
    
    '''
    Para un dado tipo de normal coplanar, calculo la diferencia 
    de los angulos con normal del fit de todas las combinaciones de subintervalos
    entre si. Si la resta entre alguno de ellos supera los zeda grados,
    considero que los calculos con los intervalos completos son inestables
    y elijo, para ese tipo de normal, la mejor combinacion de medios subintervalos.
    
    Si los calculos son inestables, la funcion me devuelve el menor angulo con la
    normal del fit, de entre los 4 calculados para ese tipo de normal, la normal que
    da ese angulo minimo y el indice:
        indice 0 = primera mitad upstream - primera mitad downstream
        indice 1 = primera mitad upstream - segunda mitad downstream
        indice 2 = segunda mitad upstream - primera mitad downstream
        indice 3 = segunda mitad upstream - segunda mitad downstream
    '''
    
    resta = np.empty([4,4])
    for i in range(4):
        for j in range(4):
            resta[i,j] = np.abs(np.abs(ang_N_n[i]) - np.abs(ang_N_n[j]))
    
    test = np.count_nonzero(resta>=zeda) #devuelve cant de elementos que cumplen condicion
    if test >= 1:
        print('los calculos son inestables')
        ind = (list(ang_N_n)).index(min(ang_N_n)) 
        ang_N_n_best = min(ang_N_n)
        half_n_best = half_n[ind,:]
        
        return half_n_best, ang_N_n_best, ind
        
    else:
        raise ValueError('los calculos son estables')



#metodo "bootstrap" para calcular Bu y Bd promedio y n. N samples del mismo intervalo upstream y downstream
def copl_boot(B1,B2,V1,V2,Ns):
    
    Bu = np.empty([Ns,3])
    Bd = np.empty([Ns,3])
    Vu = np.empty([Ns,3])
    Vd = np.empty([Ns,3])
    
    nB = np.empty([Ns,3])
    nBuV = np.empty([Ns,3])
    nBdV = np.empty([Ns,3])
    nBduV = np.empty([Ns,3])
    nV = np.empty([Ns,3])
    
    #necesito pasarlo a listas auxiliarmente para poder hacer random choice respetando la relacion de xyz
    B_u = list(B1) 
    B_d = list(B2)
    V_u = list(V1) 
    V_d = list(V2)
    
    for i in range(Ns):
        bu = [random.choice(B_u) for _ in range(len(B_u))] #genero lista random de vectores de B upstream (seleccion con reemplazo-> puedo seleccionar mas de una vez el mismo vector y entonces no agarrar todos los valores de vectores disponibles)
        bu = np.array(bu)
        Bu[i,:] = np.mean(bu, axis = 0) #cada Bd[i,:] es un promedio de la seleccion random i-esima de vectores de campo mag upstream
        
        bd = [random.choice(B_d) for _ in range(len(B_d))]
        bd = np.array(bd)
        Bd[i,:] = np.mean(bd, axis = 0)
        
        vu = [random.choice(V_u) for _ in range(len(V_u))] 
        vu = np.array(vu)
        Vu[i,:] = np.mean(vu, axis = 0) 
        
        vd = [random.choice(V_d) for _ in range(len(V_d))]
        vd = np.array(vd)
        Vd[i,:] = np.mean(vd, axis = 0) 

        nB[i,:], nBuV[i,:], nBdV[i,:], nBduV[i,:], nV[i,:] = norm_coplanar(Bd[i,:],Bu[i,:],Vd[i,:],Vu[i,:])
    
    return Bu, Bd, Vu, Vd, nB, nBuV, nBdV, nBduV, nV #genero un Bu, Bd, Vu, Vd medios para cada iteracion, y con ellos un n de cada tipo




#para armar stack de matrices B1=[Bx,By,Bz] en intervalo up/downstream variando el intervalo
def intervalo(Bx,By,Bz,t,t1,t2,q):
    
    '''
    t1 y t2 son los limites a izquierda y derecha, respectivamente, donde es posible
    encontrar el intervalo, y por ende, donde voy a hacer las variaciones.
    
    s es la cantidad de selecciones del intervalo que puedo hacer entre los limites
    extremos t1,t2 mantiendo el ancho q minutos (en hora dec) y desplazando una selccion de intervalo de
    la seleccion siguiente en medio minuto (1/120 es 1/2 min en hora decimal).
    '''
    
    #busco los tiempos que hacen de limites extremos entre los valores que tiene mi vector t
    i_t1 = (np.abs(t - t1)).argmin()
    i_t2 = (np.abs(t - t2)).argmin()
    t1 = t[i_t1]
    t2 = t[i_t2]
    
    s = int((t2 - (t1 + q))*120 + 1)   #cant de samples de ancho temporal q
    
    if s > 1: #si se puede hacer por lo menos 2 sample de ancho q
        
        #cant de elementos de t en un ancho temporal q
        i_tq = (np.abs(t - (t1 + q))).argmin()
        k = abs(i_tq - i_t1)    
       
        b1 = np.empty([k, 3, s])
        for i in range(s):
            j = (np.abs(t-(t1 + i/120))).argmin()   #indice de t con valor mas cercano a t1, t1 + medio min, etc
            b1[:,:,i] = np.array([Bx[j:j + k], By[j:j + k], Bz[j:j + k]]).T 
        return b1
    
    else:
        print('No se puede variar intervalo')

