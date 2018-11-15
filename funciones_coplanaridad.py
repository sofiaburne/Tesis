import numpy as np
import random
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



#para calcular angulos a partir de un producto interno (en grados)
def alpha(x,y):
   a = np.arccos((np.dot(x,y))/(np.linalg.norm(x)*np.linalg.norm(y)))*180/(np.pi)
   return a




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
def intervalo(Bx,By,Bz,t1,t2,t,i_u,f_u): #le doy limites temporales entre los que podria estar comprendido el intervalo
    q = f_u - i_u #cantidad de elementos en el intervalo
    s = int((t2 - (t1 + t[f_u] - t[i_u]))/0.0083) #cant de samples manteniendo el ancho original y haciendo desplazamientos de 1/2 min entre t1 y t2 (0.0083 es 1/2 min en hora decimal)
    b1 = np.empty([q, 3, s])
    for i in range(s):
        j = (np.abs(t-(t1 + i*0.0083))).argmin() #busca el indice del elemento con valor mas cercano a t1
        b1[:,:,i] = np.array([Bx[j:j + q], By[j:j + q], Bz[j:j + q]]).T 
    return b1



