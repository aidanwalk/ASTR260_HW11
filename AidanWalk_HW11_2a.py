#Aidan Walk
#ASTR 260-001
#21 April 2021, 17.00
#HW11 PDE's

#Problem 2, part a

import numpy as np
import matplotlib.pyplot as plt
import numba

def dPhi_dt(x, t):
    '''
    '''
    
    return psi(x,t=t)
    
def dPsi_dt(x=None, t=None, v=None, a=None, initialVal=None):
    '''Pass in values of phi for previous iteration
    '''
    dPsiDt = v**2/a**2 * ( phi(x+a,t, initialVal=initialVal) + 
                           phi(x-a,t, initialVal=initialVal) - 
                           2*phi(x,t, initialVal=initialVal) )
    
    
    return dPsiDt, phi(x,t, initialVal=initialVal)
    
def psi(x, t=None):
    '''
    '''
    return C * (x*(L-x)/L**2) * np.exp( -1*(x-d)**2/(2*sigma**2) )
    
def phi(x, t=None, initialVal=None):
    '''phi(x)=0 everywhere, but velocity at psi(x) !=0
       h is previously defined timestep
       '''
    if t==0: return 0 #initial condition
    else: #Eulers step
        return initialVal + timeStep*dPhi_dt(x, t)
    
if __name__ == '__main__':
    cm = 100 #cm/m
    v = 100*cm #m/s
    L = 1*cm #length of string
    d = 10 #cm
    C = 1*cm #cm/s
    sigma = 0.3*cm #m
    
    time = 0
    timeMax = 1e-6#second
    timeStep = 1e-6
    spatialStep = 1 #cm , spatial step
    
    
    #divide into set of spatial points
    points = np.arange(0, L, spatialStep)
    
    
    
    phiVal = phi(0, t=0) #find initial value of phi
    psiVal = psi(0)
    while time < timeMax:
        psiDerivList = []
        phiList = []
        psiValList = []
        
        #for each point, calculate 2nd derivative using neighboring points
        for point in points:
            #find psi
            psiVal = psi(point, t=time)
            
            
            #calculate dPsi_dt
            psiDeriv, phiVal = dPsi_dt(x=point, t=time, v=v,
                                       a=spatialStep, initialVal=phiVal)
            
            
            #forward propogate using Euler's Method
            psiVal = psiVal + timeStep*psiDeriv
            
            
            
            #append data to lists
            psiDerivList.append(psiDeriv)
            phiList.append(phiVal)
            psiValList.append(psiVal)
        
        
        #print(psiValList)
        
        plt.plot(points, psiValList)
        plt.show()
        
        
        
        time += timeStep #increment time step
    
    
    
    
    
    
    
    
    