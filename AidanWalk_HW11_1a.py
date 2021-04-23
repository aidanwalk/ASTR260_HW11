#Aidan Walk
#ASTR 260-001
#21 April 2021, 17.00
#HW11 PDE's

#Problem 1, part a

import numpy as np
import matplotlib.pyplot as plt
import numba
import time

class box:
    '''potential box to be inserted into metal square'''
    def __init__(self, length, width, positionX=None, positionY=None, charge=None):
        self.length = length #in cm, associated with x
        self.width = width #in cm, associated with y
        self.positionX = positionX #X position of top left corner of box
        self.positionY = positionY #y position of top left corner of box
        self.charge = charge #+/- 1
        
def insertBox(array, rho=None, boxes=None):
    '''inserts the potential box into an array
       positionX and positionX+width,
       positionY and positionY+length (of box) must be within bounds of array
       boxes = list of boxes to insert into array
    '''
    for box in boxes:
        xnaught, xfinal = box.positionX, box.positionX + box.width
        ynaught, yfinal = box.positionY, box.positionY + box.length
    
        #input charged box into array
        for i in range(xnaught, xfinal):
            for j in range(ynaught, yfinal):
                array[j,i] = rho*box.charge #index goes as j,i b/c row=y, column=x
            
    return array


@numba.jit(nopython = True)
def iterate(phi):
    
    phiPrime = np.zeros(phi.shape)
    for i in range(gridsize+1):
        for j in range(gridsize+1):
            #if i or j is boundary, keep values the same
            if i==0 or i==gridsize or j==0 or j==gridsize or np.abs(phi[i,j])==rho:
                phiPrime[i,j] = phi[i,j]
            else:
                phiPrime[i,j] = (1.0/4)*(phi[i+1, j] + phi[i-1, j] + \
                                         phi[i, j+1] + phi[i, j-1]  ) + gridSpacing**2/4.0*rho
    
    return phiPrime

if __name__ == "__main__":
    gridsize = 100 #n x n grid across box, in cm
    target = 1e-6 #target accuracy, in volts
    
    #charge density
    gridSpacing = 0.01 #cm in a m
    rho = 1 #coulomb/m**2
    
    
    #box characteristics 
    width = length = 20 #cm
    positiveBox = box(length, width, 
                      positionX = gridsize-length-20, 
                      positionY = 20, 
                      charge = 1)
    negativeBox = box(length, width, 
                      positionX = 20, 
                      positionY = gridsize-width-20, 
                      charge = -1)                 
    boxes = [positiveBox, negativeBox]
    
    
    #initial metal box
    metalBox = np.zeros((gridsize+1, gridsize+1))
    #insert potential boxes into metal box
    metalBox = insertBox(metalBox, rho=rho, boxes=boxes)


    phiprime = np.zeros(metalBox.shape)

    max_diff = 1.0
    iteration = 0
    
    print('Calculating potential via Jacobi Method to an accuracy of', target, 'volts...')
    start_time = time.time()
    while max_diff > target:
        #calculate new values of potential
        phiprime = iterate(metalBox)
        
        max_diff = np.max(abs(metalBox-phiprime))
        metalBox, phiprime = phiprime, metalBox
        #print('iteration:', iteration, 'Difference', max_diff)
        iteration=iteration+1
        
    end_time = time.time() #stop time
    print('--- Calculation duration: %s seconds ---' % (end_time - start_time))
    print('Value of potential at middle of box:', metalBox[int(gridsize/2), int(gridsize/2)])
    
    plt.imshow(metalBox, cmap='Greys')
    plt.savefig('AidanWalk_HW11_1a_Plot.png', dpi=300)
    print("Plot saved as 'AidanWalk_HW11_1a_Plot.png'")
