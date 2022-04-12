import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn

from MC_NeuralNetworks import *

random.seed(8888)

# Global variables
DEVICE = 'cpu'
MAXSIDE= 9
MIN_AREA_TOL = 0.00047824 # area tol

# Parameters from training data
epsilon = 0.006 # grain boundary length scale parameter used in phase field simulation 
dt= (epsilon**2)*0.25 # time step used for phase field simulation (training dataset) 

# Monte Carlo Simulation parameter
Nsample = 5000 #5000
totalTimeStep=6000 #6000

# [User input] : initial grain statistics
 
# Initialize states of rep-grains (MC_areas, MC_sides) using given initial statistics 

MC_areas = np.ones(Nsample) * (1.0/Nsample) 
MC_areas = MC_areas * (Nsample/200) 
p_initial_TP =[0.0,0.0, 1./7, 1./7, 1./7, 1./7, 1./7, 1./7,1./7,0.0]
MC_sides= np.random.choice([1,2,3,4,5,6,7,8,9,10], Nsample,replace=True,p=p_initial_TP)

# 'MC_peusedoNeighbors' is adjacency list that stores peusedo neighbors information of each rep-grain 
MC_peusedoNeighbors=np.zeros((Nsample,MAXSIDE))

# Construct initial pseudo neighbors for each rep grain
for i in range (0,Nsample):
    for j in range (0,MC_sides[i]):
        
        correctSide=False
        while (correctSide==False):
            
            random_indx = np.random.choice(Nsample, 1, replace=False)
            neighbor_side = MC_sides[int(random_indx)]
            random_indx_arb = np.random.choice(Nsample, 1, replace=False)
            
            # Here, the code may seem redundant. 
            # Yet, having the following structure will reserve a room for selecting neighbor_side in more general way
            # (not just uniformly randomly choosing some index )

            for k in range (0,Nsample):
                test_indx1 = random_indx_arb + k 
                    
                if(test_indx1 < Nsample):
                    test_indx = test_indx1
                else: 
                    test_indx = test_indx1-Nsample
            
                if(MC_sides[test_indx] == neighbor_side):
                    correctSide=True
                break
                       
        MC_peusedoNeighbors[i,j]=test_indx
                
print("Rep grains are initialized.")


dTime =0 




# For monitoring the MC simulation, we will record the mean value of side and areas
writeFile_mean=open("MeanHistory.txt",'w')

# We construct Neural Network model to handle grain topology.... 
ModelForTopology = CombinedNeuralNetworks(DEVICE)


print("MC simulation begins.....")

for t in range (0,totalTimeStep+1):
    dTime += dt
    numOfCriticalEvent=0 # Auxiliary variable. Just to monitor how frequent TP changeㄴ occur at each time step 

    # [Step 1] Do topology transformation to each rep-grain
    for i in range (0, Nsample): 
         
        MC_newSide, MC_newArea, MC_newPeusedoNeighbors, criticalEvent =ModelForTopology.handleTopology(MC_sides,MC_areas,MC_peusedoNeighbors,i,dt)
        
        MC_sides[i]= MC_newSide
        MC_areas[i]= MC_newArea
        MC_peusedoNeighbors[i,:]= MC_newPeusedoNeighbors[0,:]
        
        numOfCriticalEvent += criticalEvent
        
    # [Step 2] Apply the von Neumann's law
    for i in range (0, Nsample): #(0,Nsample):
        
        S = MC_sides[i]
        MC_areas[i] = MC_areas[i] + ((np.pi/3.0)* S - 2*np.pi) * dt

        # If either a grain disappears or become more than 10 sided
        # we need to create a grain from the current statistics
        
        if(MC_areas[i]<= MIN_AREA_TOL or MC_sides[i]>=10):

            numOfCriticalEvent+=1
            # (a) Clean out the current info
            MC_sides[i]=0
            MC_areas[i]=0
            
            for m in range (0,S):
                MC_peusedoNeighbors[i,m]=0
            
            aliveSample=False
            while(aliveSample==False):
                test_indx = np.random.choice(Nsample, 1, replace=False)
                if(MC_areas[test_indx]> MIN_AREA_TOL and MC_sides[test_indx]>3):
                    aliveSample=True
                    
            MC_sides[i]= MC_sides[test_indx]
            MC_areas[i]= MC_areas[test_indx]
          
            # a new rep-grain is created then, add new peusedo neighbors to it!!
            maxIter =Nsample
            for j in range (0,MC_sides[i]):
                reasonable=False        
                while reasonable == False:
    
                    random_indx = np.random.choice(Nsample, 1, replace=False)
                    neighbor_side = MC_sides[int(random_indx)]

                    random_indx_arb = np.random.choice(Nsample, 1, replace=False)

                    for k in range (0,Nsample):
                        test_indx1 = random_indx_arb + k
                    
                        if(test_indx1 < Nsample):
                            test_indx = test_indx1
                        else:
                            test_indx = test_indx1-Nsample
            
                        if(MC_sides[test_indx] == neighbor_side and MC_areas[test_indx]>MIN_AREA_TOL ):
                            reasonable=True
                        break
                        
                MC_peusedoNeighbors[i,j]=test_indx

    meanSide =np.mean(MC_sides)         
    meanArea =np.mean(MC_areas)
   
    stringWriting = "%.9f" %meanSide + " " + "%.9f" %meanArea + "\n"
    writeFile_mean.write(stringWriting)
    
    # Followings are for recording the evolving grain statistics 
    
    # Monitoring
    if(t%10 ==0):
        print("At iteration t=",t,"/",totalTimeStep, "time=",dTime, "  Mean Area=", meanArea, " Mean Side", meanSide)
        print("Number of critical events ", numOfCriticalEvent,  "in percent",  numOfCriticalEvent/Nsample  * 100 ," %")

    # Recording
    if(t%100==0):
        timeString = str(t)
        filenameString = "State_at_" + timeString + ".txt"
        recordRepGrainState(MC_areas,MC_sides,filenameString)
    
writeFile_mean.close()
recordRepGrainState(MC_areas,MC_sides,"MC_final_state.txt")

print("Successfully finishing program...")


            
