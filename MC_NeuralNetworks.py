import torch
from torch import nn
import numpy as np


# List of function and class 

# recordRepGrainState(MC_areas, MC_sides, stateFileName)
# NeuralNetInput(area_self, side_self, neighbors_info,dt): prepares the input data structure for neural networks
# class CombinedNeuralNetworks(nn.Module): 

def recordRepGrainState(MC_areas, MC_sides, stateFileName):
    print("recording rep grain states....")
    writeFile=open(stateFileName,'w')

    dataSize = MC_areas.shape[0]
    for i in range (0, dataSize):
        stringWriting = "%.9f" %MC_areas[i] + " " + "%d" %MC_sides[i] + "\n"
        writeFile.write(stringWriting)
    
    writeFile.close()

    return

def NeuralNetInput(area_self, side_self, neighbors_info,dt):
    
    S=int(side_self)
    A0=area_self
    
    # input
    # neighborsInfo[0,m] = m # local_index (0 to S) that conveys connectivity 
    # neighborsInfo[1,m] = MC_areas[nIndex[m]] # area
    # neighborsInfo[2,m]= ((np.pi/3.0)* MC_sides[nIndex[m]] - 2*np.pi) * dt # areaCh
    # neighborsInfo[3,m] = MC_sides[nIndex[m]] # global index
    
    # output
    Xinput = np.zeros( int( (S*S+S)*0.5+ (S+1)+ 1)  ) 
    
    #[check] sort works before and after 
    neighborsInfo = neighbors_info[:, neighbors_info[1, :].argsort()]    
    
    #print("neighborsInfo_Sorted in the function \n", neighborsInfo)
    
    #Construct adjacency graph to save the link between neigbhors
    graph = np.zeros((S,S))
 
    for m in range (0,S):
        graph[m,m]= neighborsInfo[1,m]
        local_index_m = neighborsInfo[0,m]
            
        for n in range (0,S):
            
            # use local 'm' and 'n' index to judge wheter they are neighbors 
            # if they are neighbors, then turn the boolean operator
            local_index_n= neighborsInfo[0,n]
            n_is_m_neighbor = False
                
            # local index defines the local links 
            # if the difference is 1, then they are connected 
            if(abs(local_index_n - local_index_m) -1.0 ==0
                or ( abs(local_index_n-local_index_m ) %(S-1)==0 and local_index_n != local_index_m) ):
                
                n_is_m_neighbor = True
                    
                if(n_is_m_neighbor==True):
                    graph[m,n]= (neighborsInfo[1,m] + neighborsInfo[1,n])* 0.5
    #End of adjacency graph construction 

    
    # Now, collect them and fill 'Xinput' in order
    # Do normalization as you go
    Xinput[0] = A0/A0 # self area 
    Xinput[S+1] = ((np.pi/3.0)* S - 2*np.pi) * dt /A0
    
    #neighbor areas
    for m in range (0,S):
        #index_of_m = int(neighborsInfo[3,m])
        Xinput[m+1] = neighborsInfo[1,m]/A0
        Xinput[m+S+2] = neighborsInfo[2,m]/neighborsInfo[1,m]
        
    # read from graph[m,m]
    innerIterator=0
    for m in range (0,S-1):
        for j in range (m,S-1):
            Xinput[2*S+2+innerIterator]=graph[m,j+1]/A0
            innerIterator+=1
            
    return Xinput

# This class executes suitable neural network, depending on the side of a rep-grain 
# and predict the probability of topological states at next time step

class CombinedNeuralNetworks(nn.Module):
    def __init__(self, device): 
        super(CombinedNeuralNetworks, self).__init__()

        self.maxSide = 9
        
        self.model_Tri = NeuralNetwork_Tri(3).to(device)
        self.model_Quad = NeuralNetwork_Quad(4).to(device)
        self.model_Penta = NeuralNetwork_Penta(5).to(device)    
        self.model_Hexa = NeuralNetwork_Hexa(6).to(device)   
        self.model_Hepta = NeuralNetwork_Hepta(7).to(device)  
        self.model_Octa = NeuralNetwork_Octa(8).to(device) 
        self.model_Nona = NeuralNetwork_Nona(9).to(device) 

        self.model_Tri.load_state_dict(torch.load('NetworkParameter/Tri.pth',map_location=device))
        self.model_Quad.load_state_dict(torch.load('NetworkParameter/Quad.pth',map_location=device))
        self.model_Penta.load_state_dict(torch.load('NetworkParameter/Pentagon.pth',map_location=device))
        self.model_Hexa.load_state_dict(torch.load('NetworkParameter/Hexagon.pth',map_location=device))
        self.model_Hepta.load_state_dict(torch.load('NetworkParameter/Heptagon.pth',map_location=device))
        self.model_Octa.load_state_dict(torch.load('NetworkParameter/Octagon.pth',map_location=device))
        self.model_Nona.load_state_dict(torch.load('NetworkParameter/Nonagon.pth',map_location=device))

        self.model_Quad.eval()
        self.model_Penta.eval()
        self.model_Hexa.eval()
        self.model_Hepta.eval()
        self.model_Octa.eval()
        self.model_Nona.eval()

    def model_predict(self, inputTensor,S):

        if(S==3):
            modelOut = torch.nn.functional.softmax(self.model_Tri(inputTensor), dim=0)
        elif(S==4):
            modelOut = torch.nn.functional.softmax(self.model_Quad(inputTensor), dim=0)
        elif(S==5):
            modelOut = torch.nn.functional.softmax(self.model_Penta(inputTensor), dim=0)
        elif(S==6):
            modelOut = torch.nn.functional.softmax(self.model_Hexa(inputTensor), dim=0)
        elif(S==7):
            modelOut = torch.nn.functional.softmax(self.model_Hepta(inputTensor), dim=0)
        elif(S==8):
            modelOut = torch.nn.functional.softmax(self.model_Octa(inputTensor), dim=0)
        elif(S==9):
            modelOut = torch.nn.functional.softmax(self.model_Nona(inputTensor), dim=0)
        else:
            print("Unusual things happened")
            print("S:", S)
            exit()
        preds = torch.argmax(modelOut)
        return preds
        
    def handleTopology(self, MC_sides,MC_areas,MC_peusedoNeighbors, i,dt):

        minAreatol = 0.00047824 # area tol
        
        #output
        MC_newSide=0
        MC_newPeusedoNeighbors=np.zeros((1,self.maxSide))
        MC_newPeusedoNeighbors[0,:] = MC_peusedoNeighbors[i,:]
        MC_newArea=MC_areas[i]
    
        # auxiliary varibales.
        Nsample = MC_sides.shape[0]
    
        S = MC_sides[i]
        nIndex = np.int_(MC_peusedoNeighbors[i,0:S])
        neighborsInfo=np.zeros( (5, S) )

        # 'nIndex' collects the index of neighbors
        # and 'neighborsInfo' saves the states of neighbors

        for m in range (0,S):
            neighborsInfo[0,m] = m # local_index conveys connectivity
            neighborsInfo[1,m] = MC_areas[nIndex[m]] # area
            neighborsInfo[2,m]= ((np.pi/3.0)* MC_sides[nIndex[m]] - 2*np.pi) * dt # areaCh
            neighborsInfo[3,m] = MC_sides[nIndex[m]] # number of side
            neighborsInfo[4,m] = nIndex[m] # global index
            
        # The NeuralNetInput function prepares input of the neural network function
        # from the above information
        Xinput = NeuralNetInput(MC_areas[i], MC_sides[i], neighborsInfo,dt)
        inputTensor = torch.FloatTensor(Xinput)
    
        preds=0 #default

        if(S>2 and S<10):
            #preds = ModelForTopology.model_predict(inputTensor,S)
            preds = self.model_predict(inputTensor,S)
        else:
            print("You have a grain that has side S< 2 or S>9. Better to terminate the program")
            exit()

        if(preds==0):
            criticalEvent=0
            MC_newSide = S
        else:
            criticalEvent=1
        
        if(preds==1): # a grain increases a side. so we simply add new neighbors
            MC_newSide = S+1
    
            correctSide=False
            while(correctSide == False) :
                random_indx = np.random.choice(Nsample, 1, replace=False)
                neighbor_side = MC_sides[int(random_indx)]

                random_indx_arb = np.random.choice(Nsample, 1, replace=False)
                # From the randomly picked index find a neighbor that has the same number of side
                for k in range (0,Nsample):
                    test_indx1 = random_indx_arb + k
                    
                    if(test_indx1 < Nsample):
                        test_indx = test_indx1
                    else:
                        test_indx = test_indx1-Nsample
        
                    if(MC_sides[test_indx] == neighbor_side):
                        correctSide=True
                        break
                
            MC_newPeusedoNeighbors[0,S]= random_indx_arb
        
        if(preds==2):

            MC_newSide = S-1
              
            kk=0
            minC_alpha = neighborsInfo[2,kk]/neighborsInfo[1,kk]
            minGlobalIndex = neighborsInfo[3,kk] # first guess on that has min(C_alpha)
            minLocalIndex = neighborsInfo[0,kk]
            
            ## search min c_\alpha
            for kk in range (1,S):
                temp = neighborsInfo[2,kk]/neighborsInfo[1,kk]
                    
                if(temp < minC_alpha):
                    minC_alpha = temp
                    minGlobalIndex = neighborsInfo[3,kk]
                    minLocalIndex = neighborsInfo[0,kk]
        
            for kkk in range (int(minLocalIndex), S-1):
                MC_newPeusedoNeighbors[0,kkk]=MC_newPeusedoNeighbors[0,kkk+1]
                
            # Make sure to delete the last local index, as the rep-grain loses the neighbor
            MC_newPeusedoNeighbors[0,S-1]=0
        
        
        if(preds==3):
            # clean the current info..
            for m in range (0,S):
                MC_newPeusedoNeighbors[0,m]=0
            # make random choice on rep-grain
            aliveSample=False
            while(aliveSample==False):
                test_indx = np.random.choice(Nsample, 1, replace=False)

                if(MC_areas[test_indx]> minAreatol and MC_sides[test_indx]>2):
                    aliveSample=True
                    
            MC_newSide = MC_sides[test_indx]
            MC_newArea = MC_areas[test_indx]
          
            # We will add new peusedo neighbors to it!!
            maxIter =Nsample
        
            for j in range (0,int(MC_newSide)):
                reasonable=False
                while reasonable == False:
    
                    random_indx = np.random.choice(Nsample, 1)
                    neighbor_side = MC_sides[int(random_indx)]
                    random_indx_arb = np.random.choice(Nsample, 1)

                    for k in range (0,Nsample):
                        test_indx1 = random_indx_arb + k
                    
                        if(test_indx1 < Nsample):
                            test_indx = test_indx1
                        else:
                            test_indx = test_indx1-Nsample
            
                        if(MC_sides[test_indx] == neighbor_side and MC_areas[test_indx]>minAreatol ):
                            reasonable=True
                        break
                        
                MC_newPeusedoNeighbors[0,j]=test_indx
        return MC_newSide, MC_newArea, MC_newPeusedoNeighbors, criticalEvent


class NeuralNetwork_Tri(nn.Module):
    def __init__(self, currentSide): # if this is available....
        super(NeuralNetwork_Tri, self).__init__()

        self.linear_stack = nn.Sequential(
            nn.Linear(int( (currentSide*currentSide+currentSide)/2+1+currentSide+1), 10), 
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 4),
        ) 
    def forward(self, x):
        return self.linear_stack(x)


class NeuralNetwork_Quad(nn.Module):
    def __init__(self, currentSide): # if this is available....
        super(NeuralNetwork_Quad, self).__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(int( (currentSide*currentSide+currentSide)/2+1+currentSide+1), 32), 
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
        )  
    def forward(self, x):
        return self.linear_stack(x)


class NeuralNetwork_Penta(nn.Module):
    def __init__(self, currentSide): 
        super(NeuralNetwork_Penta, self).__init__()

        self.linear_stack = nn.Sequential(
            nn.Linear(int( (currentSide*currentSide+currentSide)/2+1+currentSide+1), 65), 
            nn.ReLU(),
            nn.Linear(65, 65),
            nn.ReLU(),
            nn.Linear(65, 65),
            nn.ReLU(),
            nn.Linear(65, 4),
        )    

    def forward(self, x):
        return self.linear_stack(x)

class NeuralNetwork_Hexa(nn.Module):
    def __init__(self, currentSide): 
        super(NeuralNetwork_Hexa, self).__init__()

        self.linear_stack = nn.Sequential(
            nn.Linear(int( (currentSide*currentSide+currentSide)/2+1+currentSide+1), 70), 
            nn.ReLU(),
            nn.Linear(70, 70),
            nn.ReLU(),
            nn.Linear(70, 70),
            nn.ReLU(),
            nn.Linear(70, 70),
            nn.ReLU(),
            nn.Linear(70, 70),
            nn.ReLU(),
            nn.Linear(70, 3)
        )  
    def forward(self, x):
        return self.linear_stack(x)

class NeuralNetwork_Hepta(nn.Module):
    def __init__(self, currentSide):
        super(NeuralNetwork_Hepta, self).__init__()

        self.linear_stack = nn.Sequential(
            nn.Linear(int( (currentSide*currentSide+currentSide)/2+1+currentSide+1), 70), 
            nn.ReLU(),
            nn.Linear(70, 70),
            nn.ReLU(),
            nn.Linear(70, 70),
            nn.ReLU(),
            nn.Linear(70, 3)
        )
    def forward(self, x):
        return self.linear_stack(x)

class NeuralNetwork_Octa(nn.Module):
    def __init__(self, currentSide):
        super(NeuralNetwork_Octa, self).__init__()

        self.linear_stack = nn.Sequential(
            nn.Linear(int( (currentSide*currentSide+currentSide)/2+1+currentSide+1), 70),  
            nn.ReLU(),
            nn.Linear(70, 30),
            nn.ReLU(),
            nn.Linear(30, 30),
            nn.ReLU(),
            nn.Linear(30, 3)
        ) 
        
    def forward(self, x):
        return self.linear_stack(x)

class NeuralNetwork_Nona(nn.Module):
    def __init__(self, currentSide): 
        super(NeuralNetwork_Nona, self).__init__()

        self.linear_stack = nn.Sequential(
            nn.Linear(int( (currentSide*currentSide+currentSide)/2+1+currentSide+1), 20), 
            nn.ReLU(),
            nn.Linear(20, 7),
            nn.ReLU(),
            nn.Linear(7, 7),
            nn.ReLU(),
            nn.Linear(7, 7),
            nn.ReLU(),
            nn.Linear(7, 3) 
        )
         
    def forward(self, x):
        return self.linear_stack(x)

