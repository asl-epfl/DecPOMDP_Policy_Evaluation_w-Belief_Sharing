"""
Implementation code for the paper: Policy Evaluation in Decentralized POMDPs with Belief Sharing

Documentation: https://github.com/asl-epfl/Policy-Evaluation-in-Decentralized-POMDPs-with-Belief-Sharing./

Fatima Ghadieh - 2023
"""
from math import nan
import random
import math
import csv  
import matplotlib.pyplot as plt
import numpy as np 
import cupy as cu
from Agentclass import Agent
import os
from random import Random
import networkx as nx
import time
import matplotlib.pyplot as plt
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
import matplotlib.image as image

#Use gpu device number 3 
dev1 = cu.cuda.Device(3)
dev1.use()

def random_exclude(range_start, range_end, *excludes):
    
    r = randomlocation.randint(range_start, range_end)
    while r in excludes:
        r = randomlocation.randint(range_start, range_end)
        print(r in excludes)
    return r



class GridWorld:
    ## Initialise starting data
    def __init__(self, num = 5, height = 5, width = 5, noisy = 0, centralized = 0,sparse = False, alpha = 0.1, rho = 0.01, phi = 10, discount_factor = 0.95, beliefvectors = []):
        self.noisy = noisy
        self.alpha = alpha
        self.rho = rho
        self.phi = phi
        self.cent_r = 0
        self.beliefvectors = beliefvectors
        
        # Set information about the gridworld
        self.gamma = num #between 1 and num of agents 
        self.height = height
        self.jointaction = [0,0]
        self.width = width
        self.num =  num
        self.discount_factor = discount_factor
        """
        self.centralized refers to the types of Policy Evaluation and Execution:
        0 -> Centralized Policy Evaluation, Centralized Execution
        1 -> Decentralized Policy Evaluation, Decentralized Execution
        2 -> Centralized Policy Evaluation, Decentralized Execution
        """  
        self.centralized = centralized

        self.val = 0
         
        #states
        self.states  = [0]*(self.height*self.width) #Since this environment is a grid environment, the number of states found here is equal to the height of the grid * the width of the grid
        self.sbehistory = []

        #defining the states
        x = 0
        for i in range(self.width):
            for j in range(self.height):
                self.states[x] = [i,j] #Each state is defined by its (x,y) coordinates along the grid
                x+=1
        
        self.jointobservation = [1/(self.height*self.width)]*(self.height*self.width)
        
        #transition matrix 
        """
        Transition matrix is a matrix that allows us to determine the transition probabilities from every state S
        to every other state S2 found in the grid.
        """
        self.transition_matrix = [1/(self.height*self.width)]*(self.height*self.width)
 
        self.omegalistavg = []
        self.ErrorList = []
        self.Errorhistory = []
        
        #Initializing positions randomly
        self.ListofAgents = []
        self.xpositions = [self.width]  # List of x coordinates of all agents
        self.ypositions = [self.height] # List of y coordinates of all agents
        global randomlocation
        randomlocation =  random.Random(17)
        
        first = True
        y = 0
        centerofgravity_x = randomlocation.randrange(0, self.height-1)
        centerofgravity_y = randomlocation.randrange(0, self.width-1)
        omega_i = []

        for i in range(num):
            omega = np.random.randint(1, high = 20, size = self.height*self.width)
            omega_i.append(omega)

        self.omega_dd_init = omega_i
        self.omega_init = np.average(omega_i, axis=0) 
        for i in range(self.num):
            if (first == True): #initializing position of first agent
                posx = randomlocation.randrange(start = 0, stop = self.height-1)
                posy = randomlocation.randrange(start = 0, stop = self.width-1)
                
                while (abs(posx - centerofgravity_x ) + abs(posy - centerofgravity_y) > 3): #while it is far, keep making it small so that the graph is connected 
                            posx = random_exclude(0, self.height-1, self.xpositions) #Each agent has a unique random position on the grid
                            posy = random_exclude(0, self.width-1, self.ypositions)
              
                x = Agent(y,posx,posy,self.height,self.width,self.noisy, self.alpha, self.omega_init, self.rho, self.phi, self.num, self.discount_factor)                
                

                self.xpositions.append(x.posx)
                self.ypositions.append(x.posy)
                self.ListofAgents.append(x)
                first = False
                y +=1

            else:
                
                if (sparse == True) and (centralized != 2):  
                    posx = random_exclude(0, self.height-1, self.xpositions) #Each agent has a unique random position on the grid
                    posy = random_exclude(0, self.width-1, self.ypositions,)
                    
                    while (abs(posx - centerofgravity_x) + abs(posy - centerofgravity_y))  > 5: #while it is far, keep making it small so that the graph is connected 
                        posx = random_exclude(0, self.height-1, self.xpositions) #Each agent has a unique random position on the grid
                        posy = random_exclude(0, self.width-1, self.ypositions)
                        
                else:
                    posx = random_exclude(0, self.height-1, self.xpositions) #Each agent has a unique random position on the grid
                    posy = random_exclude(0, self.width-1, self.ypositions)
                    
                    x = random.randrange(0,len(self.xpositions)-1)  
                    while (abs(posx - self.xpositions[x]) + abs(posy - self.ypositions[x])>5):
                            posx = random_exclude(0, self.height-1, self.xpositions) #Each agent has a unique random position on the grid
                            posy = random_exclude(0, self.width-1, self.ypositions)



                x = Agent(y,posx,posy,self.height,self.width,self.noisy, self.alpha, self.omega_init, self.rho, self.phi, self.num, self.discount_factor)   
                self.xpositions.append(posx)
                self.ypositions.append(posy)
                self.ListofAgents.append(x)
                y = y + 1

        
        self.xpositions.remove(self.width)
        self.ypositions.remove(self.height)         

        self.target_posx = randomlocation.randrange(0, self.height-1)
        self.target_posy = randomlocation.randrange(0, self.width-1)
         
        

        self.CombinationMatrix = np.zeros((self.num,self.num)) 
        self.Combination_Matrix()
 
        return
        
    def transition_matrix_fn(self,a):
        self.transition_matrix =  np.ones(self.height*self.width)
        for i in range(len(self.states)): 
            #state is far from action
            if ((abs(self.states[i][0]-a[0]))  + (abs(self.states[i][1] - a[1]))) >= 4 : #far from action
                #Close to target's position
                if ((abs(self.target_posx-self.states[i][0]))  + (abs(self.target_posy - self.states[i][1]))) <= 4:
                        #high score
                        self.transition_matrix[i] =  100

                #Far from target's position        
                if ((abs(self.target_posx-self.states[i][0]))  + (abs(self.target_posy - self.states[i][1]))) > 4:
                        #medium score
                        self.transition_matrix[i] = 50
            #state is close to action
            if (((abs(self.states[i][0]-a[0]))  + (abs(self.states[i][1] - a[1])))) < 4 :
                #state is close  to target and action
                if ((abs(self.target_posx-self.states[i][0]))  + (abs(self.target_posy - self.states[i][1]))) <= 4:
                            #small score
                            self.transition_matrix[i] =  10
                #state is close to action and far from target
                if ((abs(self.target_posx-self.states[i][0]))  + (abs(self.target_posy - self.states[i][1]))) > 4:
                            #smaller score
                            self.transition_matrix[i] =  5
        sum = 0
        for j in range(len(self.transition_matrix)):
            sum += self.transition_matrix[j]
        
        for j in range(len(self.transition_matrix)):
            self.transition_matrix[j] = self.transition_matrix[j]/sum
        return
        

    def Combination_Matrix(self):
            """
            The Combination Matrix defines the weights assigned to the beliefs of the agents by other agents in the network.
            The network in the Decentralized Policy Evaluation, Decentralized Execution is not fully connected. The weights 
            between each 2 agents depends on the L1 distance between them. The further the agents are from eachother, the lower
            is the weight is.

            For the cases where training is centralized, the network is fully connected, where all the weights in the combination matrix
            are equal to 1/self.num 
            """ 
            if (self.centralized != 0):
                self.CombinationMatrix = np.zeros((self.num,self.num)) 
                for x in self.ListofAgents:
                    sum1 = 0
                    for x_1 in self.ListofAgents:
                        
                                if (abs(x.posx - x_1.posx) + abs(x.posy - x_1.posy)) < 3 :
                                    z = 90
                                    self.CombinationMatrix[x.idnum][x_1.idnum] = z
                                    sum1 += z
                                
                                elif (abs(x.posx - x_1.posx) + abs(x.posy - x_1.posy)) < 5 :
                                    z = 50
                                    self.CombinationMatrix[x.idnum][x_1.idnum] = z
                                    sum1 += z
                                
                                elif (abs(x.posx - x_1.posx) + abs(x.posy - x_1.posy)) > 7 :
                                    z = 0
                                    self.CombinationMatrix[x.idnum][x_1.idnum] = z
                                    sum1 +=z


                    for i in range(len(self.CombinationMatrix[x.idnum])):
                        self.CombinationMatrix[x.idnum][i] = self.CombinationMatrix[x.idnum][i]/sum1
                    

            else:
                self.CombinationMatrix = np.full((self.num,self.num), 1/self.num) 
 
            return




    # Real transition of the Target
    def Actual_transition(self,a):
        print("target transition")
        self.transition_matrix_fn(a)
        x = np.random.choice([i for i in range(len(self.states))], p= np.array(self.transition_matrix))  
        self.target_posx = self.states[x][0]
        self.target_posy = self.states[x][1] 
        return
 
    def Action(self,centralized):
        self.actionlist = []
        print("joint action -Gridworld")
        self.jointaction = [0,0]
        for x in self.ListofAgents:
            x.Action(centralized)
            self.jointaction[0] += x.action[0]
            self.jointaction[1] += x.action[1]
            self.actionlist.append(x.action)
            print("action for  ", x.idnum, ":", x.action )
            
        self.jointaction[0] = math.floor(self.jointaction[0]/self.num)
        self.jointaction[1] = math.floor(self.jointaction[1]/self.num)
        print("jointaction",self.jointaction)
        return
    
    def Random_Action(self):
        self.actionlist = []
        for x in self.ListofAgents:
            x.Randomized_action()
            self.jointaction[0] += x.action[0]
            self.jointaction[1] += x.action[1]
            self.actionlist.append(x.action)
        self.jointaction[0] = int(self.jointaction[0]/self.num)
        self.jointaction[1] = int(self.jointaction[1]/self.num)
        return
    
    def Random_Policy(self):
        self.actionlist = []
        for x in self.ListofAgents:
            x.Random_Policy()
            self.jointaction[0] += x.action[0]
            self.jointaction[1] += x.action[1]
            self.actionlist.append(x.action)
        self.jointaction[0] = int(self.jointaction[0]/self.num)
        self.jointaction[1] = int(self.jointaction[1]/self.num)
        return
    
 
     
    def Reward(self):
        r = 0
        for x in self.ListofAgents:
            x.Reward(self.target_posx,self.target_posy)
            r += x.reward
        self.cent_r = r/self.num #average of rewards
        print("Average Reward : ", self.cent_r)
        return
      
     
    def Observe(self,centralizedtraining):
        # w = []
       
        for x in self.ListofAgents:
             x.MakeObservation(self.target_posx,self.target_posy) 

        if (centralizedtraining == 0) or (centralizedtraining == 2):
            print("jointobs - gridworld")
            self.jointobservation = []
            for j in range(len(self.states)):
                r = 1
                for x in self.ListofAgents:
                    y = x.ObsMatrix[j][x.observedstate]
                    r = y*r
                self.jointobservation.append(r)
        return
  

    def Val_approx(self): 
        total = 0
            
        
        for x in self.ListofAgents:
            total += x.Val(self.beliefvectors)
        
        self.val = total*self.phi/self.num     #Mean Value function of all agents
        return self.val

     
    def Render(self): 
         
        fig, ax = plt.subplots()
        plt.rcParams["figure.figsize"] = (10,10)
        xy = (0.5,0.5)
        arr_img = plt.imread("background.jpg")
        imagebox = OffsetImage(arr_img, zoom=1)
        imagebox.image.axes = ax
        ab = AnnotationBbox(imagebox, xy,frameon = False)

        ax.add_artist(ab)
        for L in self.ListofAgents:
            temp_x = L.posx
            temp_y = L.posy
            xy = (temp_x/self.width ,(temp_y)/self.height)
            arr_img = plt.imread("satellite.png")
            imagebox = OffsetImage(arr_img, zoom=0.3) 
            imagebox.image.axes = ax

            ab = AnnotationBbox(imagebox, xy,frameon = False)

            ax.add_artist(ab) 

        temp_x = self.jointaction[0]/self.height
        temp_y = (self.jointaction[1] )/self.width

        xy = (temp_x,temp_y)
        arr_img = plt.imread("shot.png")

        imagebox = OffsetImage(arr_img, zoom=0.3)
        imagebox.image.axes = ax

        ab = AnnotationBbox(imagebox, xy,frameon = False)

        ax.add_artist(ab)
        ax.axis('off') 
        temp_x = self.target_posx/self.height
        temp_y = (self.target_posy )/self.width

        xy = (temp_x,temp_y)
        arr_img = plt.imread("drone.png")
        imagebox = OffsetImage(arr_img, zoom=1) 
        imagebox.image.axes = ax 
        ab = AnnotationBbox(imagebox, xy,frameon = False)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.add_artist(ab)
        ax.axis('off') 

        plt.rcParams["figure.figsize"] = (10,10)
        fig.savefig("render.jpg",dpi = 300)
        plt.close(fig)
        time.sleep(10) 
        return 
    
    def reset(self,centralized): 
        print("reset")
        self.centralized = centralized 
        self.Combination_Matrix()
        self.val = 0
        self.cent_r = 0
        self.jointobservation = [1/(height*width)]*(height*width)
        
        x = np.random.choice([i for i in range(len(self.states))], p= self.transition_matrix)       
                
        self.target_posx = self.states[x][0]
        self.target_posy = self.states[x][1] 
 
        self.omegalistavg = []
        self.ErrorList = []
        self.Errorhistory = []
        self.Errorhistory = cu.array(np.float64(self.Errorhistory))

        self.sbehistory = []
        self.sbehistory = cu.array(self.sbehistory)

        for agent in self.ListofAgents:
            agent.reset(centralized, self.omega_init, self.omega_dd_init)        
        return

    def sbe(self):
        sbe = 0 
        for x in self.ListofAgents:
            y = x.td_error
            sbe += y*y

        sbe = sbe/self.num
        print("Sbe Error: ", sbe)
        self.sbehistory = cu.append(self.sbehistory,cu.mean(np.float64(sbe))) 
        return

    def Error(self):
        y = []
        for x in self.ListofAgents:
            y.append(x.omegalist)
        
        self.omegalistavg = []

        for x in range(len(y[0])):
            r = 0
            for t in range(self.num):
                r += y[t][x]
            
            self.omegalistavg.append(r/self.num) 
        
        self.ErrorList = []
        
        for x in self.ListofAgents:
            y = x.Error(self.omegalistavg)
            self.ErrorList.append(y)

        self.omegalistavg = [] 
        self.Errorhistory = cu.append(self.Errorhistory,cu.mean(cu.array(np.float64(self.ErrorList))))
        return


    def step(self, a, centralizedtraining):
        print("Observe- Gridworld")
        self.Observe(centralizedtraining)
            
        for x in self.ListofAgents:
            if centralizedtraining == 0:
                if x.idnum == 1:
                    print("Centralized_Adapt- Gridworld")
                x.Centralized_Adapt(self.jointobservation) 

            elif centralizedtraining == 1:
                if x.idnum == 1:
                    print("Dec Adapt- Gridworld")
                x.Adapt(self.gamma) #Get m_i

            else:
                if x.idnum == 1:
                    print("Algo3: Dec Adapt & Cent Adapt- Gridworld")
                x.Centralized_Adapt(self.jointobservation) 
                x.Adapt(self.gamma)

        #EtaList_Centralized = []
        if centralizedtraining != 0: 
            EtaList = []
            for x in self.ListofAgents: 
                EtaList.append(x.n) 

            for x in self.ListofAgents: 
                if x.idnum ==1:
                    print("Combine - Gridworld")
                x.Combine(self.num, self.CombinationMatrix, EtaList)


        if (a == 0):
            self.Action(centralizedtraining) 
        

        self.Render()

        #observe, get reward
        self.Reward()
 
        for x in self.ListofAgents:
            if centralizedtraining == 0 or centralizedtraining == 2: 
                x.Centralized_Evolve(self.jointaction)

            if centralizedtraining != 0: 
                #Decentralized
                ank = []    
                anc_num = 0
                y = self.CombinationMatrix[x.idnum]
                for i in range(len(y)):
                    if y[i] > 0:
                        ank.append(self.actionlist[i]) #actions of neighbors
                    else:
                        anc_num += 1 #non neighbors
                print("Evolve", x.idnum)
                x.Evolve(ank,anc_num) #Get n_i+1

        
        print("TD Error - Gridworld")

        for x in self.ListofAgents:
           
            if centralizedtraining != 1:  
                x.TD_Error_Centralized(self.cent_r)

            #Decentralized
            else:
                x.TD_Error()
        
        #Decentralized
        if centralizedtraining == 1: 
            OmegaList = []
            for x in self.ListofAgents:
                OmegaList.append(x.omegalist)   

            for x in self.ListofAgents: 
                x.Diffusion_omega(self.CombinationMatrix, OmegaList)  
         
        self.Actual_transition(self.jointaction)  

        self.Error() 
        self.sbe()
         
        return
    
    # #done
    def smooth(self,m):
        r = [] 
        for y in range(len(m[0])): #for every iteration
                v = []
                for x in range(len(m)): #for every experiment
                    v.append(m[x][y])  
                e = np.mean(v) 
                r.append(e)
            
        return r

 
beliefvectors = []
#Parameters
height = 10
width  = height
num = 8
iterations = 12001
experiments = 7
rho = 0.0001
phi = 1
noisy = 1
alpha  = 0.1
np.random.seed(200)

keyword = str(num) + "_height_" + str(height) +  "_iterations_" + str(iterations)+ "_rho_" + str(rho) + "phi"+ str(phi) + "_exp_"+str(experiments)+"_alpha_" +str(alpha)

#Belief Vectors
for i in range(100000):
        r = [random.randint(1,100) for w in range(0, height*width)]
        s = sum(r)
        r = [ j/s for j in r ]
        beliefvectors.append(r)

env = GridWorld(num = num, height = height, width = width, centralized = 0, noisy = noisy, rho = rho, phi = phi, sparse = True, alpha = alpha, beliefvectors = beliefvectors)

#Communication Topology Graph
Gr = nx.from_numpy_array(env.CombinationMatrix) 

Dict = dict()
for x in env.ListofAgents:
    w = (x.posx,x.posy)
    Dict[x.idnum] = w
print(Dict)
pos1 = nx.kamada_kawai_layout(Gr, dist=None, pos=Dict, weight='weight', scale=1, center=None, dim=2)

fig, ax = plt.subplots(1, 1) 
plt.axis('off')
plt.xlim([-1.3, 1.1])
plt.ylim([-1.1, 1.1])



nx.draw_networkx_nodes(Gr, pos=pos1, node_color='#3C99DC',nodelist=range(0, env.num), node_size=400, edgecolors='white', linewidths=.5)
nx.draw_networkx_edges(Gr, pos=pos1, node_size=200, edge_color = "#4f4f4f", alpha=1, arrowsize=6, width=3)

tr_figure = ax.transData.transform
# Transform from display to figure coordinates
tr_axes = fig.transFigure.inverted().transform

    
fig.savefig('network.png', bbox_inches='tight', pad_inches = 0)
fig.show()        



#Experiments
for k in range(3): 
        env.reset(k)
 
        y = ['Argmax: '] 
        p = ['Centralized Evaluation, Centralized Execution', 'Decentralized Evaluation, Decentralized Execution',  'Centralized Evaluation, Decentralized Execution']
        

        d = str(num)+ '-'+ str(height) + keyword
        d1 = d + '.txt'
        
        j = 1
            
        #for more than 1 monte carlo experiment
        for m in range(experiments):  
            
            for i in range(iterations):
                #print(env.CombinationMatrix)
                plt.rcParams["figure.figsize"] = (10,10)    
                if (k == 2):
                    print("Centr: CD, iter", i, "exp", m)
                    env.step(j, centralizedtraining = 2)
                elif (k == 1):
                    print("Centr: DD, iter", i, "exp", m)
                    env.step(j, centralizedtraining = 1)
                elif (k == 0):
                    print("Centr: CC, iter", i,"exp", m)
                    env.step(j, centralizedtraining = 0)

                w2 = env.sbehistory
                fiii, ax = plt.subplots(1,1)
                ax.plot(cu.asnumpy(w2))
                ax.set_yscale('log') 
                ax.set_title("SBE Error")
                fiii.savefig(d + str(k) + 'SBEbyiter.png')
                plt.close(fiii)

                q = env.Errorhistory
                fiii, ax =plt.subplots(1,1)
                ax.plot(cu.asnumpy(q)) 
                ax.set_yscale('log')
                ax.set_title("Error History")
                fiii.savefig(d +  str(k) +'ERRORbyiter.png')
                plt.close(fiii)

                if i%100== 0:
                    data = env.Errorhistory[-100:]
                    with open('AgreementErrorHISTORY'+ d +'-'+str(k)+'-'+str(m)+'.csv', 'a', encoding="ISO-8859-1", newline='') as file:
                        write = csv.writer(file) 
                        write.writerows(map(lambda x: [x], data))

                    data = env.sbehistory[-100:]
                    with open('SBEHISTORY'+ d +'-'+str(k)+'-'+str(m)+'.csv', 'a', encoding="ISO-8859-1", newline='') as file:
                        write = csv.writer(file) 
                        write.writerows(map(lambda x: [x], data))
            
                
            env.reset(k)
