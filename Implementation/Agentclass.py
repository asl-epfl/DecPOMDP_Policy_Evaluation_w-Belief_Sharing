""" 
Agent Class
 
"""
from cmath import nan
from statistics import mean

import numpy as np 
import random
import os 

class Agent:

    # Observation matrix   
     
    def Observe(self, target_posx, target_posy): #likelihood function based on location of target
        """
        Since we are simulating a POMDP environment, the observation made by each agent is noisy. To simulate that,
        higher confidence was given to the position of the target if it is close in proximity to the agent. Otherwise, 
        the larger the distance between the agent and the target, the higher the noise, the less certain the agent is about
        the location of the target.
        """
        if (self.noisy == 1):
            obs = np.ones(self.height*self.width)
            if (abs(target_posx - self.posx) + abs(target_posy-self.posy)) < 3: #target is very close to agent
                obs = np.zeros(self.height*self.width)
                for i in range(len(obs)):
                    x = self.agentstates[i] #corresponding state in states matrix
                    if (abs(x[0] - target_posx) + abs(x[1]-target_posy) ) ==0: #cell is close to object
                        obs[i] = 400

                    elif (abs(x[0] - target_posx) + abs(x[1]-target_posy) ) <= 2: #cell is close to object
                        obs[i] = 200
                    
                    elif abs(x[0] - target_posx) + abs(x[1]-target_posy) <5: #cell is far to object
                        obs[i] = 30
                    
                    
            elif abs(target_posx - self.posx) + abs(target_posy-self.posy) < 6: #target is slightly close to agent
                obs = np.ones(self.height*self.width)
                for i in range(len(obs)):
                    x = self.agentstates[i] #corresponding state in states matrix
                    if (abs(x[0] - target_posx) + abs(x[1]-target_posy) ) == 0: #cell is close to object
                        obs[i] = 200

                    elif (abs(x[0] - target_posx) + abs(x[1]-target_posy) ) < 2: #cell is close to object
                        obs[i] = 180
                    
                    elif abs(x[0] - target_posx) + abs(x[1]-target_posy) < 5: #cell is far to object
                        obs[i] = 100
            
            elif abs(target_posx - self.posx) + abs(self.posy-self.posy) > 6: #target is far from agent
                obs = np.ones(self.height*self.width)
                for i in range(len(obs)):
                    x = self.agentstates[i] #corresponding state in states matrix
                    if abs(x[0] - target_posx) + abs(x[1]-target_posy) < 7: #cell is close to object (wider circle, less certainty)
                        obs[i] = 25
                    
                    elif abs(x[0] - self.posx) + abs(x[1]-target_posy) < 9: #cell is far from object
                        obs[i] = 4

        elif (self.noisy == 0): #less noisy
            obs = np.ones(self.height*self.width)
            if (abs(target_posx - self.posx) + abs(target_posy-self.posy)) < 3: #target is very close to agent
                obs = np.zeros(self.height*self.width)
                for i in range(len(obs)):
                    x = self.agentstates[i] #corresponding state in states matrix
                    if (abs(x[0] - target_posx) + abs(x[1]-target_posy) ) ==0: #cell is close to object
                        obs[i] = 300

                    elif (abs(x[0] - target_posx) + abs(x[1]-target_posy) ) < 2: #cell is close to object
                        obs[i] = 80
                    
                    elif abs(x[0] - target_posx) + abs(x[1]-target_posy) < 5: #cell is far to object
                        obs[i] = 20
                    
                    
            elif abs(target_posx - self.posx) + abs(target_posy-self.posy) < 6: #target is slightly close to agent
                obs = np.ones(self.height*self.width)
                for i in range(len(obs)):
                    x = self.agentstates[i] #corresponding state in states matrix
                    if (abs(x[0] - target_posx) + abs(x[1]-target_posy) ) == 0: #cell is close to object
                        obs[i] = 300

                    elif (abs(x[0] - target_posx) + abs(x[1]-target_posy) ) < 2: #cell is close to object
                        obs[i] = 150
                    
                    elif abs(x[0] - target_posx) + abs(x[1]-target_posy) < 5: #cell is far to object
                        obs[i] = 50
            
            elif abs(target_posx - self.posx) + abs(self.posy-self.posy) > 10: #target is far from agent
                obs = np.ones(self.height*self.width)
                for i in range(len(obs)):
                    x = self.agentstates[i] #corresponding state in states matrix
                    if abs(x[0] - target_posx) + abs(x[1]-target_posy) < 4: #cell is close to object (wider circle, less certainty)
                        obs[i] = 25
                    
                    elif abs(x[0] - self.posx) + abs(x[1]-target_posy) < 9: #cell is far from object
                        obs[i] = 10

        elif (self.noisy == 2): #more noisy
            obs = np.ones(self.height*self.width)
            if (abs(target_posx - self.posx) + abs(target_posy-self.posy)) < 3: #target is very close to agent
                obs = np.zeros(self.height*self.width)
                for i in range(len(obs)):
                    x = self.agentstates[i] #corresponding state in states matrix
                    if (abs(x[0] - target_posx) + abs(x[1]-target_posy) ) ==0: #cell is close to object
                        obs[i] = 300

                    elif (abs(x[0] - target_posx) + abs(x[1]-target_posy) ) < 2: #cell is close to object
                        obs[i] = 200
                    
                    elif abs(x[0] - target_posx) + abs(x[1]-target_posy) < 5: #cell is far to object
                        obs[i] = 100
                    
                    
            elif abs(target_posx - self.posx) + abs(target_posy-self.posy) < 6: #target is slightly close to agent
                obs = np.ones(self.height*self.width)
                for i in range(len(obs)):
                    x = self.agentstates[i] #corresponding state in states matrix
                    if (abs(x[0] - target_posx) + abs(x[1]-target_posy) ) == 0: #cell is close to object
                        obs[i] = 300

                    elif (abs(x[0] - target_posx) + abs(x[1]-target_posy) ) < 2: #cell is close to object
                        obs[i] = 200
                    
                    elif abs(x[0] - target_posx) + abs(x[1]-target_posy) < 5: #cell is far to object
                        obs[i] = 50
            
            elif abs(target_posx - self.posx) + abs(self.posy-self.posy) > 10: #target is far from agent
                obs = np.ones(self.height*self.width)
                for i in range(len(obs)):
                    x = self.agentstates[i] #corresponding state in states matrix
                    if abs(x[0] - target_posx) + abs(x[1]-target_posy) < 4: #cell is close to object (wider circle, less certainty)
                        obs[i] = 25
                    
                    elif abs(x[0] - self.posx) + abs(x[1]-target_posy) < 9: #cell is far from object
                        obs[i] = 10
         
        #normalize
        sum2 = 0
        for i in range(self.height*self.width):
            sum2 += obs[i]
        
        for i in range(len(self.obs)):
            obs[i] = obs[i]/sum2 #Normalizing the scores, probabilities add up to 1
            
        self.obs = obs
        
        return obs

    #Full observation matrix for each agent, probability vectors for all target states
    def ObservationMatrix(self): 
        self.ObsMatrix = []       
        for i in self.agentstates:
            x = self.Observe(i[0],i[1]) 
            self.ObsMatrix.append(x)   
        return

    def reset(self,centralized,omega,omegadd):
        print(self.omegalist)
        self.centralized_n = [1/(self.height*self.width)]*(self.height*self.width)
        self.centralized_m = [1/(self.height*self.width)]*(self.height*self.width)
        self.centralized_m = np.longdouble(self.centralized_m) 
        self.approx_transition_model =  [1/(self.height*self.width)]*(self.height*self.width)
        self.cen_transition_matrix_byagent =  [1/(self.height*self.width)]*(self.height*self.width)
        
        #Beliefs
        self.n = [1/(self.height*self.width)]*(self.height*self.width)
        self.m = [1/(self.height*self.width)]*(self.height*self.width) 
        self.errorhistory = []
        
        #Omega
        self.td_error = 0
        self.reward = 0
        self.action = []
        if centralized == 1 :
            self.omegalist = np.longdouble(omegadd[self.idnum])
            
        else:
            self.omegalist = np.longdouble(omega)
 
        #Value of Value Function
        self.val = 0  

        self.error = 0

        self.omegahistory = []
        
        #Observations
        self.ObsMatrix = []
        self.observation = [] #Coordinates of the guessed state
        self.observedstate = nan #Index of observed state        
                
        self.obs = [1/(self.height*self.width)]*(self.height*self.width)
        self.ObservationMatrix()
        return

    

    def __init__(self, idnum,posx=0,posy=0,height=0, width=0, noisy = False, alpha = 0.1, omega = [10]*(100), rho = 0.01, phi = 1, num = 1,discount_factor = 0.95):
        self.rho = rho
        self.phi = phi
        self.noisy = noisy
        self.error = 0
        self.idnum = idnum
        self.discount_factor = discount_factor
        self.posx = posx
        self.posy = posy
        self.height = height
        self.width = width 
        self.approx_transition_model =  np.ones(self.height*self.width)
        self.action = []
        self.reward = 0
        self.omegahistory = []
        self.num = num
        self.alpha = alpha
        #States 
        x = 0
        self.agentstates  = [0]*(self.height*self.width)
        for i in range(self.width):
            for j in range(self.height):
                self.agentstates[x] = [i,j]
                x+=1
        
        #Centralized Beliefs
        self.centralized_n = [1/(height*width)]*(height*width)
        self.centralized_m = [1/(height*width)]*(height*width)
        self.centralized_m = np.longdouble(self.centralized_m)
        self.centralized_m_update = [self.centralized_m,self.centralized_m]
        self.cen_transition_matrix_byagent = [1/(height*width)]*(height*width)
        #Beliefs
        self.n = [1/(height*width)]*(height*width)
        self.m = [1/(height*width)]*(height*width)
        self.m_update = [self.m,self.m]
        
        #Omega
        self.td_error = 0 
        self.omegalist = omega 
        self.omegahistory = []
        self.errorhistory = []
        #Value of Value Function
        self.val = 0

        #Observations
        self.ObsMatrix = []
        self.observation = [] #Coordinates of the guessed state
        self.observedstate = nan #Index of observed state        
                
        self.obs = np.ones(self.height*self.width)
        self.ObservationMatrix()
        return

     
    def MakeObservation(self,target_posx,target_posy):
        if (self.idnum == 1):
            print("MakeObs - Agent")
        y =  self.Observe(target_posx,target_posy) #probability vector for the likelihood of all states
        
        x = np.random.choice([i for i in range(len(self.agentstates))], p = y)
        self.observation = self.agentstates[x] #in the form of [x,y]
         
        self.observedstate = x #index of observed state
        
        return 

    def approx_transition_model_fn(self,s,a):

        self.approx_transition_model =  np.ones(self.height*self.width)
        for i in range(len(self.agentstates)): 
            #state is far from action
            if ((abs(self.agentstates[i][0]-a[0]))  + (abs(self.agentstates[i][1] - a[1]))) >= 4 : #far from action
                #Close to target's position
                if (abs(s[0]-self.agentstates[i][0]))  + (s[1] - self.agentstates[i][1]) <= 4:
                        #high score
                        self.approx_transition_model[i] =  100

                #Far from target's position        
                if (abs(s[0]-self.agentstates[i][0]))  + (s[1] - self.agentstates[i][1]) > 4:
                        #medium score
                        self.approx_transition_model[i] = 50
            #state is close to action
            if ((abs(self.agentstates[i][0]-a[0]))  + (abs(self.agentstates[i][1] - a[1]))) < 4 :
                #state is close  to target and action
                if (abs(s[0]-self.agentstates[i][0]))  + (s[1] - self.agentstates[i][1]) <= 4:
                            #small score
                            self.approx_transition_model[i] =  10
                #state is close to action and far from target
                if (abs(s[0]-self.agentstates[i][0]))  + (s[1] - self.agentstates[i][1]) > 4:
                            #smaller score
                            self.approx_transition_model[i] =  5
        sum = 0
        # for i in range(len(self.approx_transition_model)):
        #     sum += self.approx_transition_model[i]
        sum = np.sum(self.approx_transition_model)
        self.approx_transition_model = self.approx_transition_model/sum
        
        return

    def Adapt(self, gamma): 
 
         if (self.idnum == 1):
            print("Adapt - Agent")
         normalize = 0 
         for x in range(len(self.agentstates)):#for every theta i 
             y = self.ObsMatrix[x][self.observedstate] #probability of being at observed state, given it was in state s'
            
             self.m[x] = (y**gamma)*self.n[x]
              
             normalize += self.m[x]
          
         for x in range(len(self.agentstates)):
             self.m[x] = self.m[x] / normalize
         return  
     
        
     
    def Evolve(self, ank, ac_num):  
        if (self.idnum == 1):
            print("Evolve - Agent")
        jointaction_approx = [0,0]
        self.n = np.zeros(self.height*self.width)  
        for s in range(self.height*self.width):  
                for sprime_i in range(self.height*self.width):
                    sprime = self.agentstates[sprime_i]
                    #joint action approximation
                    #agent assumes that non neighbors accurately hit sprime
                    jointaction_approx[0] = ac_num*sprime[0] #non neighbors
                    jointaction_approx[1] = ac_num*sprime[1]

                    for i in range(len(ank)):
                        jointaction_approx[0] += ank[i][0] #neighbors actions (known)
                        jointaction_approx[1] += ank[i][1]

                    #average of joint action approx
                    jointaction_approx[0] = jointaction_approx[0]/self.num 
                    jointaction_approx[1] = jointaction_approx[1]/self.num

                    self.approx_transition_model_fn(sprime, jointaction_approx)
                    #print("approx")
                    self.n[s] += self.approx_transition_model[s]*self.m[sprime_i]
        if (self.idnum == 9):
            print("left evolve")
        return
    


    def Combine(self,num, CombinationMatrix, EtaList):
        if (self.idnum == 1):
            print("Combine - Agent")
        normalize = 0 
        self.n = np.ones(self.height*self.width)
        for i in range(self.width*self.height):  #for every state
            # self.n[i] = 1 #np.ones
            for j in range(num): #for every agent
                self.n[i] = self.n[i]*(EtaList[j][i]**(CombinationMatrix[self.idnum][j]))
                
            normalize += self.n[i]

        for i in range(self.width*self.height):          
            self.n[i] = self.n[i]/normalize 
        return


    def TD_Error(self): 
        if (self.idnum == 1):
            print("TD_Error - Agent")
            #print(self.omegalist)
            
        self.td_error = self.reward + self.discount_factor * self.phi * np.dot(self.omegalist, np.array(self.n)) - self.phi * np.dot(self.omegalist, self.m)
        
        gradient = self.m
        for i in range(len(self.omegalist)):
            self.omegalist[i] = self.omegalist[i]*(1-2*self.alpha*self.rho) + self.alpha*(self.td_error)*gradient[i]*self.phi
        if (self.idnum == 1):
            print("TD_Error Centralized - Agent")
            #print("self.td_error", self.td_error)
        return


    def Action(self,centralized): 
        if (self.idnum == 1):
            print("Action - Agent") 

        if centralized == 0:
            x = np.argmax(self.centralized_m)
        else:
            x = np.argmax(self.m) 
        self.action = self.agentstates[int(x)]
        return 
    
      
    def Randomized_action(self): 
        self.action = self.agentstates[np.random.choice([i for i in range(len(self.agentstates))], p =self.m)] #random choice among all states, with replacement, with transition probabilities corresponding to those from its previous state
        return      
    
    def Random_Policy(self):
        self.action = self.agentstates[np.random.choice([i for i in range(len(self.agentstates))])]
        return
    


    def Reward(self, target_posx,target_posy):
        
        if (self.action[0] == target_posx) and (self.action[1] == target_posy):
            self.reward = 1
        elif (abs(self.action[0] - target_posx) + abs(self.action[1] - target_posy)) <= 3:
             self.reward  = 0.2
        else:
            self.reward = 0
        if (self.idnum == 1):
            print("Reward - Agent: ") #, self.reward)
            print("Target Position: [", target_posx,",",target_posy,"]")
        return
    

    


    def Centralized_Adapt(self, jointobservation): 
        if (self.idnum == 1): 
            print("Centralized Adapt - Agent") 
        
        normalize = 0  

        for x in range(len(self.agentstates)):#for every theta i 
            y = jointobservation[x]
            
            self.centralized_m[x] = np.longdouble(y*self.centralized_n[x])
            
            normalize += self.centralized_m[x]
            
        for x in range(len(self.agentstates)):
            self.centralized_m[x] = self.centralized_m[x] / normalize 
        return 

    #Transition matrix calculation performed by the agent
    def centralized_transition_matrix_by_agent_fn(self,s,a):
        self.cen_transition_matrix_byagent =  np.ones(self.height*self.width)
        for i in range(len(self.agentstates)): 
            #state is far from action
            if (((abs(self.agentstates[i][0]-a[0]))  + (abs(self.agentstates[i][1] - a[1])))) >= 4 : #far from action
                #Close to target's position
                if ((abs(s[0]-self.agentstates[i][0]))  + (abs(s[1] - self.agentstates[i][1]))) <= 4:
                        #high score
                        self.cen_transition_matrix_byagent[i] =  100

                #Far from target's position        
                if ((abs(s[0]-self.agentstates[i][0]))  + (abs(s[1] - self.agentstates[i][1]))) > 4:
                        #medium score
                        self.cen_transition_matrix_byagent[i] = 50
            #state is close to action
            if (((abs(self.agentstates[i][0]-a[0]))  + (abs(self.agentstates[i][1] - a[1])))) < 4 :
                #state is close  to target and action
                if ((abs(s[0]-self.agentstates[i][0]))  + (abs(s[1] - self.agentstates[i][1]))) <= 4:
                            #small score
                            self.cen_transition_matrix_byagent[i] =  10
                #state is close to action and far from target
                if ((abs(s[0]-self.agentstates[i][0]))  + (abs(s[1] - self.agentstates[i][1]))) > 4:
                            #smaller score
                            self.cen_transition_matrix_byagent[i] =  5
        sum = 0
        for i in range(len(self.cen_transition_matrix_byagent)):
            sum += self.cen_transition_matrix_byagent[i]
            
        self.cen_transition_matrix_byagent = self.cen_transition_matrix_byagent/sum
        return

    def Centralized_Evolve(self,jointaction):  
        if (self.idnum == 1):
            print("Centralized_Evolve - Agent") 
        self.centralized_n = np.zeros(self.height*self.width)  
        for s in range(self.height*self.width):  #s index
            for sprime_i in range(self.height*self.width): #sprime index
                sprime = self.agentstates[sprime_i] #sprime = [x,y]
                self.centralized_transition_matrix_by_agent_fn(sprime,jointaction) 
                self.centralized_n[s] += (self.cen_transition_matrix_byagent[s])*self.centralized_m[sprime_i]  
        return

    
    def TD_Error_Centralized(self,reward): 
        if (self.idnum == 1):
            print("TD_Error Centralized - Agent") 

        self.td_error = reward + self.discount_factor * self.phi * np.longdouble(np.dot(self.omegalist, self.centralized_n)) - self.phi * np.longdouble(np.dot(self.omegalist,self.centralized_m))
        
        gradient = self.centralized_m
        
        for i in range(len(self.omegalist)):
            self.omegalist[i] = np.longdouble(self.omegalist[i]*(1-2*self.alpha*self.rho)) + np.longdouble(self.alpha*(self.td_error)*gradient[i]*self.phi)
         
        if (self.idnum == 1):
            print("TD_Error Centralized - Agent") 
        return 


    def Diffusion_omega(self,CombinationMatrix, OmegaList):
        if (self.idnum == 1):
            print("Diffusion - Agent")
        self.omegalist = np.zeros(self.height*self.width)
        for i in range(len(self.agentstates)):
            for j in range(self.num):   #Omega of each agent
                self.omegalist[i] += OmegaList[j][i]*(CombinationMatrix[self.idnum][j]) #Arithmetic Mean
         
        return
    
    def Val(self, beliefvectors):
        if (self.idnum == 1):
            print("Val - Agent")
        v = []
        for i in range(100000):
            val = np.dot(beliefvectors[i],self.omegalist)
            v.append(val)

        self.val = mean(v)
        return self.val
    
    def Error(self,omegalistavg):
        x = []
        if (self.idnum == 1):
            print("Error - Agent") 

        for i in range(len(self.omegalist)):
            x.append(self.omegalist[i] - omegalistavg[i])
        self.error = np.dot(np.array(x).T,np.array(x)) 
        return self.error