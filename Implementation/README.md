# Code Overview
The code is formed of two classes: 
  - Gridworld
  - Agent


# Gridworld
### Attributes/Variables:
- `num`: number of agents in the environment.

- `height`: height of the grid (number of cells)

- `width`: width of the grid

- `ListofAgents`: List containing all agents found in the environment.
  
- `CombinationMatrix` :
   The entries of the combination matrix are set such that they are inversely proportional to the ℓ1-norm distance between the agents. That is to say, the further
the agents are from each other, the less is the weight assignedto the edge connecting them. The weights smaller than some threshold are set to 0.

- `transition_matrix` (1xm) :
  A probability vector which reveals the likelihood of a cell being the next location of the target. This probability depends on the current location of the target and the location of agents’ hits.
  
- `states` (1xm) :
  contains all states found in the grid
  each state is defined by its x-y coordinates [x,y] depending on the location of the cell.

- `target_posx` (1x1) :
  x-coordinate of the target's location
  
- `target_posy` (1x1) :
  y-coordinate of the target location
  
- `Errorhistory`: stores value of network disagreeement error at each iteration

- `sbehistory`: stores value of sbe error at each iteration 

### General Methods

- `Combination_Matrix()`:
  establishes the entries of the combination matrix according to the ℓ1-norm distance between the agents.

- `transition_matrix_fn(self,a)`:
  establishes the entries of the transition matrix
  
  **Input**: 
    Joint action by agents
    
  **Procedure**
    
    Assigns scores to each state according to the ℓ1-distance from each state to the location of the target and to joint action formed by the agents’ hits. 
  
  **Output**:
  
    `transition_matrix`: a vector of probabilities that dictate the probability of transitioning to next state depending on the joint action of the agents.
  
- `Actual_transition(a)`:
  determines the state that the moving target will transition to in the next time-step. 
  
  - **Input**:
  
    Joint action a

  - **Procedure**:

    1- Updates `transition_matrix` by calling function `transition_matrix_fn` with input action `a`

    2- Chooses a state randomly (based on probabilities from *self*.transition_matrix)

    3- Updates target position coordinates (`target_posx`, `target_posy`)

  - **Output:**
  
    `target_posx`
    
    `target_posy`
    
- `Action()`:
  
  - **Input**:
    No input

  - **Procedure**:

    1- Every agent in `ListofAgents` takes an action

    2- `jointaction` is a tuple such that 

      1. `jointaction[0]` is the integer closest to the average of the x-coordinates of the actions of the agents
      2. `jointaction[1]` is the integer closest to the average of the y-coordinates of the actions of the agents

    Therefore, `jointaction` is a state on the grid such that it is the average of the actions taken by all agents in the system

  - **Output**:

    `jointaction`
    
- `Render()`:
visualize the actions of the agents and the transition of the target in the environment. The agents are represented by sensors. These sensors send narrow sector jamming beam towards the location estimate of the target intruder drone. However, upon applying those signals, the drone detects energy abnormalities and transitions into another location by favoring locations that are far from the jamming signal. The joint action taken by the sensors is represented by a noisy signal.

- `Reset(centralized)`:
  - **Input**:
  
    Type of algorithm we are running
  
  - **Procedure**: 
   
    Resets the environment while maintaining its size, number of agents, combination matrix, and transition matrix.
 
- `Step(centralized)`:

  - **Input**:
  
    Type of algorithm we are running
  
  - **Procedure**: 
   The following actions are generally taken during each step: (These steps vary depending on the type of algorithm)
    - Observe
    - Adapt
    - Action
    - Reward
    - Evolve
    - TD-Error
    - Target transition
    - Network Agreement Error
    - SBE Error 

# Agent
### Attributes/Variables
- `obs` (1xm): observation vector that reveals how certain the agent is about the target's position in the grid.

- `n`(1xm) : decentralized prior belief vector  

- `m` (1xm) : updated decentralized belief vector  

- `centralized_n` (1xm) : centralized prior belief vector

- `centralized_m` (1xm) : updated centralized prior belief vector

- `states` (1xm) :
  contains all states found in the grid
  each state is defined by its x-y coordinates [x,y] depending on the location of the cell.
  
- `reward`: reward received by the agent at iteration i.   

- `action`: action taken by the agent

### General Methods

- `Make_Observation()`:
   Since we are simulating a POMDP environment, the observation made by each agent is noisy. To simulate that,
        higher confidence was given to the position of the target if it is close in proximity to the agent. Otherwise, 
        the larger the distance between the agent and the target, the higher the noise, the less certain the agent is about
        the location of the target.
- `Action(centralized)`:
   The action taken by each agent corresponds to the maximum entry of its belief vector.
   
- `Reward(target_posx,target_posy)`:
  
  **Input**:
    Target position on the grid
  
  **Procedure**:
  - The agent receives a reward of 1 if it hits the location of the target.
  - If the agent hits a location that is 3 grid units away form the position of the target, it receives a reward of 0.2.

- Depending on the type of algorithm chosen, an agent can perform :
    - `Evolve()`
    - `Centralized_Evolve(jointaction)`
    - `Adapt()`
    - `Centralized_Adapt()`
    - `Combine()`
 
- For TD Error calculation, an agent can use:
    - `TD_Error_Centralized(reward)`
    - `TD_Error()`

- Transition Matrix: Transition matrix calculation performed by the agent.
    - `centralized_transition_matrix_by_agent_fn(s,a)`
    - `approx_transition_model_fn(s,a)`
    
  

