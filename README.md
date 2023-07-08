# Policy Evaluation in Decentralized POMDPs with Belief Sharing.

Multi-agent grid world environment with partial observability and a discrete action space. In this environment, agents (e.g., radar sensors) exploit their communication network in order to track a moving target (e.g., an intruder drone). The target moves randomly according to a pre-defined transition model that takes the actions (i.e., hits) of agents into account. 

Used in the paper [Policy Evaluation in Decentralized
POMDPs with Belief Sharing](https://ieeexplore.ieee.org/abstract/document/10129007).

## Getting started:
 
- Known dependencies: Python (3.7.4), Numpy (1.14.5), Cupy(10.2), Matplotlib(3.1.1), CUDA Toolkit, Networkx


## Code Structure
  - `./code/Gridworld.py/`: contains code for initializing the Grid, assigning agent locations, environment simulation, target transition, rendering, and step function.
  
  - `./code/Agent.py/`: contains  agent class which performs multiple functions per agent such as agent initialization, performing observations, taking individual policies, and forming beliefs.

### Creating new environments

Use the following function to create a new environment: 

`
env = GridWorld(num, height, width, centralized, noisy, rho, phi, sparse, alpha, beliefvectors)
`
 - `num`: number of agents in the grid
 - `height` : height of the grid (number of cells)
 - `width` : height of the grid (number of cells)
 - `centralized` : type of algorithm performed as included in the paper
    * `centralized = 0` : Algorithm 1, Centralized policy evaluation under POMDPs
    * `centralized = 1` : Algorithm 2, Diffusion policy evaluation under POMDPs
    * `centralized = 2` : Algorithm 3, Centralized evaluation for decentralized execution
 - `noisy`: level of noisiness of the observations performed by the agents
    * `noisy = 0`: observations of low noisiness level
    * `noisy = 1`: observations of mid-level noisiness
    * `noisy = 2`: observations of high noisiness
 - `sparse`: sparsity of the network (boolean)
 - `alpha` : learning rate

### Reset
To reset the environment while maintaining its aforementioned properties, use the  `reset(centralized)` method from the Gridworld class.

### Step
Use `step(centralized)` method from the Gridworld class to get the next time step in the environment. The following actions are generally taken during each step: (These steps vary depending on the type of algorithm)
- Observe 
- Adapt
- Action
- Reward  
- Evolve
- TD-Error (Temporal Difference)
- Target transition
- Network Agreement Error 
- SBE (Squared Bellman Error)

### Render
Use the `Render()` method from the Gridworld class to visualize the actions of the agents and the transition of the target in the environment. As rendered below, the agents are represented by sensors. These sensors aim to localize a spy drone as it moves around the grid.  Actions taken by the sensors are represented as noisy signals that aim to disrupt the communication between the intruder drone and its owner.

<p align="center">
<img src = "https://user-images.githubusercontent.com/80005419/217651585-c2e323b3-33f1-410e-8242-5fdab5d91e41.jpg" width="500" height="500">
</p> 


## Paper citation


If you used this environment for your experiments or found it helpful, consider citing the following paper:
 
<pre>
 @ARTICLE{kayaalp2023_policy,
  author={Kayaalp, Mert and Ghadieh, Fatima and Sayed, Ali H.},
  journal={IEEE Open Journal of Control Systems}, 
  title={Policy Evaluation in Decentralized POMDPs With Belief Sharing}, 
  year={2023},
  volume={2},
  number={},
  pages={125-145},
  doi={10.1109/OJCSYS.2023.3277760}}
</pre>

