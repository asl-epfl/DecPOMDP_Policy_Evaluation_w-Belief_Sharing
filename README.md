# Policy Evaluation in Decentralized POMDPs with Belief Sharing.

Multi-agent grid world environment with partial observability and a discrete action space. In this environment, agents (e.g., satellites) exploit their communication network in order to track a moving target (e.g., intruder dine). The target moves randomly according to a pre-defined transition model that takes the actions (i.e., hits) of agents into account. 

Used in the paper [Policy Evaluation in Decentralized
POMDPs with Belief Sharing](https://arxiv.org/).

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
- TD-Error 
- Target transition
- Network Agreement Error 
- SBE Error

### Render
Use the `Render()` method from the Gridworld class to visualize the actions of the agents and the transition of the target in the environment. As rendered below, the agents are represented by sensors (satellites). These satellites aim to localize a spy drone as it transitions around the grid.  Actions taken by the satellites are represented as noisy signals that aim to disrupt the communication between the intruder and its owner.

<p align="center">
<img src = "https://user-images.githubusercontent.com/80005419/216824175-30056460-fd53-40a7-8e33-7d8b800c1631.jpg" width="500" height="500">
</p>

## Paper citation

If you used this environment for your experiments or found it helpful, consider citing the following paper:
 
<pre>
@article{,
  title={Policy Evaluation in Decentralized POMDPs with Belief Sharing},
  author={M. Kayaalp, F. Ghadieh, and A. H. Sayed},
  journal={},
  year={2023}
}
</pre>

