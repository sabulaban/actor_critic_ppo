# UAV Stabilization using Actor/Critic Style PPO
The repo consists of mainly 3 notebooks as per the following:

### 1- simple_agent.ipynb:
This is an implementation of random policy search agent. The notebook presents the learning results for the policy search agent, and uses the following classes/libraries:

* Numpy
* Env class as define in env.py, defines the environment variables and init position of the UAV
* Task class as defined in taskexp.py, it interfaces with the agents, and provides the step, reset, and reward functions, task interfaces with physics_sim_up.py to take steps in space, and get state space output.
* Sys
* Pandas
* matplotlib.pyplot 
* Axes3D
* policysearch - Definition of the actual random search agent, where env.py and taskexp.py are called.

### 2- ppo-exp-serial.ipynb:
This is an implementation of PPO actor/critic for hyper parameters optimization, the notebook uses mainly:

* Task class
* Env class
* Agent class - Where the actor/critic is agent is defined

The implementation tries 81 combinations for 4 hyper parameters, mainly:

  * Batch Size
  * K
  * Gamma
  * Actor Standard Deviation Factor

Each agent out of the 81 scenarios is trained for 5000 epochs, each epoch will run K training runs for actor and critic.

Actor and Critic implementation is defined in actor.py and critic.py.

At every 100 epochs per agent, a test run is performed using play_game() method, and trajectories are collected using the get_trajs() method.

### 3- ppo-optimum-agent.ipynb:
This is an implementation of PPO actor/critic optimum agent based on ppo-exp-serial.ipynb analysis, the notebook uses mainly:

* Task class
* Env class
* agent_opt class - Where the actor/critic is agent is defined, and additional visualization for position and velocity.

The agent goes through 5000 ephocs, each epoch will run K (5 in this case) training runs for actor and critic.




 


