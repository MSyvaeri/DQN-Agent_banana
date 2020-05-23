Project: Navigation

Introduction:

In this repository, we solve the environment “banana” with a DQN-agent. The environment is a quadratic world, where the agent has to navigate and collect bananas. For yellow bananas he gets a reward of +1, for blue bananas a reward of -1. Thus, the goal is to collect as many yellow bananas while avoiding the blue ones. 
The environment is solved if the agent reaches an average reward of +13 over consecutive 100 attempts.

The state space has 37 dimensions and contains the speed of the agent, position of the next bananas etc. With these information the agent has to choose one of four following actions: Going forward, going left, going right or going backwards. 

The agent can be trained by running the jupyter notebook “Run.ipynb”, while the information about the agent are stored in the file “dqn_agent.py”. The neural network is stored in the file “model.py”, while the weights of a trained network are in the file “model.pt”.
