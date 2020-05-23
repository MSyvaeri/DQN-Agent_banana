#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import random
from collections import namedtuple, deque

from model import Network

import torch
import torch.nn.functional as F
import torch.optim as optim
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3             # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = Network(state_size, action_size, seed).to(device)
        self.qnetwork_target = Network(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        
        
        
        
    
    
    def step(self,states, action, reward, next_states, done):
        ## Implementation of collecting the data for the replay memory. 
        ## It is important to regocnize that we have to compute the TD-error because this enables us to perform a priorized replay
        
        
        next_state_1=torch.from_numpy(np.vstack([next_states])).float().to(device)
        state_1=torch.from_numpy(np.vstack([states])).float().to(device)
        action_1 = torch.from_numpy(np.vstack([action])).long().to(device)
        Q_next= self.qnetwork_target(next_state_1).detach().max(1)[0].unsqueeze(1)
        Q_targets = reward + (GAMMA * Q_next * (1 - done))
        Q_state = self.qnetwork_local(state_1).gather(1, action_1)
        prob=F.mse_loss(Q_state,Q_targets).data.cpu().numpy().flatten()[0] 
        
        
        
        self.memory.add(states, action, reward, next_states, done,prob)
        
        
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.get_loss(experiences, GAMMA)
        
        
        
    def find_action(self,state,eps):
        
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
        
    
    def get_loss(self, experiences, gamma):
        ## Implementation of the Double DQN
        
        states, actions, rewards, next_states, dones, samples, probabilities = experiences
        Q_max_action = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
        Q_next = self.qnetwork_target(next_states).detach().gather(1, Q_max_action)

        Q_targets = rewards + (gamma * Q_next * (1 - dones))
        Q_state = self.qnetwork_local(states).gather(1, actions)
        
        
        loss = (torch.FloatTensor(probabilities) * F.mse_loss(Q_state, Q_targets)).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_q_network(self.qnetwork_local, self.qnetwork_target, TAU)  
      
        
        Q_next= self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards+ (gamma * Q_next * (1 - dones))

        Q_state = self.qnetwork_local(states).gather(1, actions)
        loss= ((Q_state-Q_targets)**2).data.cpu().numpy().flatten()

        self.memory.update(loss,samples)
        
    def update_q_network(self,model, target_model, tau):
        for target_param, model in zip(target_model.parameters(), model.parameters()):
            target_param.data.copy_(tau*model.data + (1.0-tau)*target_param.data) 
            
        
class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size,seed):
        
        self.action_size = action_size
        
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.probability = deque(maxlen=buffer_size)    #### memory for the priorized experience replay
        self.alpha=0.1
        self.epsilon=0.01
        self.beta=0.6
        self.beta_change=0.01
        
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        
    def add(self,state, action, reward, next_state, done, prob):
        prob=prob+self.epsilon
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        
        self.probability.append(prob)
        
    def update(self,loss,sample):
        for i,j in enumerate(sample):
            self.probability[j]=loss[i]+self.epsilon
  
    
    def sample(self):
        probabilities=np.array([self.probability])[0]
        self.beta=min(1,self.beta+self.beta_change)
        
        probabilities=np.power(probabilities,self.alpha)
        probabilities=probabilities/np.sum(probabilities)

        weight_is = np.power((len(self.memory)*probabilities),-self.beta)
        weight_is = weight_is / np.max(weight_is)
        
        index =np.random.choice(np.arange(len(self.memory)),size=self.batch_size,p=probabilities)
        
        states  = torch.from_numpy(np.vstack([self.memory[i].state for i in index if i is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([self.memory[i].action for i in index if i is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([self.memory[i].reward for i in index if i is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([self.memory[i].next_state for i in index if i is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([self.memory[i].done for i in index if i is not None]).astype( np.uint8)).float( ).to(device)
        
        weights=np.vstack([weight_is[i] for i in index if i is not None])
      
        
        return (states, actions, rewards, next_states, dones,index,weights)


    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    


