#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F


#### Implemenation of a Duelling architecture

class Network(nn.Module):

    def __init__(self, state_size, action_size, seed, size_1=64, size_2=64,size_3=64):
        
        super(Network, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.action_size=action_size
        self.dense_1 = nn.Linear(state_size, size_1)
        self.dense_2 = nn.Linear(size_1, size_2)
        
        self.dense_a_1 = nn.Linear(size_2, size_3)
        self.dense_a_2 = nn.Linear(size_3, action_size)
        


        self.dense_v_1 = nn.Linear(size_2, size_3)
        self.dense_v_2 = nn.Linear(size_3, 1)

    def forward(self, state):
        x=self.dense_1(state)
        x=F.relu(x)
        x=self.dense_2(x)
        x=F.relu(x)
        
        x_a=self.dense_a_1(x)
        x_a=F.relu(x_a)
       
        x_a=self.dense_a_2(x_a)
        
        x_v=self.dense_v_1(x)
        x_v=F.relu(x_v)
        x_v=self.dense_v_2(x_v)
        x_v=x_v.expand(x.size(0), self.action_size)
        
        
        
        
        
        x = x_v + x_a - x_a.mean(1).unsqueeze(1).expand(x.size(0),self.action_size)
        return x


# In[ ]:




