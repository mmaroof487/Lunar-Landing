import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
from collections import deque, namedtuple

class Network(nn.Module):
        def __init__(self, state_size, action_size, seed=42):  #statesize=8 no of values that define agent, agentsize = 4
                super(Network,self).__init__()
                self.seed = torch.manual_seed(seed)
                self.fc1 = nn.Linear(state_size, 64) #input, 1st layer
                self.fc2 = nn.Linear(64, 64) #1st , 2nd layer
                self.fc3 = nn.Linear(64, action_size) #2nd layer, output

        def forward(self, state):
                x = self.fc1(state)
                x = F.relu(x)
                x = self.fc2(x)
                x = F.relu(x)
                return self.fc3(x)
