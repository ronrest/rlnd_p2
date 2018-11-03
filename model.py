import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=32, hidden_sizes=[256, 128],leak=0.01, seed=42):
        """ Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_size (int): Number of nodes in hidden layers
            leak: amount of leakiness in leaky relu
        """
        super(Actor, self).__init__()
        self.leak = leak
        self.seed = torch.manual_seed(seed)
        self.bn0 = nn.BatchNorm1d(state_size)

        self.fc1 = nn.Linear(state_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], action_size)
        self.initialize_weights()

    def initialize_weights(self):
        """ Initilaize the weights using He et al (2015) weights """
        torch.nn.init.kaiming_normal_(self.fc1.weight.data, a=self.leak, mode='fan_in')
        torch.nn.init.kaiming_normal_(self.fc2.weight.data, a=self.leak, mode='fan_in')
        torch.nn.init.uniform_(self.fc3.weight.data, -3e-3, 3e-3)

#     def initialize_weights(self):
#         """ Initilaize the weights using He et al (2015) weights """
#         torch.nn.init.kaiming_uniform_(self.fc1.weight.data, a=self.leak, mode='fan_in')
#         torch.nn.init.kaiming_uniform_(self.fc2.weight.data, a=self.leak, mode='fan_in')
#         torch.nn.init.uniform_(self.fc3.weight.data, -1e-3, 1e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        state = self.bn0(state)
        x = F.leaky_relu(self.fc1(state), negative_slope=self.leak)
        x = F.leaky_relu(self.fc2(x), negative_slope=self.leak)
        x =  torch.tanh(self.fc3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_sizes=[256, 128,128], leak=0.01, seed=42):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
            hidden_size:
        """
        super(Critic, self).__init__()
        self.leak = leak
        self.seed = torch.manual_seed(seed)
        self.bn0 = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0]+action_size, hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fc4 = nn.Linear(hidden_sizes[2], 1)
        self.initialize_weights()

#     def initialize_weights(self):
#         self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
#         self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
#         self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def initialize_weights(self):
        """ Initilaize the weights using He et al (2015) weights """
        torch.nn.init.kaiming_normal_(self.fc1.weight.data, a=self.leak, mode='fan_in')
        torch.nn.init.kaiming_normal_(self.fc2.weight.data, a=self.leak, mode='fan_in')
        torch.nn.init.uniform_(self.fc3.weight.data, -3e-3, 3e-3)

#     def initialize_weights(self):
#         """ Initilaize the weights using He et al (2015) weights """
#         torch.nn.init.kaiming_uniform_(self.fc1.weight.data, a=self.leak, mode='fan_in')
#         torch.nn.init.kaiming_uniform_(self.fc2.weight.data, a=self.leak, mode='fan_in')
#         torch.nn.init.uniform_(self.fc3.weight.data, -3e-3, 3e-3)

    def forward(self, state, action):
        """ Build a critic (value) network that maps (state, action) pairs -> Q-values.
            is action a vector of actions? or probability distribution of actions?
            or a single action class that was chosen?
        """
        state = self.bn0(state)
        x = F.leaky_relu(self.fc1(state), negative_slope=self.leak)
        x = torch.cat((x, action), dim=1)
        x = F.leaky_relu(self.fc2(x), negative_slope=self.leak)
        x = F.leaky_relu(self.fc3(x), negative_slope=self.leak)
        x =  self.fc4(x)
        return x
