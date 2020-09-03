import torch
import torch.nn as nn
import numpy as np

class Actor(nn.Module):
    def __init__(self, num_inputs, num_outputs, args):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_inputs, args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.fc3 = nn.Linear(args.hidden_size, num_outputs)
        
        # self.fc3.weight.data.mul_(0.1)
        # self.fc3.bias.data.mul_(0.0)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        out = torch.relu(self.fc3(x))
        dist_arg0 = out[:,0:1] + 0.1
        dist_arg1 = out[:,1:2] + 0.1
        # logstd = torch.zeros_like(mu) - 0.5
        # std = torch.exp(logstd)
        # std = torch.zeros_like(mu)
        return dist_arg0, dist_arg1


class Critic(nn.Module):
    def __init__(self, num_inputs, args):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_inputs, args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.fc3 = nn.Linear(args.hidden_size, 1)
        
        # self.fc3.weight.data.mul_(0.1)
        # self.fc3.bias.data.mul_(0.0)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        v = torch.relu(self.fc3(x))
        return v


class Discriminator(nn.Module):
    def __init__(self, num_inputs, args):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(num_inputs, args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.fc3 = nn.Linear(args.hidden_size, 1)
        
        # self.fc3.weight.data.mul_(0.1)
        # self.fc3.bias.data.mul_(0.0)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        prob = torch.sigmoid(self.fc3(x))
        return prob


class DiscriminatorInterpolcation(nn.Module):
    def __init__(self, state_num_inputs, state_outputs, args):
        super(DiscriminatorInterpolcation, self).__init__()
        self.fc1_state = nn.Linear(state_num_inputs, args.hidden_size_state)
        self.fc2_state = nn.Linear(args.hidden_size, args.hidden_size)
        self.fc3_state = nn.Linear(args.hidden_size, state_num_inputs)

        self.fc1 = nn.Linear(state_num_inputs, args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.fc3 = nn.Linear(args.hidden_size, 1)

        # self.fc3.weight.data.mul_(0.1)
        # self.fc3.bias.data.mul_(0.0)

    def forward(self, state_s):
        x_state = torch.relu(self.fc1_state(state_s))
        x_state = torch.relu(self.fc2_state(x_state))
        x_state = self.fc3_state(x_state)

        x = torch.relu(self.fc1(x_state))
        x = torch.relu(self.fc2(x))
        prob = torch.sigmoid(self.fc3(x))
        return prob, x_state

