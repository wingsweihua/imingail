import math
import torch
from torch.distributions import Normal, Beta
import numpy as np

def get_action(dist_args, args):

    if args.stat_policy == "Gaussian":
        action = torch.normal(dist_args[0], dist_args[1])
    elif args.stat_policy == "Beta":
        action = Beta(dist_args[0], dist_args[1]).sample()
    action = action.data.numpy()
    # action = np.maximum(action, 0)
    # action = action + 5
    return [np.array(_) for _ in action]

def get_entropy(dist_args, args):
    if args.stat_policy == "Gaussian":
        dist = Normal(dist_args[0], dist_args[1])
    elif args.stat_policy == "Beta":
        dist = Beta(dist_args[0], dist_args[1])
    entropy = dist.entropy().mean()
    return entropy

def log_prob_density(x, dist_args, args):
    if args.stat_policy == "Gaussian":
        log_prob_density = -(x - dist_args[0]).pow(2) / (2 * dist_args[1].pow(2)) \
                         - 0.5 * math.log(2 * math.pi)
    elif args.stat_policy == "Beta":
        log_prob_density = Beta(dist_args[0], dist_args[1]).log_prob(x)
    return log_prob_density.sum(1, keepdim=True)

def get_reward(discrim, state, action):
    state = torch.Tensor(state)
    action = torch.Tensor(action)
    state_action = torch.cat([state, action], 1)
    with torch.no_grad():
        a = discrim(state_action)
        return [_ for _ in -torch.log(a).data.numpy()[:,0]]


def get_reward_interpolate(discrim, state, action):
    state = torch.Tensor(state)
    delta_time = torch.Tensor([[0.0]]*len(action))
    action = torch.Tensor(action)
    state_action = torch.cat([state, action], 1)
    state_action_input = torch.cat([state_action, state_action, delta_time], 1)
    with torch.no_grad():
        a, _ = discrim(state_action_input)
        return [_ for _ in -torch.log(a).data.numpy()[:,0]]



def save_checkpoint(state, filename):
    torch.save(state, filename)