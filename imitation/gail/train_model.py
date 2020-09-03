import torch
import numpy as np
from imitation.gail.cal_utils.utils import get_entropy, log_prob_density


def train_discrim_interpol_pretrain(discrim, memory, discrim_optim, demonstration_lists, args):
    # learner
    memory = np.array(memory)
    states = np.vstack(memory[:, 0])
    actions = list(memory[:, 1])
    delta_time = torch.Tensor([np.zeros_like(action) for action in actions])


    states = torch.Tensor(states)
    actions = torch.Tensor(actions)
    state_action = torch.cat([states, actions], dim=1)
    state_action_input = torch.cat([state_action, state_action, delta_time], dim=1)

    state_action_input = torch.Tensor(state_action_input)

    criterion = torch.nn.BCELoss()
    interpolation_creterion = torch.nn.MSELoss()

    list_discrim_loss = []
    list_interpol_loss = []
    list_discrim_interpola_loss = []

    for _ in range(args.discrim_update_num_pretrain):
        learner, state_interpolated_lrn = discrim(state_action_input)
        demonstrations = torch.Tensor(demonstration_lists[0].values)
        expert, state_interpolated_exp = discrim(demonstrations)

        discrim_loss = criterion(learner, torch.ones((states.shape[0], 1))) + \
                       criterion(expert, torch.zeros((demonstrations.shape[0], 1)))

        interpolation_loss = interpolation_creterion(state_action_input, state_interpolated_lrn)

        discrim_interpol_loss = discrim_loss + interpolation_loss

        discrim_optim.zero_grad()
        discrim_interpol_loss.backward()
        discrim_optim.step()

        list_discrim_loss.append(discrim_loss.tolist())
        list_interpol_loss.append(interpolation_loss.tolist())
        list_discrim_interpola_loss.append(discrim_interpol_loss.tolist())

    expert_acc = ((discrim(demonstrations)[0] < 0.5).float()).mean()
    learner_acc = ((discrim(state_action_input)[0] > 0.5).float()).mean()

    return expert_acc, learner_acc, np.array([list_discrim_loss,
                                              list_interpol_loss,
                                              list_discrim_interpola_loss]
                                             )[0]


def train_discrim_interpol(discrim, memory, discrim_optim, demonstration_lists, args):
    # learner
    memory = np.array(memory)
    states = np.vstack(memory[:, 0])
    actions = list(memory[:, 1])
    delta_time = torch.Tensor([np.zeros_like(action) for action in actions])


    states = torch.Tensor(states)
    actions = torch.Tensor(actions)
    state_action = torch.cat([states, actions], dim=1)
    state_action_input = torch.cat([state_action, state_action, delta_time], dim=1)

    state_action_input = torch.Tensor(state_action_input)

    criterion = torch.nn.BCELoss()
    interpolation_creterion = torch.nn.MSELoss()

    list_discrim_loss = []
    list_interpol_loss = []
    list_discrim_interpola_loss = []

    for _ in range(args.discrim_update_num_pretrain):
        learner, state_interpolated_lrn = discrim(state_action_input)
        demonstrations = torch.Tensor(demonstration_lists[0].values)
        expert, state_interpolated_exp = discrim(demonstrations)

        discrim_loss = criterion(learner, torch.ones((states.shape[0], 1))) + \
                       criterion(expert, torch.zeros((demonstrations.shape[0], 1)))

        interpolation_loss = interpolation_creterion(state_action_input, state_interpolated_lrn)

        discrim_interpol_loss = discrim_loss + interpolation_loss

        discrim_optim.zero_grad()
        discrim_interpol_loss.backward()
        discrim_optim.step()

        list_discrim_loss.append(discrim_loss.tolist())
        list_interpol_loss.append(interpolation_loss.tolist())
        list_discrim_interpola_loss.append(discrim_interpol_loss.tolist())

    expert_acc = ((discrim(demonstrations)[0] < 0.5).float()).mean()
    learner_acc = ((discrim(state_action_input)[0] > 0.5).float()).mean()

    return expert_acc, learner_acc, np.array([list_discrim_loss,
                                              list_interpol_loss,
                                              list_discrim_interpola_loss]
                                             )[0]

def train_discrim_pretrain(discrim, memory, discrim_optim, demonstrations, args):
    # learner
    memory = np.array(memory)
    states = np.vstack(memory[:, 0])
    actions = list(memory[:, 1])

    states = torch.Tensor(states)
    actions = torch.Tensor(actions)

    criterion = torch.nn.BCELoss()

    list_discrim_loss = []

    for _ in range(args.discrim_update_num_pretrain):
        learner = discrim(torch.cat([states, actions], dim=1))
        demonstrations = torch.Tensor(demonstrations)
        expert = discrim(demonstrations)

        discrim_loss = criterion(learner, torch.ones((states.shape[0], 1))) + \
                       criterion(expert, torch.zeros((demonstrations.shape[0], 1)))

        discrim_optim.zero_grad()
        discrim_loss.backward()
        discrim_optim.step()

        list_discrim_loss.append(discrim_loss.tolist())

    expert_acc = ((discrim(demonstrations) < 0.5).float()).mean()
    learner_acc = ((discrim(torch.cat([states, actions], dim=1)) > 0.5).float()).mean()

    return expert_acc, learner_acc, np.array(list_discrim_loss)


def train_discrim(discrim, memory, discrim_optim, demonstrations, args):
    memory = np.array(memory) 
    states = np.vstack(memory[:, 0]) 
    actions = list(memory[:, 1]) 

    states = torch.Tensor(states)
    actions = torch.Tensor(actions)
        
    criterion = torch.nn.BCELoss()

    list_discrim_loss = []

    for _ in range(args.discrim_update_num):
        learner = discrim(torch.cat([states, actions], dim=1))
        demonstrations = torch.Tensor(demonstrations)
        expert = discrim(demonstrations)

        discrim_loss = criterion(learner, torch.ones((states.shape[0], 1))) + \
                        criterion(expert, torch.zeros((demonstrations.shape[0], 1)))
                
        discrim_optim.zero_grad()
        discrim_loss.backward()
        discrim_optim.step()

        list_discrim_loss.append(discrim_loss.tolist())


    expert_acc = ((discrim(demonstrations) < 0.5).float()).mean()
    learner_acc = ((discrim(torch.cat([states, actions], dim=1)) > 0.5).float()).mean()

    return expert_acc, learner_acc, np.array(list_discrim_loss)


def train_actor_critic(actor, critic, memory, actor_optim, critic_optim, args):
    memory = np.array(memory) 
    states = np.vstack(memory[:, 0]) 
    actions = list(memory[:, 1]) 
    rewards = list(memory[:, 2]) 
    masks = list(memory[:, 3]) 

    old_values = critic(torch.Tensor(states))
    returns, advants = get_gae(rewards, masks, old_values, args)
    
    dist_args = actor(torch.Tensor(states))
    old_policy = log_prob_density(torch.Tensor(actions), dist_args, args)

    criterion = torch.nn.MSELoss()
    n = len(states)
    arr = np.arange(n)

    list_loss = []

    for _ in range(args.actor_critic_update_num):
        np.random.shuffle(arr)

        for i in range(n // args.batch_size): 
            batch_index = arr[args.batch_size * i : args.batch_size * (i + 1)]
            batch_index = torch.LongTensor(batch_index)
            
            inputs = torch.Tensor(states)[batch_index]
            actions_samples = torch.Tensor(actions)[batch_index]
            returns_samples = returns.unsqueeze(1)[batch_index]
            advants_samples = advants.unsqueeze(1)[batch_index]
            oldvalue_samples = old_values[batch_index].detach()
            
            values = critic(inputs)
            clipped_values = oldvalue_samples + \
                             torch.clamp(values - oldvalue_samples,
                                         -args.clip_param, 
                                         args.clip_param)
            critic_loss1 = criterion(clipped_values, returns_samples)
            critic_loss2 = criterion(values, returns_samples)
            critic_loss = torch.max(critic_loss1, critic_loss2).mean()

            loss, ratio, entropy = surrogate_loss(actor, advants_samples, inputs,
                                         old_policy.detach(), actions_samples,
                                         batch_index, args)
            clipped_ratio = torch.clamp(ratio,
                                        1.0 - args.clip_param,
                                        1.0 + args.clip_param)
            clipped_loss = clipped_ratio * advants_samples
            actor_loss = -torch.min(loss, clipped_loss).mean()

            loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

            critic_optim.zero_grad()
            loss.backward(retain_graph=True) 
            critic_optim.step()

            actor_optim.zero_grad()
            loss.backward()
            actor_optim.step()


        batch_index = arr
        batch_index = torch.LongTensor(batch_index)

        inputs = torch.Tensor(states)[batch_index]
        actions_samples = torch.Tensor(actions)[batch_index]
        returns_samples = returns.unsqueeze(1)[batch_index]
        advants_samples = advants.unsqueeze(1)[batch_index]
        oldvalue_samples = old_values[batch_index].detach()

        values = critic(inputs)
        clipped_values = oldvalue_samples + \
                         torch.clamp(values - oldvalue_samples,
                                     -args.clip_param,
                                     args.clip_param)
        critic_loss1 = criterion(clipped_values, returns_samples)
        critic_loss2 = criterion(values, returns_samples)
        critic_loss = torch.max(critic_loss1, critic_loss2).mean()

        loss, ratio, entropy = surrogate_loss(actor, advants_samples, inputs,
                                              old_policy.detach(), actions_samples,
                                              batch_index, args)
        clipped_ratio = torch.clamp(ratio,
                                    1.0 - args.clip_param,
                                    1.0 + args.clip_param)
        clipped_loss = clipped_ratio * advants_samples
        actor_loss = -torch.min(loss, clipped_loss).mean()

        loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

        list_loss.append(loss.tolist())

    return np.array(list_loss)


def get_gae(rewards, masks, values, args):
    rewards = torch.Tensor(rewards)
    masks = torch.Tensor(masks)
    returns = torch.zeros_like(rewards)
    advants = torch.zeros_like(rewards)
    
    running_returns = 0
    previous_value = 0
    running_advants = 0

    for t in reversed(range(0, len(rewards))):

        if masks[t] == 0:
            cnt_accu = 1
        else:
            cnt_accu += 1

        running_returns = rewards[t] + (args.gamma * running_returns * masks[t])

        if args.average_running_returns:
            returns[t] = running_returns/cnt_accu
        else:
            returns[t] = running_returns

        running_delta = rewards[t] + (args.gamma * previous_value * masks[t]) - \
                                        values.data[t]
        # td error
        previous_value = values.data[t]
        
        running_advants = running_delta + (args.gamma * args.lamda * \
                                            running_advants * masks[t])

        if args.average_running_returns:
            advants[t] = running_advants/cnt_accu
        else:
            advants[t] = running_advants

    advants = (advants - advants.mean()) / advants.std()
    return returns, advants

def surrogate_loss(actor, advants, states, old_policy, actions, batch_index, args):
    dist_args = actor(states)
    new_policy = log_prob_density(actions, dist_args, args)
    old_policy = old_policy[batch_index]

    ratio = torch.exp(new_policy - old_policy)
    surrogate_loss = ratio * advants
    entropy = get_entropy(dist_args, args)

    return surrogate_loss, ratio, entropy