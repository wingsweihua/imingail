import os
import gym
import pickle
import argparse
import numpy as np
from collections import deque
import gym_citycar
import pandas as pd
import sys
import datetime
import torch.nn as nn
from shutil import copy

import torch
import torch.optim as optim
from tensorboardX import SummaryWriter 

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
print(os.getcwd())
sys.path.append('.')
from utils.log_utils import Stack, Memory, convert_trajs_to_array, convert_trajs_to_array_interpolate
from utils.check_utils import check_action_equals_next_speed, filter_unreal_speed

from imitation.gail.cal_utils.utils import *
from imitation.gail.cal_utils.zfilter import ZFilter
from imitation.gail.model import Actor, Critic, Discriminator, DiscriminatorInterpolcation
from imitation.gail.train_model import train_actor_critic, train_discrim, train_discrim_pretrain
from imitation.gail.train_model import train_discrim_interpol, train_discrim_interpol_pretrain

parser = argparse.ArgumentParser(description='PyTorch GAIL')
parser.add_argument('--gamma', type=float, default=0,
                    help='discounted factor (default: 0.99)')
parser.add_argument('--lamda', type=float, default=0.98, 
                    help='GAE hyper-parameter (default: 0.98)')
parser.add_argument('--hidden_size', type=int, default=32,
                    help='hidden unit size of actor, critic and discrim networks (default: 100)')
parser.add_argument('--hidden_size_state', type=int, default=32,
                    help='hidden unit size of interpolate networks (default: 100)')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='learning rate of models (default: 3e-4)')
parser.add_argument('--l2_rate', type=float, default=1e-3, 
                    help='l2 regularizer coefficient (default: 1e-3)')
parser.add_argument('--clip_param', type=float, default=0.2, 
                    help='clipping parameter for PPO (default: 0.2)')
parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size to update (default: 64)')
parser.add_argument('--suspend_accu_exp', type=float, default=0.9,
                    help='accuracy for suspending discriminator about expert data (default: 0.8)')
parser.add_argument('--suspend_accu_gen', type=float, default=0.9,
                    help='accuracy for suspending discriminator about generated data (default: 0.8)')

parser.add_argument('--max_iter_num', type=int, default=30,
                    help='maximal number of main iterations (default: 50)')
parser.add_argument('--max_episode_len', type=int, default=500,
                    help='maximal length of episodes (default: 1000)')

parser.add_argument('--discrim_update_num_pretrain', type=int, default=500,
                    help='update number of discriminator for pre-train (default: 500)')
parser.add_argument('--discrim_pretrain_iter', type=int, default=1,
                    help='number of iterations of pretrain for discriminator (default: 1)')
parser.add_argument('--discrim_update_num', type=int, default=1,
                    help='update number of discriminator (default: 2)')
parser.add_argument('--actor_critic_update_num', type=int, default=20,
                    help='update number of actor-critic (default: 10)')
parser.add_argument('--refresh_sample_size', type=int, default=50,
                    help='total sample size to collect before PPO update (default: 2048)')
parser.add_argument('--on_policy', action="store_true")
parser.add_argument('--average_running_returns', action="store_true")
parser.add_argument('--on_policy_start_iter', type=int, default=30)


parser.add_argument('--env_name', type=str, default="gym_citycar-v0",
                    help='name of the environment to run')
parser.add_argument('--load_model', type=str, default=None,
                    help='path to load the saved model')
parser.add_argument('--render', action="store_true", default=False,
                    help='if you dont want to render, set this to False')
parser.add_argument('--seed', type=int, default=500,
                    help='random seed (default: 500)')
parser.add_argument('--logdir', type=str, default='logs',
                    help='tensorboardx logs directory')
parser.add_argument('--stat_policy', type=str, default='Beta')
parser.add_argument('--path_to_sim_conf', type=str, default="config/simulator/default.json")
parser.add_argument('--memo', type=str, default="default_memo")
parser.add_argument('--scenario', type=str, default="default")
parser.add_argument('--model_name', type=str, default="GAIL")
parser.add_argument('--reward_func', type=int, default=1)
parser.add_argument('--interpolated', type=str, default="sparse")
parser.add_argument('--sample_rate', type=float, default=0.2,
                    help='sample rate from expert data (default: 0.8)')

args = parser.parse_args()


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def main():
    print(args)
    # ============== initial output ==================
    run_ts = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '_' + args.scenario
    path_to_output = os.path.join("data", "output", args.memo, args.model_name, run_ts)
    if not os.path.exists(path_to_output):
        os.makedirs(path_to_output)
    path_to_save_model = os.path.join("saved_model", args.memo, args.model_name, run_ts)
    if not os.path.exists(path_to_save_model):
        os.makedirs(path_to_save_model)

    # ============== copy the conf files ==================
    path_to_expert_traj = os.path.join("data", "expert_trajs", args.memo, args.scenario)
    sim_conf_file = os.path.join(path_to_expert_traj, "{}.json".format(args.scenario))
    copy(sim_conf_file, path_to_output)

    # ============== initial env =====================
    env = gym.make(args.env_name, normalize=True, path_to_conf_file=sim_conf_file,
                   list_vars_to_subscribe=["interval", "speed", "pos_in_lane", "lane_max_speed", "if_exit_lane", "dist_to_signal", "phase", "if_leader", "leader_speed", "dist_to_leader",],
                   reward_function=args.reward_func)
    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]
    obs_header = env.observation_header
    action_header = env.action_header

    print('state size:', num_inputs)
    print('action size:', num_actions)

    # ============== initial network =====================
    torch.manual_seed(args.seed)

    if args.stat_policy == "Gaussian":
        actor = Actor(num_inputs, 2*num_actions, args)
    elif args.stat_policy == "Beta":
        actor = Actor(num_inputs, 2*num_actions, args)
    actor.apply(init_weights)
    critic = Critic(num_inputs, args)
    critic.apply(init_weights)
    actor_optim = optim.Adam(actor.parameters(), lr=args.learning_rate)
    critic_optim = optim.Adam(critic.parameters(), lr=args.learning_rate,
                              weight_decay=args.l2_rate)

    if args.interpolated == "interpolated":
        discrim = DiscriminatorInterpolcation((num_inputs + num_actions)*2 + 1, num_inputs, args)
        discrim.apply(init_weights)
        discrim_optim = optim.Adam(discrim.parameters(), lr=args.learning_rate)
    elif args.interpolated == "sparse":
        discrim = Discriminator(num_inputs + num_actions, args)
        discrim.apply(init_weights)
        discrim_optim = optim.Adam(discrim.parameters(), lr=args.learning_rate)
    else:
        discrim = Discriminator(num_inputs + num_actions, args)
        discrim.apply(init_weights)
        discrim_optim = optim.Adam(discrim.parameters(), lr=args.learning_rate)

    # ============== load demonstrations =====================
    if args.interpolated == "interpolated_pretrain":
        expert_demo = pickle.load(open((os.path.join(path_to_expert_traj, "traj_interpolated_gail.pkl")), "rb"))
        demonstrations = convert_trajs_to_array(expert_demo, obs_header, action_header)
        print("demonstrations.shape", demonstrations.shape)
    elif args.interpolated == "interpolated":
        expert_demo = pickle.load(open((os.path.join(path_to_expert_traj, "traj_sparse.pkl")), "rb"))
        demonstrations = convert_trajs_to_array_interpolate(expert_demo, obs_header, action_header)
        print("demonstrations.shape", demonstrations[0].shape)
        sample_indexes = np.random.randint(demonstrations[0].shape[0],
                                           size=int(demonstrations[0].shape[0] * args.sample_rate))

        demonstrations = [demonstrations[0].iloc[sample_indexes, :], demonstrations[1].iloc[sample_indexes, :]]
        print("Sample_rate: ", args.sample_rate)
        print("demonstrations.shape", demonstrations[0].shape)
    elif args.interpolated == "sparse":
        expert_demo = pickle.load(open((os.path.join(path_to_expert_traj, "traj_sparse.pkl")), "rb"))
        demonstrations = convert_trajs_to_array(expert_demo, obs_header, action_header)
        # todo add this into config for sample rate
        demonstrations = demonstrations[np.random.randint(demonstrations.shape[0],
                                                         size=int(demonstrations.shape[0]*args.sample_rate)), :]
        print("Sample_rate: ", args.sample_rate)
        print("demonstrations.shape", demonstrations.shape)
    else:
        expert_demo = pickle.load(open((os.path.join(path_to_expert_traj, "traj_sample.pkl")), "rb"))
        demonstrations = convert_trajs_to_array(expert_demo, obs_header, action_header)
        print("demonstrations.shape", demonstrations.shape)

    train_discrim_flag = True
    on_policy_update = False

    list_discrim_loss = []
    list_policy_loss = []
    list_learner_acc = []
    list_expert_acc = []

    for iter in range(args.max_iter_num):

        if args.on_policy:
            if iter >= args.on_policy_start_iter:
                on_policy_update = True

        list_traj = Stack(obs_header, action_header)

        n_obs, n_info = env.reset()
        n_action = []  # default value

        cnt_update = 0

        actor.eval(), critic.eval(), discrim.eval()
        memory = deque()
        list_vec_id = []
        list_ts = []

        for t in range(args.max_episode_len):

            if t % 100 == 0:
                print("Iter: {0}, t: {1}".format(iter+1, t))

            if len(n_obs) == 0:

                next_n_obs, n_reward, n_done, next_n_info = env.step(n_action, n_info)

            else:
                # follow policy to generate samples

                dist_args = actor(torch.Tensor(n_obs))
                n_action = get_action(dist_args, args)

                if on_policy_update:
                    n_action = filter_unreal_speed(n_action, n_info["next_speed_est"])
                    next_n_obs, n_reward, n_done, next_n_info = env.step(n_action, n_info)
                else:
                    n_action_excute = [np.array([a]) for a in n_info["next_speed_est"]]
                    next_n_obs, n_reward, n_done, next_n_info = env.step(n_action_excute, n_info)
                if args.interpolated == "interpolated":
                    n_irl_reward = get_reward_interpolate(discrim, n_obs, n_action)
                else:
                    n_irl_reward = get_reward(discrim, n_obs, n_action)
                n_mask = [1-int(d) for d in n_done]

                # if on_policy_update:
                #     assert check_action_equals_next_speed(next_n_obs, next_n_info, n_action, n_info, obs_header)
                # else:
                #     assert check_action_equals_next_speed(next_n_obs, next_n_info, n_action_excute, n_info, obs_header)
                list_traj.append(
                    [n_info["vec_id"], n_info["current_time"], n_info["lane_id"], n_obs, n_action, n_reward,
                     n_irl_reward])
                assert len(n_obs) == len(n_action), print("obs: {0}, act: {1}".format(len(n_obs), len(n_action)))
                assert len(n_obs) == len(n_irl_reward), print("obs: {0}, r: {1}".format(len(n_obs), len(n_irl_reward)))
                assert len(n_obs) == len(n_mask), print("obs: {0}, mask: {1}".format(len(n_obs), len(n_mask)))
                for ind_sample in range(len(n_obs)):
                    memory.append([n_obs[ind_sample], n_action[ind_sample], n_irl_reward[ind_sample], n_mask[ind_sample]])
                    list_vec_id.append(n_info["vec_id"][ind_sample])
                    list_ts.append(n_info["current_time"][ind_sample])

            n_obs = next_n_obs
            n_info = next_n_info

            if iter >= args.discrim_pretrain_iter:

                train_discrim_flag = False

                if t % args.refresh_sample_size == 0 and t != 0:

                    # sort the memory by vec_id and time

                    list_ts, list_vec_id, memory = (list(t) for t in zip(*sorted(zip(list_ts, list_vec_id, memory))))
                    list_vec_id, list_ts, memory = (list(t) for t in zip(*sorted(zip(list_vec_id, list_ts, memory))))
                    memory = deque(memory)

                    for ind_sample in range(len(list_vec_id)):
                        if ind_sample == 0:
                            cur_vec_id = list_vec_id[ind_sample]
                        elif ind_sample == len(list_vec_id) - 1:
                            memory[ind_sample][3] = 0
                        else:
                            if list_vec_id[ind_sample] != cur_vec_id:
                                memory[ind_sample-1][3] = 0
                            cur_vec_id = list_vec_id[ind_sample]

                    actor.train(), critic.train(), discrim.train()
                    if train_discrim_flag:
                        if args.interpolated == "interpolated":
                            expert_acc, learner_acc, discrim_loss = train_discrim_interpol(discrim, memory,
                                                                                           discrim_optim,
                                                                                           demonstrations,
                                                                                           args)
                        else:
                            expert_acc, learner_acc, discrim_loss = train_discrim(discrim, memory, discrim_optim,
                                                                                  demonstrations, args)
                        print("Expert: %.2f%% | Learner: %.2f%%" % (expert_acc * 100, learner_acc * 100))
                        if expert_acc > args.suspend_accu_exp and learner_acc > args.suspend_accu_gen:
                            train_discrim_flag = False
                        for ind_epoch in range(args.discrim_update_num):
                            list_discrim_loss.append([iter, cnt_update, ind_epoch, discrim_loss[ind_epoch]])

                    policy_loss = train_actor_critic(actor, critic, memory, actor_optim, critic_optim, args)
                    for ind_epoch in range(args.actor_critic_update_num):
                        list_policy_loss.append([iter, cnt_update, ind_epoch, policy_loss[ind_epoch]])

                    list_expert_acc.append([iter, cnt_update, expert_acc])
                    list_learner_acc.append([iter, cnt_update, learner_acc])

                    print("one training finished")

                    cnt_update += 1

                    actor.eval(), critic.eval(), discrim.eval()
                    memory = deque()
                    list_vec_id = []
                    list_ts = []



        if iter < args.discrim_pretrain_iter:
            actor.train(), critic.train(), discrim.train()
            if args.interpolated == "interpolated":
                expert_acc, learner_acc, discrim_loss = train_discrim_interpol_pretrain(discrim, memory, discrim_optim,
                                                                               demonstrations, args)
            else:
                expert_acc, learner_acc, discrim_loss = train_discrim_pretrain(discrim, memory, discrim_optim,
                                                                               demonstrations, args)

            print("Expert: %.2f%% | Learner: %.2f%%" % (expert_acc * 100, learner_acc * 100))
            for ind_epoch in range(args.discrim_update_num_pretrain):
                list_discrim_loss.append([iter, cnt_update, ind_epoch, discrim_loss[ind_epoch]])
            actor.eval(), critic.eval(), discrim.eval()
            memory = deque()

        # ============== post process and dump data =======

        # full trajectory -- for comparison: dataframe
        df = list_traj.convert_to_df()
        pickle.dump(df, open(os.path.join(path_to_output, "traj_raw_iter{}.pkl".format(iter)), "wb"))
        df.to_csv(os.path.join(path_to_output, "traj_raw_iter{}.csv".format(iter)))
        print("================================================================")
        print("********* full trajectory logged")
        print("================================================================")

        ckpt_path = os.path.join(path_to_save_model, 'ckpt_iter{0}.pth.tar'.format(iter))

        torch.save({
            'actor': actor,
            'critic': critic,
            'discrim': discrim,
            'args': args
        }, ckpt_path)


    df_discrim_loss = pd.DataFrame(np.array(list_discrim_loss), columns=["iter", "update", "epoch", "loss"])
    df_discrim_loss.to_csv(os.path.join(path_to_output, "discrim_loss.csv"))

    df_policy_loss = pd.DataFrame(np.array(list_policy_loss), columns=["iter", "update", "epoch", "loss"])
    df_policy_loss.to_csv(os.path.join(path_to_output, "policy_loss.csv"))

    df_expert_acc = pd.DataFrame(np.array(list_expert_acc), columns=["iter", "update", "acc"])
    df_expert_acc.to_csv(os.path.join(path_to_output, "expert_acc.csv"))

    df_learner_acc = pd.DataFrame(np.array(list_learner_acc), columns=["iter", "update", "acc"])
    df_learner_acc.to_csv(os.path.join(path_to_output, "learner_acc.csv"))


        # score_avg = np.mean(scores)
        # print('{}:: {} episode score is {:.2f}'.format(iter, episodes, score_avg))
        # writer.add_scalar('log/score', float(score_avg), iter)


if __name__=="__main__":
    main()