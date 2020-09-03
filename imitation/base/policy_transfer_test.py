import os
import gym
import pickle
import json
import torch
from utils.log_utils import Stack
from utils.check_utils import check_action_equals_next_speed, filter_unreal_speed
from imitation.base import evaluate
import pandas as pd
import numpy as np
import datetime
from shutil import copy
from collections import deque
import torch.optim as optim
import torch.nn as nn

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def test_memo_scenario(args, get_action, get_reward, train_actor_critic, class_actor, class_critic, iter_num):
    # ============== check train output and model ================

    path_to_output_pre = os.path.join("data", "output", args.memo, args.model_name)
    if not os.path.exists(path_to_output_pre):
        print("not well trained")
        return
    else:
        list_output_dirs = os.listdir(path_to_output_pre)


    path_to_save_model_pre = os.path.join("saved_model", args.memo, args.model_name)
    if not os.path.exists(path_to_save_model_pre):
        print("not well trained")
        return
    else:
        list_model_dirs = os.listdir(path_to_save_model_pre)

    list_dirs = list(set(list_model_dirs).intersection(set(list_output_dirs)))

    for d in list_dirs:
        if d.endswith(args.scenario):
            if args.ts_only == "all":
                test_all_iters(path_to_output_pre, path_to_save_model_pre, d, args.scenario, get_action, get_reward,
                               train_actor_critic, class_actor, class_critic, args, iter_num)

            else:
                if d.startswith(args.ts_only):
                    test_all_iters(path_to_output_pre, path_to_save_model_pre, d, args.scenario, get_action, get_reward, train_actor_critic,  class_actor, class_critic, args, iter_num)

def test_all_iters(path_to_output_pre, path_to_save_model_pre, run_ts, scenario, get_action, get_reward, train_actor_critic,  class_actor, class_critic, args, iter_num):
    # ============== check train output and model ================

    list_model_files = os.listdir(os.path.join(path_to_save_model_pre, run_ts))

    test_one_iter(
        os.path.join(path_to_save_model_pre, run_ts),
        get_action,
        get_reward,
        train_actor_critic,
        class_actor,
        class_critic,
        args,
        iter_num
    )

    return


def test_one_iter(path_to_save_model, get_action, get_reward, train_actor_critic, class_actor, class_critic, args, iter_num="iter20"):

    # ============== initial output ==================
    run_ts = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '_' + args.scenario
    path_to_output = os.path.join("data", "output", args.memo + '_' + args.dynamics, args.model_name, run_ts+'_'+args.extra_memo)
    path_to_model_output = os.path.join("saved_model", args.memo + '_' + args.dynamics, args.model_name, run_ts+'_'+args.extra_memo)
    if not os.path.exists(path_to_output):
        os.makedirs(path_to_output)
    if not os.path.exists(path_to_model_output):
        os.makedirs(path_to_model_output)

    output_pkl = os.path.join(path_to_output, "traj_raw_{}.test.pkl")
    output_csv = os.path.join(path_to_output, "traj_raw_{}.test.csv")
    output_reward_pkl = os.path.join(path_to_output, "traj_reward_{}.test.pkl")
    output_reward_csv = os.path.join(path_to_output, "traj_reward_{}.test.csv")

    if os.path.exists(output_csv):
        print("already tested")
        return

    # ============== copy the conf files ==================
    path_to_expert_traj = os.path.join("data", "expert_trajs", args.memo, args.scenario)
    sim_conf_file = os.path.join(path_to_expert_traj, "{}.json".format(args.scenario))
    copy(sim_conf_file, path_to_output)
    dic_sim_conf = json.load(open(sim_conf_file, "r"))
    old_dir = dic_sim_conf["dir"]
    if old_dir[-1] == os.path.sep:
        head, tail = os.path.split(old_dir[:-1])
    else:
        head, tail = os.path.split(old_dir)
    dic_sim_conf["dir"] = os.path.join(head, args.scenario + '_' + args.dynamics) + os.path.sep
    json.dump(dic_sim_conf, open(sim_conf_file, "w"))

    # ============== initial env =====================
    env = gym.make(args.env_name,
                   normalize=True,
                   path_to_conf_file=sim_conf_file,
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

    ckpt = torch.load(os.path.join(path_to_save_model, "ckpt_{}.pth.tar".format(iter_num)))

    discrim = ckpt["discrim"]

    if args.retrain_policy:
        actor = class_actor(num_inputs, 2 * num_actions, args)
        actor.apply(init_weights)
        critic = class_critic(num_inputs, args)
        critic.apply(init_weights)
        print("training from new policy")
    else:
        actor = ckpt["actor"]
        critic = ckpt["critic"]
        print("using transfered policy")

    actor_optim = optim.Adam(actor.parameters(), lr=args.learning_rate)
    critic_optim = optim.Adam(critic.parameters(), lr=args.learning_rate,
                              weight_decay=args.l2_rate)

    print("Loaded OK ")


    train_discrim_flag = False

    list_policy_loss = []

    for iter in range(args.max_iter_num):

        list_traj = Stack(obs_header, action_header)
        list_reward = []

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

                n_action = filter_unreal_speed(n_action, n_info["next_speed_est"])
                next_n_obs, n_reward, n_done, next_n_info = env.step(n_action, n_info)

                if args.model_name == "GAIL":
                    n_irl_reward = get_reward(discrim, n_obs, n_action)
                elif args.model_name == "AIRL":
                    n_irl_reward = get_reward(discrim, actor, n_obs, n_action, args)
                n_mask = [1-int(d) for d in n_done]

                list_traj.append(
                    [n_info["vec_id"], n_info["current_time"], n_info["lane_id"], n_obs, n_action, n_reward,
                     n_irl_reward])
                list_reward += n_reward
                # assert len(n_obs) == len(n_action), print("obs: {0}, act: {1}".format(len(n_obs), len(n_action)))
                # assert len(n_obs) == len(n_irl_reward), print("obs: {0}, r: {1}".format(len(n_obs), len(n_irl_reward)))
                # assert len(n_obs) == len(n_mask), print("obs: {0}, mask: {1}".format(len(n_obs), len(n_mask)))
                for ind_sample in range(len(n_obs)):
                    memory.append([n_obs[ind_sample], n_action[ind_sample], n_irl_reward[ind_sample], n_mask[ind_sample]])
                    list_vec_id.append(n_info["vec_id"][ind_sample])
                    list_ts.append(n_info["current_time"][ind_sample])

            n_obs = next_n_obs
            n_info = next_n_info

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

                actor.train(), critic.train()

                policy_loss = train_actor_critic(actor, critic, memory, actor_optim, critic_optim, args)
                for ind_epoch in range(args.actor_critic_update_num):
                    list_policy_loss.append([iter, cnt_update, ind_epoch, policy_loss[ind_epoch]])

                print("one training finished")

                cnt_update += 1

                actor.eval(), critic.eval(), discrim.eval()
                memory = deque()
                list_vec_id = []
                list_ts = []


        # ============== post process and dump data =======

        ckpt_path = os.path.join(path_to_model_output, 'ckpt_iter{0}.pth.tar'.format(iter))

        torch.save({
            'actor': actor,
            'critic': critic,
            'discrim': discrim,
            'args': args
        }, ckpt_path)


        # df_reward = pd.DataFrame(np.array([list_reward]).transpose(), columns=["reward"])
        # pickle.dump(df_reward, open(output_reward_pkl.format(iter), "wb"))
        # df_reward.to_csv(output_reward_csv.format(iter))

        # full trajectory -- for comparison: dataframe
        df = list_traj.convert_to_df()
        pickle.dump(df, open(output_pkl.format("iter{}".format(iter)), "wb"))
        df.to_csv(output_csv.format("iter{}".format(iter)))
        print("================================================================")
        print("********* full trajectory logged")
        print("================================================================")

        dic_roadnet_len = json.load(open(os.path.join("data", "traffic", args.scenario, "roadnet_len.json"), "r"))

        evaluate.evaluate(output_pkl.format("iter{}".format(iter)), output_pkl.format("iter{}".format(iter)), dic_roadnet_len, args.max_episode_len)

    # pay attention to the length of the lane