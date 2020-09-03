from utils.log_utils import Stack
from utils.check_utils import check_action_equals_next_speed, filter_unreal_speed
import torch
import gym
import gym_citycar
import os
import pickle
import numpy as np


def gen_traj_with_policy(args, policy_model):

    '''
    Generate trajectory data with trained policy.
    :param args: global arguments
    :param policy_model: trained policy neural net
    :return: raw_traj_df, trimmed_traj_df
    '''
    # ============== initial env =====================

    path_to_sim_conf = os.path.join("config", "simulator", "{}.json".format(args.scenario))
    env = gym.make('gym_citycar-v0', normalize=True, path_to_conf_file=path_to_sim_conf,
                   list_vars_to_subscribe=["interval", "speed", "pos_in_lane", "lane_max_speed", "if_exit_lane",
                                           "dist_to_signal", "phase", "if_leader", "leader_speed", "dist_to_leader"],
                   reward_function=args.reward_function)

    obs_header = env.observation_header
    action_header = env.action_header

    # ============== initial storage =================
    # organize the trajs by vehicle
    list_traj = Stack(obs_header, action_header)

    # ============== start looping experiment ========
    for ind_exp in range(args.N_EXP):
        n_obs, n_info = env.reset()
        for t in range(args.max_episode_len):
            # follow policy to generate samples
            if n_obs == []:
                n_action = []
            else:
                n_obs_array = np.array(n_obs)
                n_obs_array = np.concatenate([n_obs_array[:, 1:2], n_obs_array[:, 7:10]], axis=1)
                n_obs_tensor = torch.from_numpy(n_obs_array.astype(np.float))
                n_action_array = policy_model(n_obs_tensor.float()).data.numpy()
                n_action = [a for a in n_action_array]

            n_action = filter_unreal_speed(n_action, n_info['next_speed_est'])
            next_n_obs, n_reward, n_done, next_n_info = env.step(n_action, n_info)
            assert check_action_equals_next_speed(next_n_obs, next_n_info, n_action, n_info, obs_header)

            n_irl_reward = [a for a in np.zeros(len(n_reward))]

            list_traj.append([n_info["vec_id"], n_info["current_time"], n_info["lane_id"], n_obs, n_action,
                              n_reward, n_irl_reward])

            n_obs = next_n_obs
            n_info = next_n_info

    # ============== post process and dump data =======

    # full trajectory -- for comparison: dataframe
    df = list_traj.convert_to_df()

    if not os.path.exists(os.path.join(args.data_dir, args.memo, args.scenario)):
        os.mkdir(os.path.join(args.data_dir, args.memo, args.scenario))
    pickle.dump(df, open(os.path.join(args.data_dir, args.memo, args.scenario, "traj_lrn_raw.pkl"), "wb"))

    # partial trajectory -- for learning: sample wise

    # remove non-continuous samples and end of each sample of each trajs

    df.sort_values(by=["vec_id", "ts"], axis=0, inplace=True)
    df.reset_index(drop=True, inplace=True)

    list_end_traj_ind = np.where(df["vec_id"][:len(df) - 1].values != df["vec_id"][1:len(df)].values)[0].tolist() + [
        len(df) - 1]
    list_discontinuous_ind = \
        np.where(df["ts"][:len(df) - 1].values + df["interval"][:len(df) - 1].values != df["ts"][1:len(df)].values)[
            0].tolist()
    list_rows_to_drop = np.unique(list_end_traj_ind + list_discontinuous_ind).tolist()
    df.drop(list_rows_to_drop, inplace=True)

    pickle.dump(df, open(os.path.join(args.data_dir, args.memo, args.scenario, "traj_lrn_trimmed.pkl"), "wb"))

