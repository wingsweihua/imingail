import gym
import gym_citycar
import os
import json
import sys
import pandas as pd
import pickle
import numpy as np
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append('.')
from utils.log_utils import Stack
from utils.check_utils import check_action_equals_next_speed
from utils.traj_utils import extract_sparse_from_dense
import argparse
from shutil import copy

parser = argparse.ArgumentParser(description='PyTorch GAIL')
parser.add_argument('--env_name', type=str, default="gym_citycar-v0",
                    help='name of the environment to run')
parser.add_argument('--max_episode_len', type=int, default=500,
                    help='how many time steps to run in each episode')
parser.add_argument('--max_iter_num', type=int, default=1,
                    help='how many iterations to generate')
parser.add_argument('--scenario', type=str, default="4x4_gudang")
parser.add_argument('--memo', type=str, default="default_memo")

args = parser.parse_args()

LIST_VARS = [
        # simulation params
        "interval",
        # all vehicle obs

        # all lane obs

        # current vehicle static params
        "max_pos_acc", "max_neg_acc", "max_speed", "min_gap", "headway_time",
        # current vehicle dynamic obs
        "speed", "pos_in_lane", "lane_max_speed", "if_exit_lane", "dist_to_signal", "phase", "if_leader",
        # leader vehicle static params
        "leader_max_pos_acc", "leader_max_neg_acc", "leader_max_speed",
        # leader vehicle dynamic obs
        "leader_speed", "dist_to_leader",

    ]

def main():

    # ============== initial output ==================
    path_to_output = os.path.join("data", "expert_trajs", args.memo, args.scenario)
    if not os.path.exists(path_to_output):
        os.makedirs(path_to_output)

    # ============== copy the conf files ==================
    path_to_sim_conf = os.path.join("config", "simulator", "{}.json".format(args.scenario))
    copy(path_to_sim_conf, path_to_output)

    # ============== initial env =====================

    env = gym.make('gym_citycar-v0', normalize=True, path_to_conf_file=path_to_sim_conf,
                   list_vars_to_subscribe=["interval", "speed", "pos_in_lane", "lane_max_speed", "if_exit_lane", "dist_to_signal", "phase", "if_leader", "leader_speed", "dist_to_leader",],
                   reward_function=1)
    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]
    obs_header = env.observation_header
    action_header = env.action_header

    # ============== initial storage =================
    # organize the trajs by vehicle
    list_traj = Stack(obs_header, action_header)

    # ============== start looping experiment ========
    for ind_exp in range(args.max_iter_num):
        n_obs, n_info = env.reset()
        for t in range(args.max_episode_len):
            # follow policy to generate samples
            n_action = [np.array([a]) for a in n_info["next_speed_est"]]
            next_n_obs, n_reward, n_done, next_n_info = env.step(n_action, n_info)
            assert check_action_equals_next_speed(next_n_obs, next_n_info, n_action, n_info, obs_header)

            list_traj.append([n_info["vec_id"], n_info["current_time"], n_info["lane_id"], n_obs, n_action, n_reward, np.zeros_like(n_reward).tolist()])

            n_obs = next_n_obs
            n_info = next_n_info

    # ============== post process and dump data =======

    # full trajectory -- for comparison: dataframe
    df = list_traj.convert_to_df()
    pickle.dump(df, open(os.path.join(path_to_output, "traj_raw.pkl"), "wb"))
    print("================================================================")
    print("********* full trajectory logged")
    print("================================================================")

    # partial trajectory -- for learning: sample wise

    # remove non-continuous samples and end of each sample of each trajs



    df.sort_values(by=["vec_id", "ts"], axis=0, inplace=True)
    df.reset_index(drop=True, inplace=True)

    list_end_traj_ind = np.where(df["vec_id"][:len(df) - 1].values != df["vec_id"][1:len(df)].values)[0].tolist() + \
                        [len(df) - 1]
    list_discontinuous_ind = \
        np.where(df["ts"][:len(df) - 1].values + df["interval"][:len(df) - 1].values != df["ts"][1:len(df)].values)[
            0].tolist()
    list_rows_to_drop = np.unique(list_end_traj_ind + list_discontinuous_ind).tolist()
    df_sample = df.drop(list_rows_to_drop)
    pickle.dump(df_sample, open(os.path.join(path_to_output, "traj_sample.pkl"), "wb"))
    print("================================================================")
    print("********* drop {0} rows for discontinuous samples".format(len(list_rows_to_drop)))
    print("================================================================")

    # log only x and y  - todo - decide whether need to do this

    # sparse trajectory

    # only keep trajs at beginning, middle and end
    df_sparse = extract_sparse_from_dense(df)
    pickle.dump(df_sparse, open(os.path.join(path_to_output, "traj_sparse.pkl"), "wb"))


if __name__ == "__main__":
    main()


