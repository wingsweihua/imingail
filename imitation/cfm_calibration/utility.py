import numpy as np
import pandas as pd
import gym
import gym_citycar
import os

from imitation.cfm_calibration.cfm import krauss_default
from utils.traj_utils import extract_abs_pos, traj_by_time, extract_sparse_from_dense
from utils.log_utils import Stack
from utils.check_utils import filter_unreal_speed


def get_flow_speed(traj_abs_pos, start_time, end_time, observe_point):

    traj_period = traj_abs_pos[(traj_abs_pos.ts >= start_time) & (traj_abs_pos.ts < end_time)]
    vec_left = set(list(traj_period[traj_period.abs_pos <= observe_point]['vec_id'].values))
    vec_right = set(list(traj_period[traj_period.abs_pos > observe_point]['vec_id'].values))
    flow = len(vec_left & vec_right) / (end_time - start_time)
    speed = traj_period['speed'].values.mean()

    return flow, speed


def MANE_fitness(traj_exp, traj_lrn, len_lane, args):

    time_period = args.time_period
    max_episode_len = args.max_episode_len
    observe_point = args.observe_point

    traj_lrn = extract_abs_pos(traj_lrn, len_lane=len_lane)
    traj_lrn = traj_by_time(traj_lrn, name_ts='ts', name_vec_id='vec_id')
    if args.interpolated == "sparse":
        traj_lrn['interval'] = 1
        traj_lrn = extract_sparse_from_dense(traj_lrn)

    traj_exp = extract_abs_pos(traj_exp, len_lane=len_lane)
    traj_exp = traj_by_time(traj_exp, name_ts='ts', name_vec_id='vec_id')

    start_time = 200

    err_flow = []
    err_speed = []

    while True:

        flow_exp, speed_exp = get_flow_speed(traj_exp, start_time, start_time + time_period, observe_point)
        flow_lrn, speed_lrn = get_flow_speed(traj_lrn, start_time, start_time + time_period, observe_point)

        err_flow.append(abs(flow_exp - flow_lrn) / flow_exp if flow_exp!=0 else 0)
        err_speed.append(abs(speed_exp - speed_lrn) / speed_exp)

        start_time += time_period
        if start_time + time_period > max_episode_len:
            break

    score = np.mean(err_flow) + np.mean(err_speed)

    return score


def gen_trajs_with_CFM(args, params, if_drop):

    # initialize env
    path_to_sim_conf = os.path.join("config", "simulator", "{}.json".format(args.scenario))
    env = gym.make('gym_citycar-v0', normalize=True, path_to_conf_file=path_to_sim_conf,
                   list_vars_to_subscribe=["interval", "speed", "pos_in_lane", "lane_max_speed", "if_exit_lane",
                                           "dist_to_signal", "phase", "if_leader", "leader_speed", "dist_to_leader"],
                   reward_function=args.reward_function)
    obs_header = env.observation_header
    action_header = env.action_header

    # initial storage, organize the trajs by vehicle
    list_traj = Stack(obs_header, action_header)

    # start looping experiment
    for ind_exp in range(args.N_EXP):
        n_obs, n_info = env.reset()
        for t in range(args.max_episode_len):
            # follow policy to generate samples
            if n_obs == []:
                n_action = []
            else:
                n_obs_array = np.array(n_obs)
                speed_curr, speed_lead, if_leader, gap = n_obs_array[:, 1], n_obs_array[:, 8], \
                                                       n_obs_array[:, 7], n_obs_array[:, 9]
                maxSpeed, maxPosAcc, maxNegAcc = params['maxSpeed'], params['maxPosAcc'], params['maxNegAcc']
                n_action_array = krauss_default(maxSpeed, maxPosAcc, maxNegAcc,
                                                speed_curr, speed_lead, if_leader, args.ts, gap)
                n_action = [a for a in n_action_array.reshape(len(n_action_array), 1)]
                n_action = filter_unreal_speed(n_action, n_info['next_speed_est'])

            next_n_obs, n_reward, n_done, next_n_info = env.step(n_action, n_info)
            n_irl_reward = [a for a in np.zeros(len(n_reward))]
            list_traj.append([n_info["vec_id"], n_info["current_time"], n_info["lane_id"], n_obs, n_action,
                              n_reward, n_irl_reward])

            n_obs = next_n_obs
            n_info = next_n_info

    df = list_traj.convert_to_df()

    if if_drop:
        list_end_traj_ind = np.where(df["vec_id"][:len(df) - 1].values != df["vec_id"][1:len(df)].values)[0].tolist() + [
        len(df) - 1]
        list_discontinuous_ind = \
        np.where(df["ts"][:len(df) - 1].values + df["interval"][:len(df) - 1].values != df["ts"][1:len(df)].values)[
            0].tolist()
        list_rows_to_drop = np.unique(list_end_traj_ind + list_discontinuous_ind).tolist()
        df.drop(list_rows_to_drop, inplace=True)

    return df


