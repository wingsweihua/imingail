from utils.log_utils import Stack
from utils.check_utils import check_action_equals_next_speed, filter_unreal_speed
import torch
import gym
import gym_citycar
import os
import pickle
import numpy as np
import pandas as pd


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


def build_training_data_for_interpolation(dense_traj_file, path_to_output, gail=False, sparse_second=5):

    '''
    Build training data for interpolation.
    :param dense_traj_file: dense traj file
                e.g, ../data/expert_trajs/1x1/hangzhou_kn_hz_1h_7_8_827/traj_sample.pkl
    :param sparse_traj_file:
                e.g. ../data/expert_trajs/1x1/hangzhou_kn_hz_1h_7_8_827/traj_sparse.pkl
    :return: save as training_for_interpolation.pkl

    data should be like:
    ['speed_s', 'if_leader_s', 'leader_speed_s', 'dist_to_leader_s', 'action_s',
     'speed_e', 'if_leader_e', 'leader_speed_e', 'dist_to_leader_e', 'action_e',
     'delta_t',
     'speed', 'if_leader', 'leader_speed', 'dist_to_leader', 'action']

    '''

    df = pickle.load(dense_traj_file)
    df.sort_values(by=["vec_id", "ts"], axis=0, inplace=True)
    df.reset_index(drop=True, inplace=True)

    dense_df_grouped = df.groupby(['vec_id'])

    start_trajs = []
    middle_trajs = []
    time_delta = []
    end_trajs = []

    group_count = 0
    for group_name, df_group in dense_df_grouped:
        if group_count % 20 == 0:
            print("iterate ", group_name)
        group_count += 1
        df_group_sort = df_group.sort_values(by=['ts'])
        df_group_sort.reset_index(inplace=True, drop=True)
        df_group_sort['next_ts_in_record'] = df_group_sort['ts'].shift(periods=-1)
        # get the start time
        row_start = df_group_sort.iloc[0]
        sparse_count = 0
        for row_index, row in df_group_sort.iterrows():
            sparse_count += 1
            # if current row ts and next ts diffs more than 1, do the data generation process
            if row['ts'] + 1 != row['next_ts_in_record'] or sparse_count >= sparse_second:
                if gail:
                    row_end = row[['speed', 'if_leader', 'leader_speed', 'dist_to_leader', 'action',
                                   'interval', 'phase', 'if_exit_lane', 'lane_max_speed', 'pos_in_lane', 'dist_to_signal']]
                else:
                    row_end = row[['speed', 'if_leader', 'leader_speed', 'dist_to_leader', 'action']]
                # generate data, extend row end to every records
                end_trajs.extend([row_end.values.tolist()]*(len(start_trajs)-len(end_trajs)))
                if row_index+1 < len(df_group_sort):
                    row_start = df_group_sort.iloc[row_index + 1]
                    sparse_count = 0
            else:
                if gail:
                    start_trajs.append(row[['speed', 'if_leader', 'leader_speed', 'dist_to_leader', 'action',
                                            'interval', 'phase', 'if_exit_lane', 'lane_max_speed', 'pos_in_lane',
                                            'dist_to_signal']].values.tolist()
                                       )
                    middle_trajs.append(row[['speed', 'if_leader', 'leader_speed', 'dist_to_leader', 'action',
                                            'interval', 'phase', 'if_exit_lane', 'lane_max_speed', 'pos_in_lane',
                                            'dist_to_signal']].values.tolist()
                                       )
                else:
                    start_trajs.append(
                        row_start[['speed', 'if_leader', 'leader_speed', 'dist_to_leader', 'action']].values.tolist()
                    )
                    middle_trajs.append(
                        row[['speed', 'if_leader', 'leader_speed', 'dist_to_leader', 'action']].values.tolist()
                    )
                time_delta.append((row['ts']-row_start['ts']))

    training_df = pd.DataFrame(
        {'start_traj': start_trajs,
         'end_traj': end_trajs,
         'time_interval': time_delta,
         'middle_traj': middle_trajs,
         })
    if gail:
        results = pd.DataFrame(training_df['start_traj'].values.tolist(),
                               columns=['speed_s', 'if_leader_s', 'leader_speed_s', 'dist_to_leader_s', 'action_s',
                                        'interval_s', 'phase_s', 'if_exit_lane_s', 'lane_max_speed_s', 'pos_in_lane_s',
                                        'dist_to_signal_s'])
        results[['speed_e', 'if_leader_e', 'leader_speed_e', 'dist_to_leader_e', 'action_e',
                 'interval_e', 'phase_e', 'if_exit_lane_e', 'lane_max_speed_e', 'pos_in_lane_e',
                 'dist_to_signal_e']] = pd.DataFrame(training_df['end_traj'].values.tolist(),
                                                     index=results.index)
        results['time_interval'] = training_df['time_interval']
        results[['speed', 'if_leader', 'leader_speed', 'dist_to_leader', 'action',
                 'interval', 'phase', 'if_exit_lane', 'lane_max_speed', 'pos_in_lane',
                 'dist_to_signal']] = pd.DataFrame(training_df['middle_traj'].values.tolist(),
                                                   index=results.index)
    else:
        results = pd.DataFrame(training_df['start_traj'].values.tolist(),
                               columns=['speed_s', 'if_leader_s', 'leader_speed_s', 'dist_to_leader_s', 'action_s'])
        results[['speed_e', 'if_leader_e', 'leader_speed_e',
                 'dist_to_leader_e', 'action_e']] = pd.DataFrame(training_df['end_traj'].values.tolist(),
                                                                 index=results.index)
        results['time_interval'] = training_df['time_interval']
        results[['speed', 'if_leader', 'leader_speed',
                 'dist_to_leader', 'action']] = pd.DataFrame(training_df['middle_traj'].values.tolist(),
                                                             index=results.index)

    pickle.dump(results, path_to_output)
    return results





