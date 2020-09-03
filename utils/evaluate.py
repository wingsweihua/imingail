import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from utils.traj_utils import extract_abs_pos, traj_by_time, rmse_eval, get_reward_from_traj, \
    rmse_eval_time, extract_sparse_from_dense, rmse_eval_time_dense


def evaluate(path_exp_traj, path_lrn_traj, len_lane, max_episode_len, sparse=False):
    posfix = ''
    if sparse:
        posfix = "_sparse"

    traj_exp_df = pd.read_pickle(path_exp_traj)
    traj_lrn_df = pd.read_pickle(path_lrn_traj)

    traj_exp = extract_abs_pos(traj_exp_df, len_lane=len_lane)
    traj_lrn = extract_abs_pos(traj_lrn_df, len_lane=len_lane)

    traj_exp_origin = traj_exp
    if sparse:
        traj_lrn['interval'] = 1
        traj_lrn_origin = extract_sparse_from_dense(traj_lrn)
    else:
        traj_lrn['interval'] = 1
        traj_lrn_origin = traj_lrn

    traj_exp = traj_by_time(traj_exp, name_ts='ts', name_vec_id='vec_id')
    traj_lrn = traj_by_time(traj_lrn, name_ts='ts', name_vec_id='vec_id')


    rmse_pos, rmse_speed, \
    num_vec, num_lrn, \
    dict_exp, dict_lrn, \
    speedAVE_exp, speedAVE_lrn = rmse_eval(traj_exp, traj_lrn, len_lane, max_episode_len)

    path = os.path.split(path_lrn_traj)[0]
    fname = os.path.split(path_lrn_traj)[1]
    names = ['iter', 'rmse_pos', 'rmse_speed', 'ave_exp_speed', 'ave_lrn_speed', 'num_exp', 'num_lrn',
             'rewardVecExp', 'rewardTsExp', 'rewardVecLrn', 'rewardTsLrn']
    iter = fname.split(".")[0][-2:]

    # plot
    path_figure = os.path.join(path, 'figure_iter'+iter)
    if not os.path.exists(path_figure):
        os.mkdir(path_figure)

    plt.switch_backend('agg')
    x = np.linspace(1, max_episode_len, max_episode_len)
    plt.figure(0)
    plt.subplot(211)
    plt.title('RMSE Analysis')
    plt.plot(x, rmse_pos, color='blue')
    plt.ylabel('Position')
    plt.subplot(212)
    plt.plot(x, rmse_speed, color='red')
    plt.xlabel('Time Stamp')
    plt.ylabel('Speed')
    plt.savefig(os.path.join(path_figure, 'RMSE{}.png'.format(posfix)))
    plt.close(0)

    plt.figure(1)
    plt.title('Vehicle Number in System')
    x = np.linspace(1, max_episode_len, max_episode_len)
    plt.plot(x, num_vec, color='green', label='Expert')
    plt.plot(x, num_lrn, color='yellow', label='Learned')
    plt.xlabel('Time Stamp')
    plt.ylabel('Vehicle Number')
    plt.legend()
    plt.savefig(os.path.join(path_figure, 'VecNum{}.png'.format(posfix)))
    plt.close(1)

    plt.figure(2)
    plt.title('Average Speed')
    x = np.linspace(1, max_episode_len, max_episode_len)
    plt.plot(x, speedAVE_exp, color='green', label='Expert')
    plt.plot(x, speedAVE_lrn, color='yellow', label='Learned')
    plt.xlabel('Time Stamp')
    plt.ylabel('Average Speed (m/s)')
    plt.legend()
    plt.savefig(os.path.join(path_figure, 'AveSpeed{}.png'.format(posfix)))
    plt.close(2)

    rmse_pos = np.array(rmse_pos).mean()
    rmse_speed = np.array(rmse_speed).mean()

    if 'reward' in list(traj_exp_df):
        rewardVecExp, rewardTsExp = get_reward_from_traj(traj_exp_df, max_episode_len)
        rewardVecExp, rewardTsExp = np.array(rewardVecExp).mean(), np.array(rewardTsExp).mean()
    else:
        rewardVecExp, rewardTsExp = 0, 0

    if 'reward' in list(traj_lrn_df):
        rewardVecLrn, rewardTsLrn = get_reward_from_traj(traj_lrn_df, max_episode_len)
        rewardVecLrn, rewardTsLrn = np.array(rewardVecLrn).mean(), np.array(rewardTsLrn).mean()
    else:
        rewardVecLrn, rewardTsLrn = 0, 0

    print('Reward: expert - [%0.4f, %0.4f], learned - [%0.4f, %0.4f]'
          % (rewardVecExp, rewardTsExp, rewardVecLrn, rewardTsLrn))

    print('Overall RMSE of Position =', rmse_pos)
    print('Overall RMSE of Speed =', rmse_speed)
    if sparse:
        rmse_time_per_vehicle = rmse_eval_time(traj_exp_origin, traj_lrn_origin)
        rmse_time = np.array(rmse_time_per_vehicle).mean()
        print('on sparse data, Overall RMSE of time =', rmse_time)
    else:
        rmse_time = rmse_eval_time_dense(traj_exp, traj_lrn)
        print('on dense data, Overall RMSE of time =', rmse_time)

    print(('Average Speed of Expert Trajectory is {0}.'.format(np.array(speedAVE_exp).mean())))
    print(('Average Speed of Learned Trajectory is {0}.'.format(np.array(speedAVE_lrn).mean())))

    print('In expert trajectory, {0} vehicles has shown in system.'.format(len(dict_exp)))
    print('In learned trajectory, {0} vehicles has shown in system.'.format(len(dict_lrn)))

    if not os.path.exists(os.path.join(path, 'evaluate{}.pkl'.format(posfix))):
        df_result = pd.DataFrame(columns=names)
    else:
        df_result = pd.read_pickle(os.path.join(path, 'evaluate{}.pkl'.format(posfix)))
    list_result = list(df_result.values)
    data_save = [iter, rmse_pos, rmse_speed, np.array(speedAVE_exp).mean(), np.array(speedAVE_lrn).mean(),
                 len(dict_exp), len(dict_lrn), rewardVecExp, rewardTsExp, rewardVecLrn, rewardTsLrn]
    if data_save[0] in list(df_result['iter'].values):
        list_result[-1] = data_save
    else:
        list_result.append(data_save)
    array_result = np.array(list_result)
    df_result = pd.DataFrame(array_result, columns=names)
    df_result.sort_values(by=['iter'], axis=0, inplace=True)
    df_result.to_pickle(os.path.join(path, 'evaluate{}.pkl'.format(posfix)))
    df_result.to_csv(os.path.join(path, 'evaluate{}.csv'.format(posfix)))

    return rmse_pos, rmse_speed

