import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
import numpy as np
import json
from matplotlib import pyplot as plt
import argparse
from utils.traj_utils import *


def evaluate(path_exp_traj, path_lrn_traj, len_lane, max_episode_len):

    '''
    Evaluate RMSE value and plot.
    :param traj_exp_df:
    :param traj_lrn_df:
    :return:
    '''

    traj_exp_df = pd.read_pickle(path_exp_traj)
    traj_lrn_df = pd.read_pickle(path_lrn_traj)

    traj_lrn = extract_abs_pos(traj_lrn_df, len_lane=len_lane)
    traj_lrn = traj_by_time(traj_lrn, name_ts='ts', name_vec_id='vec_id')

    traj_exp = extract_abs_pos(traj_exp_df, len_lane=len_lane)
    traj_exp = traj_by_time(traj_exp, name_ts='ts', name_vec_id='vec_id')

    rmse_pos, rmse_speed, \
    num_vec, num_lrn, \
    dict_exp, dict_lrn, \
    speedAVE_exp, speedAVE_lrn = rmse_eval(traj_exp, traj_lrn, len_lane, max_episode_len)

    path = os.path.split(path_lrn_traj)[0]
    fname = os.path.split(path_lrn_traj)[1]
    names = ['iter', 'rmse_pos', 'rmse_speed', 'ave_exp_speed', 'ave_lrn_speed', 'num_exp', 'num_lrn']
    iter = fname[13:15]

    # plot
    path_figure = os.path.join(path, 'figure')
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
    plt.savefig(os.path.join(path_figure, 'RMSE.png'))
    plt.close(0)

    plt.figure(1)
    plt.title('Vehicle Number in System')
    x = np.linspace(1, max_episode_len, max_episode_len)
    plt.plot(x, num_vec, color='green', label='Expert')
    plt.plot(x, num_lrn, color='yellow', label='Learned')
    plt.xlabel('Time Stamp')
    plt.ylabel('Vehicle Number')
    plt.legend()
    plt.savefig(os.path.join(path_figure, 'VecNum.png'))
    plt.close(1)

    plt.figure(2)
    plt.title('Average Speed')
    x = np.linspace(1, max_episode_len, max_episode_len)
    plt.plot(x, speedAVE_exp, color='green', label='Expert')
    plt.plot(x, speedAVE_lrn, color='yellow', label='Learned')
    plt.xlabel('Time Stamp')
    plt.ylabel('Average Speed (m/s)')
    plt.legend()
    plt.savefig(os.path.join(path_figure, 'AveSpeed.png'))
    plt.close(2)

    rmse_pos = np.array(rmse_pos).mean()
    rmse_speed = np.array(rmse_speed).mean()

    print('Overall RMSE of Position =', rmse_pos)
    print('Overall RMSE of Speed =', rmse_speed)

    print(('Average Speed of Expert Trajectory is {0}.'.format(np.array(speedAVE_exp).mean())))
    print(('Average Speed of Learned Trajectory is {0}.'.format(np.array(speedAVE_lrn).mean())))

    print('In expert trajectory, {0} vehicles has shown in system.'.format(len(dict_exp)))
    print('In learned trajectory, {0} vehicles has shown in system.'.format(len(dict_lrn)))

    # if not os.path.exists(os.path.join(path, 'evaluate.pkl')):
    #     df_result = pd.DataFrame(columns=names)
    # else:
    #     df_result = pd.read_pickle(os.path.join(path, 'evaluate.pkl'))
    # list_result = list(df_result.values)
    # data_save = [iter, rmse_pos, rmse_speed, np.array(speedAVE_exp).mean(), np.array(speedAVE_lrn).mean(),
    #              len(dict_exp), len(dict_lrn)]
    # if data_save[0] in list(df_result['iter'].values):
    #     list_result[-1] = data_save
    # else:
    #     list_result.append(data_save)
    # array_result = np.array(list_result)
    # df_result = pd.DataFrame(array_result, columns=names)
    # df_result.sort_values(by=['iter'], axis=0, inplace=True)
    # df_result.to_pickle(os.path.join(path, 'evaluate.pkl'))
    # df_result.to_csv(os.path.join(path, 'evaluate.csv'))


if __name__ == "__main__":

    os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/../')
    print(os.getcwd())

    scenario = 'LA'
    imitation_method = 'bc'

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_exp_traj', type=str, default="data/expert_trajs/default_memo/{0}/traj_raw.pkl".format(scenario),
                        help='dir to expert trajectory')
    parser.add_argument('--path_lrn_traj', type=str, default="imitation/{0}/data/{1}/traj_lrn_raw.pkl".format(imitation_method, scenario),
                        help='dir to learned trajectory')
    parser.add_argument('--path_to_len_lane', type=str, default="data/{0}/roadnet_len.json".format(scenario),
                        help='dir to lane length json file')
    parser.add_argument('--max_episode_len', type=int, default=500,
                        help='time length for trajectory')
    args = parser.parse_args()

    f = open(args.path_to_len_lane, encoding='utf-8')
    len_lane = json.load(f)

    evaluate(path_exp_traj='/mnt/project/LearnSim/data/expert_trajs/default_memo/LA/traj_raw.pkl',
             path_lrn_traj='/mnt/traj_raw_iter25.pkl',
             len_lane=len_lane,
             max_episode_len=args.max_episode_len)
