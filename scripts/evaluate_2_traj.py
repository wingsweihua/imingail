import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
import numpy as np
import json
from matplotlib import pyplot as plt
import argparse

from utils.traj_utils import *


def evaluate(path_exp_traj, path_lrn_traj_0, path_lrn_traj_1, len_lane, max_episode_len, path_save_fig):

    '''
    Evaluate RMSE value and plot.
    :param traj_exp_df:
    :param traj_lrn_df:
    :return:
    '''

    traj_exp_df = pd.read_pickle(path_exp_traj)
    traj_lrn_0_df = pd.read_pickle(path_lrn_traj_0)
    traj_lrn_1_df = pd.read_pickle(path_lrn_traj_1)

    traj_lrn_0 = extract_abs_pos(traj_lrn_0_df, len_lane=len_lane)
    traj_lrn_0 = traj_by_time(traj_lrn_0, name_ts='ts', name_vec_id='vec_id')
    traj_lrn_1 = extract_abs_pos(traj_lrn_1_df, len_lane=len_lane)
    traj_lrn_1 = traj_by_time(traj_lrn_1, name_ts='ts', name_vec_id='vec_id')

    traj_exp = extract_abs_pos(traj_exp_df, len_lane=len_lane)
    traj_exp = traj_by_time(traj_exp, name_ts='ts', name_vec_id='vec_id')

    rmse_pos_0, rmse_speed_0, \
    num_vec, num_lrn_0, \
    dict_exp, dict_lrn_0, \
    speedAVE_exp, speedAVE_lrn_0 = rmse_eval(traj_exp, traj_lrn_0, len_lane, max_episode_len)

    rmse_pos_1, rmse_speed_1, \
    num_vec, num_lrn_1, \
    dict_exp, dict_lrn_1, \
    speedAVE_exp, speedAVE_lrn_1 = rmse_eval(traj_exp, traj_lrn_1, len_lane, max_episode_len)

    # plot
    if not os.path.exists(path_save_fig):
        os.mkdir(path_save_fig)

    plt.switch_backend('agg')
    x = np.linspace(1, max_episode_len, max_episode_len)
    plt.figure(0)
    plt.subplot(211)
    plt.title('RMSE Analysis')
    plt.plot(x, rmse_pos_0, color='blue', label='AIRL')
    plt.plot(x, rmse_pos_1, color='red', label='GAIL')
    plt.ylabel('Position')
    plt.legend()
    plt.subplot(212)
    plt.plot(x, rmse_speed_0, color='blue', label='AIRL')
    plt.plot(x, rmse_speed_1, color='red', label='GAIL')
    plt.xlabel('Time Stamp')
    plt.ylabel('Speed')
    plt.legend()
    plt.savefig(os.path.join(path_save_fig, 'RMSE.png'))
    plt.close(0)

    plt.figure(1)
    plt.title('Vehicle Number in System')
    x = np.linspace(1, max_episode_len, max_episode_len)
    plt.plot(x, num_vec, color='green', label='Expert')
    plt.plot(x, num_lrn_0, color='blue', label='AIRL')
    plt.plot(x, num_lrn_1, color='red', label='GAIL')
    plt.xlabel('Time Stamp')
    plt.ylabel('Vehicle Number')
    plt.legend()
    plt.savefig(os.path.join(path_save_fig, 'VecNum.png'))
    plt.close(1)

    plt.figure(2)
    plt.title('Average Speed')
    x = np.linspace(1, max_episode_len, max_episode_len)
    plt.plot(x, speedAVE_exp, color='green', label='Expert')
    plt.plot(x, speedAVE_lrn_0, color='blue', label='AIRL')
    plt.plot(x, speedAVE_lrn_1, color='red', label='GAIL')
    plt.xlabel('Time Stamp')
    plt.ylabel('Average Speed (m/s)')
    plt.legend()
    plt.savefig(os.path.join(path_save_fig, 'AveSpeed.png'))
    plt.close(2)

    rmse_pos_0 = np.array(rmse_pos_0).mean()
    rmse_pos_1 = np.array(rmse_pos_1).mean()
    rmse_speed_0 = np.array(rmse_speed_0).mean()
    rmse_speed_1 = np.array(rmse_speed_1).mean()

    print('Overall RMSE of Position =', rmse_pos_0, rmse_pos_1)
    print('Overall RMSE of Speed =', rmse_speed_0, rmse_speed_1)

    print(('Average Speed of Expert Trajectory is {0}.'.format(np.array(speedAVE_exp).mean())))
    print(('Average Speed of BC Trajectory is {0}.'.format(np.array(speedAVE_lrn_0).mean())))
    print(('Average Speed of GAIL Trajectory is {0}.'.format(np.array(speedAVE_lrn_1).mean())))

    print('In expert trajectory, {0} vehicles has shown in system.'.format(len(dict_exp)))
    print('In BC trajectory, {0} vehicles has shown in system.'.format(len(dict_lrn_0)))
    print('In GAIL trajectory, {0} vehicles has shown in system.'.format(len(dict_lrn_1)))

    # data_save = [rmse_pos, rmse_speed, np.array(speedAVE_exp).mean(), np.array(speedAVE_lrn).mean(),
    #              len(dict_exp), len(dict_lrn)]
    # data_save = np.array(data_save)
    # np.savetxt(os.path.join(path_save_fig, 'eval_output.csv'), data_save)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/../')
    print(os.getcwd())

    scenario = 'hangzhou_kn_hz_1h_7_8_827'
    imitation_method_0 = 'cfm_calibration'
    imitation_method_1 = 'bc'

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_exp_traj', type=str, default="data/expert_trajs/default_memo/{0}/traj_raw.pkl".format(scenario),
                        help='dir to expert trajectory')
    parser.add_argument('--path_lrn_traj_0', type=str, default="imitation/{0}/data/{1}/traj_lrn_raw.pkl".format(imitation_method_0, scenario),
                        help='dir to learned trajectory')
    parser.add_argument('--path_lrn_traj_1', type=str, default="imitation/{0}/data/{1}/traj_lrn_raw.pkl".format(imitation_method_1, scenario),
                        help='dir to learned trajectory')
    parser.add_argument('--path_to_len_lane', type=str, default="data/{0}/roadnet_len.json".format(scenario),
                        help='dir to lane length json file')
    parser.add_argument('--path_save_fig', type=str, default="data/figure/compare_{0}_VS_{1}_{2}".format(imitation_method_0, imitation_method_1, scenario),
                        help='time length for trajectory')
    parser.add_argument('--max_episode_len', type=int, default=500,
                        help='time length for trajectory')
    args = parser.parse_args()

    f = open(args.path_to_len_lane, encoding='utf-8')
    len_lane = json.load(f)

    args = parser.parse_args()
    evaluate(path_exp_traj=args.path_exp_traj,
             path_lrn_traj_0=args.path_lrn_traj_0,
             path_lrn_traj_1=args.path_lrn_traj_1,
             len_lane=len_lane,
             max_episode_len=args.max_episode_len,
             path_save_fig=args.path_save_fig)
