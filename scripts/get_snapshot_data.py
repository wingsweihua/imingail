import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
import numpy as np
import json
from matplotlib import pyplot as plt
import argparse

from utils.traj_utils import *


def evaluate(path_exp_traj, path_lrn_traj_0, path_lrn_traj_1, path_lrn_traj_2, path_lrn_traj_3, path_lrn_traj_4,
             path_lrn_traj_5, len_lane, max_episode_len, path_save_fig, color, shot_time):

    '''
    Evaluate RMSE value and plot.
    :param traj_exp_df:
    :param traj_lrn_df:
    :return:
    '''

    path_list = [path_lrn_traj_0, path_lrn_traj_1, path_lrn_traj_2, path_lrn_traj_3, path_lrn_traj_4,
                 path_lrn_traj_5, path_exp_traj]

    snapshot_data = np.zeros([len(path_list), len(shot_time)])
    a_list = []
    b_list = []

    for i in range(len(path_list)):
        traj_df = pd.read_pickle(path_list[i])
        speedList, posList, a, b = get_single_vehicle_traj(traj_df, max_episode_len, len_lane, args.vec_id)
        a_list.append(a)
        b_list.append(b)
        for j in range(len(shot_time)):
            time = shot_time[j]
            pos = posList[time]
            snapshot_data[i, j] = pos * 1000
    aa = min(a_list)
    bb = max(b_list)

    print(args.vec_id)
    print(shot_time)
    print(snapshot_data)

    fig = plt.figure(1)
    ax = plt.subplot(111)

    for i in range(len(path_list)):

        y = snapshot_data[i, :]
        x = np.broadcast_to(i, [len(y),])
        ax.scatter(x, y)

    plt.savefig(os.path.join(path_save_fig, 'Scatter.pdf'))
    plt.savefig(os.path.join(path_save_fig, 'Scatter.png'))





if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    print(os.getcwd())

    data_index = 2
    data_list = ["hangzhou_bc_tyc_1h_10_11_2021", "hangzhou_kn_hz_1h_7_8_827", "hangzhou_sb_sx_1h_7_8_1671",
                 "4x4_gudang", "LA"]
    data_name_list = ["HZ-1", "HZ-2", "HZ-3", "GD", "LA"]
    scenario = data_list[data_index]
    data_name = data_name_list[data_index]
    data_dir = "/mnt/project/LearnSim/data/output/figure"
    shot_time = [300, 350, 400, 450, 499]

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_exp_traj', type=str, default="data/expert_trajs/default_memo/{0}/traj_raw.pkl".format(scenario),
                        help='dir to expert trajectory')
    parser.add_argument('--path_lrn_traj_0', type=str, default="{}/trajs_for_plot/{}/traj_lrn_CFM_RS.pkl".format(data_dir, scenario),
                        help='dir to learned trajectory')
    parser.add_argument('--path_lrn_traj_1', type=str, default="{}/trajs_for_plot/{}/traj_lrn_CFM_TS.pkl".format(data_dir, scenario),
                        help='dir to learned trajectory')
    parser.add_argument('--path_lrn_traj_2', type=str, default="{}/trajs_for_plot/{}/traj_lrn_BC.pkl".format(data_dir, scenario),
                        help='dir to learned trajectory')
    parser.add_argument('--path_lrn_traj_3', type=str, default="{}/trajs_for_plot/{}/traj_lrn_MaxEnt.pkl".format(data_dir, scenario),
                        help='dir to learned trajectory')
    parser.add_argument('--path_lrn_traj_4', type=str, default="{}/trajs_for_plot/{}/traj_lrn_GAIL.pkl".format(data_dir, scenario),
                        help='dir to learned trajectory')
    parser.add_argument('--path_lrn_traj_5', type=str, default="{}/trajs_for_plot/{}/traj_lrn_AIRL.pkl".format(data_dir, scenario),
                        help='dir to learned trajectory')

    parser.add_argument('--max_episode_len', type=int, default=500,
                        help='time length for trajectory')
    parser.add_argument('--vec_id', type=str, default='flow_90_0',
                        help='vehicle ID')
    parser.add_argument('--path_save_fig', type=str, default="{}/fig_6_15/{}".format(data_dir, scenario),
                        help='dir to model file')
    args = parser.parse_args()

    path_to_len_lane = "data/{0}/roadnet_len.json".format(scenario)
    f = open(path_to_len_lane, encoding='utf-8')
    len_lane = json.load(f)

    color = ['dimgray', 'darkgrey', 'C8', 'C1', 'C9', 'b', 'C3']

    evaluate(path_exp_traj=args.path_exp_traj,
             path_lrn_traj_0=args.path_lrn_traj_0,
             path_lrn_traj_1=args.path_lrn_traj_1,
             path_lrn_traj_2=args.path_lrn_traj_2,
             path_lrn_traj_3=args.path_lrn_traj_3,
             path_lrn_traj_4=args.path_lrn_traj_4,
             path_lrn_traj_5=args.path_lrn_traj_5,
             len_lane=len_lane,
             max_episode_len=args.max_episode_len,
             path_save_fig=args.path_save_fig,
             color=color,
             shot_time=shot_time)
