import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import os
import argparse
import pickle

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from utils.evaluate import evaluate
from interpolation.model import FCNetwork, model_train
from imitation.bc.utilities import gen_traj_with_policy
from interpolation.data_preparation import build_training_data_for_interpolation

parser = argparse.ArgumentParser(description='PyTorch Behavioral Cloning')

# gym environment

parser.add_argument('--env_name', type=str, default="gym_citycar-v0",
                    help='name of the environment to run')
parser.add_argument('--scenario', type=str, default="LA")
parser.add_argument('--memo', type=str, default="default_memo")

# training process
parser.add_argument('--lr', type=int, default=0.0001,
                    help='learning rate of optimizer')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size')
parser.add_argument('--hidden_size', type=int, default=40,
                    help='hidden size of FCNetwork')
parser.add_argument('--num_epochs', type=int, default=20,
                    help='number of training epochs')

# names
parser.add_argument('--experiment_name', type=str, default="bc",
                    help='experiment name')
parser.add_argument('--model_name', type=str, default="model_exp.pkl",
                    help='model_name')

# path
parser.add_argument('--data_dir', type=str, default="data/output/bc",
                    help='dir to data files')

# experiment hyper-parameter
parser.add_argument('--N_EXP', type=int, default=1,
                    help='number of experiment')
parser.add_argument('--max_episode_len', type=int, default=300,
                    help='time length for trajectory')
parser.add_argument('--reward_function', type=int, default=1,
                    help='reward_function')
parser.add_argument('--isgail', type=str, default=None,
                    help='whether used for gail')
args = parser.parse_args()


def main():
    if args.isgail is None:
        gail = False
        posfix = ""
    else:
        gail = True
        posfix = "_gail"
    # =====data preparation ======
    traj_exp_df = build_training_data_for_interpolation(
       dense_traj_file=open("data/expert_trajs/{0}/{1}/traj_raw.pkl".format(args.memo, args.scenario, posfix), 'rb'),
       path_to_output=open("data/expert_trajs/{0}/{1}/training_for_interpolation{2}.pkl".
                           format(args.memo, args.scenario, posfix), "wb"),
       gail=gail,
       sparse_second=100000
    )
    # =====extract data and pocess the data =====
    traj_filename = "data/expert_trajs/{0}/{1}/training_for_interpolation{2}.pkl"\
        .format(args.memo, args.scenario, posfix)

    traj_exp_df = pd.read_pickle(traj_filename)
    # list_x_name = ['interval', 'speed', 'pos_in_lane', 'lane_max_speed', 'if_exit_lane', 'dist_to_signal', 'phase',
    #                'if_leader', 'leader_speed', 'dist_to_leader']
    if gail:
        list_x_name = ['speed_s', 'if_leader_s', 'leader_speed_s', 'dist_to_leader_s', 'action_s',
                       'interval_s', 'phase_s', 'if_exit_lane_s', 'lane_max_speed_s', 'pos_in_lane_s',
                       'dist_to_signal_s',
                       'speed_e', 'if_leader_e', 'leader_speed_e', 'dist_to_leader_e', 'action_e',
                       'interval_e', 'phase_e', 'if_exit_lane_e', 'lane_max_speed_e', 'pos_in_lane_e',
                       'dist_to_signal_e',
                       'time_interval']
        list_y_name = ['speed', 'if_leader', 'leader_speed', 'dist_to_leader', 'action',
                       'interval', 'phase', 'if_exit_lane', 'lane_max_speed', 'pos_in_lane',
                       'dist_to_signal']
        multiply = 10
    else:
        list_x_name = ['speed_s', 'if_leader_s', 'leader_speed_s', 'dist_to_leader_s', 'action_s',
                       'speed_e', 'if_leader_e', 'leader_speed_e', 'dist_to_leader_e', 'action_e',
                       'time_interval']
        list_y_name = ['speed', 'if_leader', 'leader_speed', 'dist_to_leader', 'action']
        multiply = 4


    for col_name in traj_exp_df.columns.values:
        if 'action' == col_name:
            traj_exp_df[col_name] = traj_exp_df[col_name] * multiply

    # todo make the predictions has vec_id, ts
    x_array = traj_exp_df[list_x_name].values
    y_array = traj_exp_df[list_y_name].values

    # x_mean = x_array.mean(axis=0)
    # x_std = x_array.std(axis=0)
    # x_array = (x_array - x_array.mean())/x_array.std()

    x = torch.from_numpy(x_array.astype(np.float))
    y = torch.from_numpy(y_array.astype(np.float))

    full_dataset = TensorDataset(x, y)

    dataloaders = {}
    dataloaders['train'] = DataLoader(dataset=full_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    dataloaders['test'] = DataLoader(dataset=full_dataset, batch_size=y_array.shape[0], shuffle=False, num_workers=0)

    # build interpolation network

    net = FCNetwork(len(list_x_name), len(list_y_name), args.hidden_size)
    # criterion = nn.L1Loss()
    criterion = [nn.MSELoss(), nn.L1Loss()]
    optimizer = optim.Adam(net.parameters(), args.lr)

    # ===== training the interpolation model =====

    model_train(args, dataloaders, optimizer, net, criterion)

    # ===== testing the interpolation model =====
    print('Testing...')
    net = torch.load(os.path.join(args.data_dir, args.memo, args.scenario, args.model_name))
    net.eval()

    # add interpolation prediction, save prediction as "traj_interpolated.pkl"
    dataiter = iter(dataloaders['test'])
    Xs, Ys = dataiter.next()
    # Xs = (Xs.data.cpu().numpy() - x_mean)/x_std
    # Xs = torch.from_numpy(Xs.astype(np.float))

    outputs = net(Xs.float())
    arr = outputs.data.cpu().numpy()
    arr = pd.DataFrame(arr, columns=[list_y_name])
    arr['action'] = arr[['action']]/multiply
    pickle.dump(arr, open(os.path.join("/home/weihua/PycharmProjects/LearnSim/data/expert_trajs/{0}/{1}"
                                       .format(args.memo, args.scenario),
                                       "traj_interpolated{}.pkl".format(posfix)), "wb"))
    print('Done!')


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    print(os.getcwd())
    main()
