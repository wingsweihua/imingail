import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import os
import argparse
import json

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')
from utils.evaluate import evaluate
from imitation.bc.model import FCNetwork, model_train
from imitation.bc.utilities import gen_traj_with_policy

parser = argparse.ArgumentParser(description='PyTorch Behavioral Cloning')

# gym environment

parser.add_argument('--env_name', type=str, default="gym_citycar-v0",
                    help='name of the environment to run')
parser.add_argument('--scenario', type=str, default="LA")
parser.add_argument('--memo', type=str, default="default_memo")

# training process
parser.add_argument('--lr', type=int, default=0.001,
                    help='learning rate of optimizer')
parser.add_argument('--batch_size', type=int, default=120,
                    help='batch size')
parser.add_argument('--hidden_size', type=int, default=40,
                    help='hidden size of FCNetwork')
parser.add_argument('--num_epochs', type=int, default=10,
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
parser.add_argument('--interpolated', type=str, default="interpolated",
                    help='interpolated')
args = parser.parse_args()


def main():

    print(args)
    # =====extract data and build network=====
    if args.interpolated == "interpolated":
        traj_filename = "data/expert_trajs/{0}/{1}/traj_interpolated.pkl".format(args.memo, args.scenario)
    elif args.interpolated == "sparse":
        traj_filename = "data/expert_trajs/{0}/{1}/traj_sparse.pkl".format(args.memo, args.scenario)
    else:
        traj_filename = "data/expert_trajs/{0}/{1}/traj_sample.pkl".format(args.memo, args.scenario)

    print("Loading {}".format(traj_filename))
    traj_exp_df = pd.read_pickle(traj_filename)
    # list_x_name = ['interval', 'speed', 'pos_in_lane', 'lane_max_speed', 'if_exit_lane', 'dist_to_signal', 'phase',
    #                'if_leader', 'leader_speed', 'dist_to_leader']
    list_x_name = ['speed',

                   'if_leader', 'leader_speed', 'dist_to_leader']
    list_y_name = ['action']
    x_array = traj_exp_df[list_x_name].values
    y_array = traj_exp_df[list_y_name].values

    x = torch.from_numpy(x_array.astype(np.float))
    y = torch.from_numpy(y_array.astype(np.float))

    full_dataset = TensorDataset(x, y)

    dataloaders = {}
    dataloaders['train'] = DataLoader(dataset=full_dataset, batch_size=args.batch_size,
                                      shuffle=True, num_workers=0)

    # build policy network

    net = FCNetwork(len(list_x_name), len(list_y_name), args.hidden_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), args.lr)

    # =====training=====

    model_train(args, dataloaders, optimizer, net, criterion)

    # =====testing=====

    # load trained policy network
    print('Testing...')
    print("Loading model {}".format(os.path.join(args.data_dir, args.memo, args.scenario, args.model_name)))
    net = torch.load(os.path.join(args.data_dir, args.memo, args.scenario, args.model_name))
    net.eval()

    gen_traj_with_policy(args, policy_model=net)

    print('Calculating RMSE...')

    path_to_len_lane = "data/{0}/roadnet_len.json".format(args.scenario)
    f = open(path_to_len_lane, encoding='utf-8')
    len_lane = json.load(f)

    evaluate(path_exp_traj="data/expert_trajs/{0}/{1}/traj_raw.pkl".format(args.memo, args.scenario),
             path_lrn_traj=os.path.join(args.data_dir, args.memo, args.scenario, 'traj_lrn_raw.pkl'),
             len_lane=len_lane,
             max_episode_len=args.max_episode_len)
    evaluate(path_exp_traj="data/expert_trajs/{0}/{1}/traj_sparse.pkl".format(args.memo, args.scenario),
             path_lrn_traj=os.path.join(args.data_dir, args.memo, args.scenario, 'traj_lrn_raw.pkl'),
             len_lane=len_lane,
             max_episode_len=args.max_episode_len, sparse=True)

    print('Done!')


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    print(os.getcwd())
    main()
