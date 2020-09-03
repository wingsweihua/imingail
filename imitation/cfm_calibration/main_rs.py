# calibration of CFM with Random Search
import argparse
import pickle
import os

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

from imitation.cfm_calibration.utility import *
from utils.evaluate import evaluate
from utils.traj_utils import get_rmse
from imitation.cfm_calibration.calibration import RandomSearch

parser = argparse.ArgumentParser(description=None)

parser.add_argument('--env_name', type=str, default="gym_citycar-v0",
                    help='name of the environment to run')
parser.add_argument('--scenario', type=str, default='LA')
parser.add_argument('--memo', type=str, default='default_memo')
parser.add_argument('--data_dir', type=str, default='data/output/cfm-rs')

# experiment hyper-parameter
parser.add_argument('--N_EXP', type=int, default=1,
                    help='number of experiment')
parser.add_argument('--max_episode_len', type=int, default=500,
                    help='time length for trajectory')
parser.add_argument('--ts', type=int, default=1,
                    help='time stamp (s)')
parser.add_argument('--observe_point', type=float, default=0.05,
                    help='position for observing flow data')
parser.add_argument('--time_period', type=int, default=100,
                    help='time period length for calculating match error')
parser.add_argument('--max_list_size', type=int, default=10,
                    help='max size of accepted list of params')
parser.add_argument('--max_iter', type=int, default=30,
                    help='number of maximum iteration')
parser.add_argument('--tolerence', type=float, default=0.7,
                    help='number of maximum iteration')
parser.add_argument('--reward_function', type=int, default=1,
                    help='reward function')
parser.add_argument('--interpolated', type=str, default="sparse",
                    help='interpolated')

ARGS = parser.parse_args()
print(ARGS)


def main():

    RS = RandomSearch(ARGS)

    accepted = []

    for itr in range(ARGS.max_iter):

        params = RS.get_params()
        print("Iter: {0}".format(itr), '[%0.2f, %0.2f, %0.2f]' % (params['maxSpeed'], params['maxPosAcc'],
                                                                  params['maxNegAcc']), end='')

        try:
            traj_lrn = gen_trajs_with_CFM(ARGS, params, if_drop=False)
        except:
            traj_lrn = None

        try:
            loss = MANE_fitness(RS.traj_exp_train, traj_lrn, RS.len_lane, ARGS)
        except:
            loss = 999

        print(" -- loss %f" % loss, end='')

        if loss < ARGS.tolerence or len(accepted)==0:
            params_ls = list(params.values())
            params_ls.append(loss)
            accepted.append(params_ls)
            print("-- Accepted!")
        else:
            print("-- Drop")

        if len(accepted) > ARGS.max_list_size:
            lossArr = np.array(accepted)[:, 3]
            accepted.remove(accepted[np.argmax(lossArr)])

    acceptedArr = np.array(accepted)
    if not os.path.exists(os.path.join(ARGS.data_dir, ARGS.memo)):
        os.mkdir(os.path.join(ARGS.data_dir, ARGS.memo))
    if not os.path.exists(os.path.join(ARGS.data_dir, ARGS.memo, ARGS.scenario)):
        os.mkdir(os.path.join(ARGS.data_dir, ARGS.memo, ARGS.scenario))
    print(acceptedArr)

    loss = acceptedArr[:, 3]
    ind = np.argmin(loss)
    paramsBestList = list(acceptedArr[ind, 0:3])
    paramsBestDict = RS.param_list_to_dict(paramsBestList)

    traj_lrn = gen_trajs_with_CFM(ARGS, paramsBestDict, if_drop=False)
    pickle.dump(traj_lrn, open(os.path.join(ARGS.data_dir, ARGS.memo, ARGS.scenario, "traj_best_loss.pkl"), "wb"))

    score = []

    for params in accepted:

        params = RS.param_list_to_dict(params)
        traj = gen_trajs_with_CFM(ARGS, params, if_drop=False)

        rmse_pos, rmse_speed, _, aveSpeed, _, numVec = get_rmse(traj_exp=RS.traj_exp_test,
                                                                traj_lrn=traj,
                                                                len_lane=RS.len_lane,
                                                                max_episode_len=ARGS.max_episode_len)
        score.append([rmse_pos, rmse_speed, aveSpeed, numVec])

    scoreArr = np.array(score)
    rmse_pos = scoreArr[:, 0]
    paramsBestRMSE = accepted[np.argmin(rmse_pos)]
    paramsDictBestRMSE = RS.param_list_to_dict(paramsBestRMSE)
    trajBestRMSE = gen_trajs_with_CFM(ARGS, paramsDictBestRMSE, if_drop=False)
    pickle.dump(trajBestRMSE, open(os.path.join("/home/weihua/PycharmProjects/LearnSim",
                                                ARGS.data_dir, ARGS.memo, ARGS.scenario, "traj_best_RMSE.pkl"), "wb"))

    print('Best Loss: [%0.2f, %0.2f, %0.2f]' % (paramsBestList[0], paramsBestList[1], paramsBestList[2]),
          'Best RMSE: [%0.2f, %0.2f, %0.2f]' % (paramsBestRMSE[0], paramsBestRMSE[1], paramsBestRMSE[2]))

    print(ARGS)
    evaluate(path_exp_traj=os.path.join(RS.path_to_exp_traj, "traj_raw.pkl"),
             path_lrn_traj=os.path.join("/home/weihua/PycharmProjects/LearnSim",
                                        ARGS.data_dir, ARGS.memo, ARGS.scenario, "traj_best_RMSE.pkl"),
             len_lane=RS.len_lane,
             max_episode_len=ARGS.max_episode_len)
    evaluate(path_exp_traj=os.path.join(RS.path_to_exp_traj, "traj_sparse.pkl"),
             path_lrn_traj=os.path.join("/home/weihua/PycharmProjects/LearnSim",
                                        ARGS.data_dir, ARGS.memo, ARGS.scenario, "traj_best_RMSE.pkl"),
             len_lane=RS.len_lane,
             max_episode_len=ARGS.max_episode_len,
             sparse=True)

    np.savetxt(os.path.join("/home/weihua/PycharmProjects/LearnSim",
                            ARGS.data_dir, ARGS.memo, ARGS.scenario, 'accepted.csv'),
               np.array(accepted), fmt="%.5f, %.5f, %.5f, %.5f", delimiter=',')
    np.savetxt(os.path.join("/home/weihua/PycharmProjects/LearnSim",
                            ARGS.data_dir, ARGS.memo, ARGS.scenario, 'score.csv'),
               score, fmt="%.5f, %.5f, %.5f, %d", delimiter=',')


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    print(os.getcwd())
    main()
