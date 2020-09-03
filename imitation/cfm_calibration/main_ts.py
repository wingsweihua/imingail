# calibration of CFM with Tabu Search
import os
import argparse
import pickle

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

from imitation.cfm_calibration.utility import *
from utils.evaluate import evaluate
from utils.traj_utils import get_rmse
from imitation.cfm_calibration.calibration import TabuSearch

parser = argparse.ArgumentParser(description=None)

parser.add_argument('--env_name', type=str, default="gym_citycar-v0",
                    help='name of the environment to run')
parser.add_argument('--scenario', type=str, default='LA')
parser.add_argument('--memo', type=str, default='default_memo')
parser.add_argument('--data_dir', type=str, default='data/output/cfm-ts')

# experiment hyper-parameter
parser.add_argument('--N_EXP', type=int, default=1,
                    help='number of experiment')
parser.add_argument('--max_episode_len', type=int, default=300,
                    help='time length for trajectory')
parser.add_argument('--ts', type=int, default=1,
                    help='time stamp (s)')
parser.add_argument('--observe_point', type=float, default=0,
                    help='position for observing flow data')
parser.add_argument('--time_period', type=int, default=100,
                    help='time period length for calculating match error')
parser.add_argument('--maxTabuSize', type=int, default=10,
                    help='Max size of Tabu List')
parser.add_argument('--max_iter', type=int, default=10,
                    help='number of maximum iteration')
parser.add_argument('--num_neighbors', type=int, default=10,
                    help='number of neighbors')
parser.add_argument('--reward_function', type=int, default=1,
                    help='reward function')
parser.add_argument('--interpolated', type=str, default="sparse",
                    help='interpolated')

ARGS = parser.parse_args()
print(ARGS)


def main():

    k = ARGS.num_neighbors

    TS = TabuSearch(ARGS)
    # paramsInit = TS.paramsInit()
    paramsInit = [0.6, 0.4, 0.4]
    lossBest = TS.fitness(paramsInit, ARGS)
    print('Initialization: [%0.2f, %0.2f, %0.2f] -- loss: %0.3f' % (paramsInit[0], paramsInit[1],
                                                     paramsInit[2], lossBest))
    paramsBest = paramsInit
    bestCandidate = paramsInit
    lossBestCandidate = TS.fitness(bestCandidate, ARGS)

    tabuList = []

    param_space = TS.get_param_space()

    for itr in range(ARGS.max_iter):

        paramsNeighbour = TS.get_Neighbours(bestCandidate, param_space, k)
        print("Iter: {0}".format(itr), "-- search space: {0}".format(len(param_space)),
              '-- best: [%0.2f, %0.2f, %0.2f]' % (paramsBest[0], paramsBest[1], paramsBest[2]))

        for paramsCandidate in paramsNeighbour:

            lossCandidate = TS.fitness(paramsCandidate, ARGS)
            print('[%0.2f, %0.2f, %0.2f]' % (paramsCandidate[0], paramsCandidate[1], paramsCandidate[2]), end='')

            if (paramsCandidate not in tabuList) and (lossCandidate < lossBestCandidate):
                bestCandidate = paramsCandidate
                lossBestCandidate = TS.fitness(bestCandidate, ARGS)
                print('-- loss: %0.3f -- update' % lossCandidate)
            else:
                print('-- loss: %0.3f -- drop' % lossCandidate)
            param_space.remove(paramsCandidate)

        if lossBestCandidate < lossBest:
            paramsBest = bestCandidate
            lossBest = lossBestCandidate
            print('Best parameters updated! -- [%0.2f, %0.2f, %0.2f] '
                  '-- loss: %0.3f' % (paramsBest[0], paramsBest[1], paramsBest[2], lossBest))

        # if paramsBest in tabuList:
        #     break

        if bestCandidate not in tabuList:
            tabuList.append(bestCandidate)
        if len(tabuList) > ARGS.maxTabuSize:
            tabuList.pop(0)

    print("TabuList: \n", np.array(tabuList))

    paramsDictBest = TS.param_list_to_dict(paramsBest)
    trajBestTabu = gen_trajs_with_CFM(ARGS, paramsDictBest, if_drop=False)

    if not os.path.exists(os.path.join(ARGS.data_dir, ARGS.scenario)):
        os.mkdir(os.path.join(ARGS.data_dir, ARGS.scenario))
    pickle.dump(trajBestTabu, open(os.path.join(ARGS.data_dir, ARGS.scenario, "traj_best_loss.pkl"), "wb"))

    score = []

    for params in tabuList:

        params = TS.param_list_to_dict(params)
        traj = gen_trajs_with_CFM(ARGS, params, if_drop=False)

        rmse_pos, rmse_speed, _, aveSpeed, _, numVec = get_rmse(traj_exp=TS.traj_exp_test,
                                                                traj_lrn=traj,
                                                                len_lane=TS.len_lane,
                                                                max_episode_len=ARGS.max_episode_len)
        score.append([rmse_pos, rmse_speed, aveSpeed, numVec])

    scoreArr = np.array(score)
    rmse_pos = scoreArr[:, 0]
    paramsBestRMSE = tabuList[np.argmin(rmse_pos)]
    paramsDictBestRMSE = TS.param_list_to_dict(paramsBestRMSE)
    trajBestRMSE = gen_trajs_with_CFM(ARGS, paramsDictBestRMSE, if_drop=False)
    pickle.dump(trajBestRMSE, open(os.path.join(ARGS.data_dir, ARGS.scenario, "traj_best_RMSE.pkl"), "wb"))

    print('Best Loss: [%0.2f, %0.2f, %0.2f]' % (paramsBest[0], paramsBest[1], paramsBest[2]),
          'Best RMSE: [%0.2f, %0.2f, %0.2f]' % (paramsBestRMSE[0], paramsBestRMSE[1], paramsBestRMSE[2]))
    print(ARGS)
    evaluate(path_exp_traj=os.path.join(TS.path_to_exp_traj, "traj_raw.pkl"),
             path_lrn_traj=os.path.join(ARGS.data_dir, ARGS.scenario, "traj_best_RMSE.pkl"),
             len_lane=TS.len_lane,
             max_episode_len=ARGS.max_episode_len)

    evaluate(path_exp_traj=os.path.join(TS.path_to_exp_traj, "traj_sparse.pkl"),
             path_lrn_traj=os.path.join(ARGS.data_dir, ARGS.scenario, "traj_best_RMSE.pkl"),
             len_lane=TS.len_lane,
             max_episode_len=ARGS.max_episode_len,
             sparse=True)

    np.savetxt(os.path.join(ARGS.data_dir, ARGS.scenario, 'TabuList.csv'),
               np.array(tabuList), fmt="%.5f, %.5f, %.5f", delimiter=',')
    np.savetxt(os.path.join(ARGS.data_dir, ARGS.scenario, 'score.csv'),
               score, fmt="%.5f, %.5f, %.5f, %d", delimiter=',')


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    print(os.getcwd())
    main()
