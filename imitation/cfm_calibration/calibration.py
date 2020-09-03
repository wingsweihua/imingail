from imitation.cfm_calibration.utility import *

import os
import pandas as pd
import random
import json
import numpy as np


class RandomSearch():

    def __init__(self, ARGS):

        self.params_template = {
            'maxSpeed': 0.6,
            'maxPosAcc': 0.2,
            'maxNegAcc': 0.2,
        }

        self.maxSpeedMin = 0
        self.maxSpeedMax = 1
        self.maxPosAccMin = 0
        self.maxPosAccMax = 0.5
        self.maxNegAccMin = 0
        self.maxNegAccMax = 0.5

        self.path_to_exp_traj = "data/expert_trajs/{0}/{1}".format(ARGS.memo, ARGS.scenario)
        if ARGS.interpolated == "sparse":
            self.traj_exp_train = pd.read_pickle(os.path.join(self.path_to_exp_traj, "traj_sparse.pkl"))
        elif ARGS.interpolated == "interpolated":
            self.traj_exp_train = pd.read_pickle(
                os.path.join(self.path_to_exp_traj, "traj_interpolated.pkl"))
        else:
            self.traj_exp_train = pd.read_pickle(os.path.join(self.path_to_exp_traj, "traj_sample.pkl"))
        self.traj_exp_test = pd.read_pickle(os.path.join(self.path_to_exp_traj, "traj_raw.pkl"))
        path_to_len_lane = "data/{0}/roadnet_len.json".format(ARGS.scenario)
        f = open(path_to_len_lane, encoding='utf-8')
        self.len_lane = json.load(f)

    def get_params(self):

        params = self.params_template.copy()
        params["maxSpeed"] = self.maxSpeedMin + random.uniform(self.maxSpeedMax, self.maxSpeedMin)
        params["maxPosAcc"] = self.maxPosAccMin + random.uniform(self.maxPosAccMax, self.maxPosAccMin)
        params["maxNegAcc"] = self.maxNegAccMin + random.uniform(self.maxNegAccMax, self.maxNegAccMin)

        return params

    def param_list_to_dict(self, paramsList):

        paramsDict = {
            'maxSpeed': paramsList[0],
            'maxPosAcc': paramsList[1],
            'maxNegAcc': paramsList[2],
        }

        return paramsDict


class TabuSearch:

    def __init__(self, ARGS):

        self.maxSpeedMin = 0.1
        self.maxSpeedMax = 0.8
        self.maxPosAccMin = 0.1
        self.maxPosAccMax = 0.5
        self.maxNegAccMin = 0.1
        self.maxNegAccMax = 0.5

        path_to_len_lane = "data/{0}/roadnet_len.json".format(ARGS.scenario)
        f = open(path_to_len_lane, encoding='utf-8')
        self.len_lane = json.load(f)

        self.path_to_exp_traj = "data/expert_trajs/{0}/{1}".format(ARGS.memo, ARGS.scenario)

        if ARGS.interpolated == "sparse":
            self.traj_exp_train = pd.read_pickle(os.path.join(self.path_to_exp_traj, "traj_sparse.pkl"))
        elif ARGS.interpolated == "interpolated":
            self.traj_exp_train = pd.read_pickle(
                os.path.join(self.path_to_exp_traj, "traj_interpolated.pkl"))
        else:
            self.traj_exp_train = pd.read_pickle(os.path.join(self.path_to_exp_traj, "traj_sample.pkl"))

        self.traj_exp_test = pd.read_pickle(os.path.join(self.path_to_exp_traj, "traj_raw.pkl"))

    def paramsInit(self):

        maxSpeed = self.maxSpeedMin + random.uniform(self.maxSpeedMax, self.maxSpeedMin)
        maxPosAcc = self.maxPosAccMin + random.uniform(self.maxPosAccMax, self.maxPosAccMin)
        maxNegAcc = self.maxNegAccMin + random.uniform(self.maxNegAccMax, self.maxNegAccMin)
        params = [maxSpeed, maxPosAcc, maxNegAcc]

        return params

    def get_param_space(self):

        maxSpeed_space = np.linspace(self.maxSpeedMin, self.maxSpeedMax,
                                     int((self.maxSpeedMax - self.maxSpeedMin) / 0.05), endpoint=False)
        maxPosAcc_space = np.linspace(self.maxPosAccMin, self.maxPosAccMax,
                                      int((self.maxPosAccMax - self.maxPosAccMin) / 0.05), endpoint=False)
        maxNegAcc_space = np.linspace(self.maxNegAccMin, self.maxNegAccMax,
                                      int((self.maxNegAccMax - self.maxNegAccMin) / 0.05), endpoint=False)

        param_space = []
        for maxSpeed in maxSpeed_space:
            for maxPosAcc in maxPosAcc_space:
                for maxNegAcc in maxNegAcc_space:
                    paramsList = [maxSpeed, maxPosAcc, maxNegAcc]
                    param_space.append(paramsList)

        return param_space

    def get_Neighbours(self, params, param_space, k):

        paramsArr = np.array(params)
        paramSpaceArr = np.array(param_space)

        diff = paramsArr - paramSpaceArr
        sqrDiff = diff ** 2
        sqrDistance = sqrDiff.sum(axis=1)
        sortedIdx = sqrDistance.argsort()

        neighbourArr = paramSpaceArr[sortedIdx[0:k]]
        neighbourList = neighbourArr.tolist()

        return neighbourList

    def param_list_to_dict(self, paramsList):

        paramsDict = {
            'maxSpeed': paramsList[0],
            'maxPosAcc': paramsList[1],
            'maxNegAcc': paramsList[2],
        }

        return paramsDict

    def fitness(self, params, ARGS):

        paramsDict = self.param_list_to_dict(params)

        try:
            self.traj_lrn = gen_trajs_with_CFM(ARGS, paramsDict, if_drop=False)
        except:
            self.traj_lrn = None

        try:
            loss = MANE_fitness(self.traj_exp_train, self.traj_lrn, self.len_lane, ARGS)
        except:
            loss = 999

        return loss

