import pandas as pd
import os
import argparse
from matplotlib import pyplot as plt


def get_one_ts_result(path_to_folder):

    df = pd.read_csv(os.path.join(path_to_folder, "evaluate.csv"), header=0)
    # df.sort_values(by="rmse_pos", inplace=True)
    # list_ret = df[["rmse_pos", "rmse_speed", "ave_exp_speed", "ave_lrn_speed", "num_exp", "num_lrn"]].values[0].tolist()
    # df.sort_values(by="iter", inplace=True)

    list_ret = []

    # ["rmse_speed", "rmse_speed_round", "rmse_pos", "rmse_pos_round", "rewardTsLrn", "rewardTsLrn_round", "rewardVecLrn", "rewardVecLrn_round", "ave_exp_speed", "ave_lrn_speed", "num_exp", "num_lrn", "rewardVecExp", "rewardTsExp"]

    ind_min_rmse_speed = df["rmse_speed"].values.argmin()
    list_ret += [df["rmse_speed"].values[ind_min_rmse_speed], ind_min_rmse_speed]
    ind_min_rmse_pos = df["rmse_pos"].values.argmin()
    list_ret += [df["rmse_pos"].values[ind_min_rmse_pos], ind_min_rmse_pos]

    ind_max_r_ts = df["rewardTsLrn"].values.argmax()
    list_ret += [df["rewardTsLrn"].values[ind_max_r_ts], ind_max_r_ts]
    ind_max_r_vec = df["rewardVecLrn"].values.argmax()
    list_ret += [df["rewardVecLrn"].values[ind_max_r_vec], ind_max_r_vec]

    list_ret += df[["ave_exp_speed", "ave_lrn_speed", "num_exp", "num_lrn", "rewardVecExp", "rewardTsExp"]].max().tolist()

    return list_ret, df


def main(path_to_summary):

    list_result = []

    dic_df = {}

    for model_name in os.listdir(path_to_summary):
        if model_name not in ["AIRL", "GAIL"]:
            continue
        directory = os.path.join(path_to_summary, model_name)
        if not os.path.isdir(directory):
            continue
        for ts in os.listdir(directory):
            scenario = '_'.join(ts.split('_')[1:])
            if scenario not in dic_df:
                dic_df[scenario] = {}
            path_to_folder = os.path.join(path_to_summary, model_name, ts)
            list_ret, df = get_one_ts_result(path_to_folder)
            list_result.append([model_name, ts, scenario] + list_ret)
            dic_df[scenario][model_name+'_'+ts.split('_')[0]] = df

    df_result = pd.DataFrame(list_result, columns=["model_name", "ts", "scenario",
                                                   "rmse_speed", "rmse_speed_round", "rmse_pos", "rmse_pos_round",
                                                    "rewardTsLrn", "rewardTsLrn_round", "rewardVecLrn",
                                                    "rewardVecLrn_round", "ave_exp_speed", "ave_lrn_speed", "num_exp",
                                                    "num_lrn", "rewardVecExp", "rewardTsExp"
                                                   ])
    df_result.sort_values(by="scenario", inplace=True)
    df_result.to_csv(os.path.join(path_to_summary, "summary.csv"))

    for scenario, sce_models in dic_df.items():
        plt.figure()
        for model_ts, df in sce_models.items():
            plt.plot(df["iter"], df["rmse_pos"], label=model_ts)
        plt.legend()
        plt.savefig(os.path.join(path_to_summary, "rmse_pos_{}.png".format(scenario)))
        plt.clf()


    for scenario, sce_models in dic_df.items():
        plt.figure()
        for model_ts, df in sce_models.items():
            plt.plot(df["iter"], df["rmse_speed"], label=model_ts)
        plt.legend()
        plt.savefig(os.path.join(path_to_summary, "rmse_speed_{}.png".format(scenario)))
        plt.clf()

    for scenario, sce_models in dic_df.items():
        plt.figure()
        for model_ts, df in sce_models.items():
            plt.plot(df["iter"], df["rewardVecLrn"], label=model_ts)
        plt.legend()
        plt.savefig(os.path.join(path_to_summary, "rewardvec_{}.png".format(scenario)))
        plt.clf()


    for scenario, sce_models in dic_df.items():
        plt.figure()
        for model_ts, df in sce_models.items():
            plt.plot(df["iter"], df["rewardTsLrn"], label=model_ts)
        plt.legend()
        plt.savefig(os.path.join(path_to_summary, "rewardts_{}.png".format(scenario)))
        plt.clf()

if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

    parser = argparse.ArgumentParser(description='PyTorch GAIL')
    parser.add_argument('--memo', type=str, default="fake_2_4x4_maxPosAcc_10_usualPosAcc_10")
    arguments = parser.parse_args()

    path_to_output = os.path.join("data", "output", arguments.memo)
    main(path_to_output)
