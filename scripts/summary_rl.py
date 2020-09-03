import pandas as pd
import os
import argparse
from matplotlib import pyplot as plt


def get_one_ts_result(path_to_folder):

    df = pd.read_csv(os.path.join(path_to_folder, "evaluate.csv"), header=0)
    df.sort_values(by="ave_lrn_speed", ascending=False, inplace=True)

    list_ret = df[["iter", "rmse_pos", "rmse_speed", "ave_exp_speed", "ave_lrn_speed", "num_exp", "num_lrn", "rewardVecLrn", "rewardTsLrn"]].values[0].tolist()
    list_ret[0] = len(df)
    df.sort_values(by="iter", inplace=True)

    return list_ret, df


def main(path_to_summary):

    list_result = []

    dic_df = {}

    for ts in os.listdir(path_to_summary):
        scenario = ts
        path_to_folder = os.path.join(path_to_summary, scenario)
        if not os.path.isdir(path_to_folder):
            continue
        if scenario not in dic_df:
            dic_df[scenario] = {}
        list_ret, df = get_one_ts_result(path_to_folder)
        list_result.append([scenario] + list_ret)
        dic_df[scenario] = df

    df_result = pd.DataFrame(list_result, columns=["scenario", "max_iter", "rmse_pos", "rmse_speed", "ave_exp_speed", "ave_lrn_speed", "num_exp", "num_lrn", "rewardVecLrn", "rewardTsLrn"])
    df_result.sort_values(by="scenario", inplace=True)
    df_result.to_csv(os.path.join(path_to_summary, "summary.csv"))

    for scenario, df in dic_df.items():
        plt.figure()
        plt.plot(df["iter"], df["ave_lrn_speed"])
        plt.legend()
        plt.savefig(os.path.join(path_to_summary, "speed_{}.png".format(scenario)))
        plt.clf()

    for scenario, df in dic_df.items():
        plt.figure()
        plt.plot(df["iter"], df["rewardVecLrn"])
        plt.legend()
        plt.savefig(os.path.join(path_to_summary, "reward1_{}.png".format(scenario)))
        plt.clf()

    for scenario, df in dic_df.items():
        plt.figure()
        plt.plot(df["iter"], df["rewardTsLrn"])
        plt.legend()
        plt.savefig(os.path.join(path_to_summary, "reward2_{}.png".format(scenario)))
        plt.clf()

if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

    parser = argparse.ArgumentParser(description='PyTorch GAIL')
    parser.add_argument('--memo', type=str, default='fake_reward_8')
    arguments = parser.parse_args()

    path_to_output = os.path.join("data", "rl", arguments.memo)
    main(path_to_output)
