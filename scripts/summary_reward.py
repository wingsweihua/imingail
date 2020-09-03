import pandas as pd
import os
import argparse
from matplotlib import pyplot as plt


def get_one_ts_result(path_to_folder):

    df = pd.read_csv(os.path.join(path_to_folder, "reward.csv"), header=0)
    df.sort_values(by="iter", inplace=True)
    list_ret = df[["mean", "sum"]].values[15:].mean(axis=0).tolist()

    return list_ret, df


def main(path_to_summary):

    list_result = []

    dic_df = {}

    for model_name in os.listdir(path_to_summary):
        directory = os.path.join(path_to_summary, model_name)
        if not os.path.isdir(directory):
            continue
        for ts in os.listdir(directory):
            scenario = '_'.join(ts.split('_')[1:])
            if scenario not in dic_df:
                dic_df[scenario] = {}
            try:
                path_to_folder = os.path.join(path_to_summary, model_name, ts)
                list_ret, df = get_one_ts_result(path_to_folder)
                list_result.append([model_name, ts, scenario] + list_ret)
                dic_df[scenario][model_name+'_'+ts.split('_')[0]] = df
            except KeyError:
                pass

    df_result = pd.DataFrame(list_result, columns=["model_name", "ts", "scenario", "mean_reward", "sum_reward"])
    df_result.sort_values(by="scenario", inplace=True)
    df_result.to_csv(os.path.join(path_to_summary, "summary_reward.csv"))

    for scenario, sce_models in dic_df.items():
        plt.figure()
        for model_ts, df in sce_models.items():
            plt.plot(df["iter"], df["mean"], label=model_ts)
        plt.legend()
        plt.savefig(os.path.join(path_to_summary, "mean_reward_{}.png".format(scenario)))
        plt.clf()

    for scenario, sce_models in dic_df.items():
        plt.figure()
        for model_ts, df in sce_models.items():
            plt.plot(df["iter"], df["sum"], label=model_ts)
        plt.legend()
        plt.savefig(os.path.join(path_to_summary, "sum_reward_{}.png".format(scenario)))
        plt.clf()

if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

    parser = argparse.ArgumentParser(description='PyTorch GAIL')
    parser.add_argument('--memo', type=str, default='fake_reward_minGap_1.5_headwayTime_1.0')
    arguments = parser.parse_args()

    path_to_output = os.path.join("data", "output", arguments.memo)
    main(path_to_output)
