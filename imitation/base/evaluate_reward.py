import os
import pandas as pd


def test_memo_scenario(args):
    # ============== check train output and model ================

    path_to_output_pre = os.path.join("data", "output", args.memo, args.model_name)
    if not os.path.exists(path_to_output_pre):
        print("not well trained")
        return
    else:
        list_output_dirs = os.listdir(path_to_output_pre)

    for d in list_output_dirs:
        if d.endswith(args.scenario):
            test_all_iters(os.path.join(path_to_output_pre, d))

def test_all_iters(path_to_output_pre):
    # ============== check train output and model ================

    list_reward = []

    for file in os.listdir(path_to_output_pre):
        if file.startswith("traj_reward") and file.endswith(".csv"):
            iter_num = file.split('.')[0].split('_')[-1]
            if "iter" in iter_num:
                iter_num = iter_num[4:]

            mean_r, sum_r = cal_one_iter(path_to_output_pre, iter_num)

            list_reward.append([int(iter_num), mean_r, sum_r])

    df_reward = pd.DataFrame(list_reward, columns=["iter", "mean", "sum"])
    df_reward.sort_values(by="iter", inplace=True)
    df_reward.to_csv(os.path.join(path_to_output_pre, "reward.csv"))


def cal_one_iter(path_to_output, iter_num):

    if os.path.isfile(os.path.join(path_to_output, "traj_reward_{0}.test.csv".format(iter_num))):
        file_name = os.path.join(path_to_output, "traj_reward_{0}.test.csv".format(iter_num))
    elif os.path.isfile(os.path.join(path_to_output, "traj_reward_iter{0}.test.csv".format(iter_num))):
        file_name = os.path.join(path_to_output, "traj_reward_iter{0}.test.csv".format(iter_num))

    df_reward = pd.read_csv(file_name, header=0)
    mean_r = df_reward["reward"].mean()
    sum_r = df_reward["reward"].sum()

    return mean_r, sum_r

