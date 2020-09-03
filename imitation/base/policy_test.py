import os
import gym
import pickle
import json
import torch
from utils.log_utils import Stack
from utils.check_utils import check_action_equals_next_speed, filter_unreal_speed
from utils.traj_utils import extract_sparse_from_dense
from imitation.base import evaluate
import pandas as pd
import numpy as np
from multiprocessing import Process


def test_memo_scenario(args, get_action):
    # ============== check train output and model ================

    path_to_output_pre = os.path.join("data", "output", args.memo, args.model_name)
    if not os.path.exists(path_to_output_pre):
        print("not well trained")
        return
    else:
        list_output_dirs = os.listdir(path_to_output_pre)


    path_to_save_model_pre = os.path.join("saved_model", args.memo, args.model_name)
    if not os.path.exists(path_to_save_model_pre):
        print("not well trained")
        return
    else:
        list_model_dirs = os.listdir(path_to_save_model_pre)

    list_dirs = list(set(list_model_dirs).intersection(set(list_output_dirs)))

    for d in list_dirs:
        if d.endswith(args.scenario):
            test_all_iters(path_to_output_pre, path_to_save_model_pre, d, args.scenario, get_action, args)

def test_all_iters(path_to_output_pre, path_to_save_model_pre, run_ts, scenario, get_action, args):
    # ============== check train output and model ================

    list_model_files = os.listdir(os.path.join(path_to_save_model_pre, run_ts))

    for file in list_model_files:
        iter_num = file.split('.')[0].split('_')[-1]
        p = Process(target = test_one_iter,
                    args=(os.path.join(path_to_output_pre, run_ts),
                os.path.join(path_to_save_model_pre, run_ts),
                iter_num,
                scenario,
                get_action,
                args))
        p.start()
        p.join()


        # test_one_iter(
        #     os.path.join(path_to_output_pre, run_ts),
        #     os.path.join(path_to_save_model_pre, run_ts),
        #     iter_num,
        #     scenario,
        #     get_action,
        #     args
        # )


    return


def test_one_iter(path_to_output, path_to_save_model, iter_num, scenario, get_action, args):

    output_pkl = os.path.join(path_to_output, "traj_raw_{}.test.pkl".format(iter_num))
    output_csv = os.path.join(path_to_output, "traj_raw_{}.test.csv".format(iter_num))
    output_sparse_pkl = os.path.join(path_to_output, "traj_sparse_{}.test.pkl".format(iter_num))
    output_sparse_csv = os.path.join(path_to_output, "traj_sparse_{}.test.csv".format(iter_num))
    expert_dense_pkl = os.path.join("data", "expert_trajs", args.memo, args.scenario, "traj_raw.pkl")
    expert_sparse_pkl = os.path.join("data", "expert_trajs", args.memo, args.scenario, "traj_sparse.pkl")
    conf_file = os.path.join(path_to_output, "{}.json".format(scenario))
    output_reward_pkl = os.path.join(path_to_output, "traj_reward_{}.test.pkl".format(iter_num))
    output_reward_csv = os.path.join(path_to_output, "traj_reward_{}.test.csv".format(iter_num))

    if os.path.exists(output_csv):
        print("already tested", output_csv)
        return
    else:
        print("Testing ", output_csv)

    # ============== initial env =====================
    env = gym.make(args.env_name,
                   normalize=True,
                   path_to_conf_file=conf_file,
                   list_vars_to_subscribe=["interval", "speed", "pos_in_lane", "lane_max_speed", "if_exit_lane", "dist_to_signal", "phase", "if_leader", "leader_speed", "dist_to_leader",],
                   reward_function=args.reward_func)
    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]
    obs_header = env.observation_header
    action_header = env.action_header

    print('state size:', num_inputs)
    print('action size:', num_actions)

    # ============== initial network =====================
    torch.manual_seed(args.seed)

    ckpt = torch.load(os.path.join(path_to_save_model, "ckpt_{}.pth.tar".format(iter_num)))

    actor = ckpt["actor"]
    critic = ckpt["critic"]
    discrim = ckpt["discrim"]

    print("Loaded OK ")

    # ============== start looping experiment ========

    list_traj = Stack(obs_header, action_header)

    list_reward = []

    actor.eval(), critic.eval(), discrim.eval()

    new_samples = 0

    n_obs, n_info = env.reset()
    n_action = []  # default value

    for t in range(args.max_episode_len):

        if t % 100 == 0:
            print("t: {0}".format(t))

        if len(n_obs) == 0:

            next_n_obs, n_reward, n_done, next_n_info = env.step(n_action, n_info)

        else:
            # follow policy to generate samples

            # n_obs = running_state(n_obs)
            dist_args = actor(torch.Tensor(n_obs))
            n_action = get_action(dist_args, args)

            n_action = filter_unreal_speed(n_action, n_info["next_speed_est"])

            next_n_obs, n_reward, n_done, next_n_info = env.step(n_action, n_info)
            n_mask = [1-int(d) for d in n_done]

            # assert check_action_equals_next_speed(next_n_obs, next_n_info, n_action, n_info, obs_header)

            list_traj.append(
                [n_info["vec_id"], n_info["current_time"], n_info["lane_id"], n_obs, n_action, n_reward, np.zeros_like(n_reward).tolist()])
            list_reward += n_reward
        n_obs = next_n_obs
        n_info = next_n_info

        new_samples += len(n_obs)

    # ============== post process and dump data =======
    df_reward = pd.DataFrame(np.array([list_reward]).transpose(), columns=["reward"])
    pickle.dump(df_reward, open(output_reward_pkl, "wb"))
    df_reward.to_csv(output_reward_csv)


    # full trajectory -- for comparison: dataframe
    df = list_traj.convert_to_df()
    pickle.dump(df, open(output_pkl, "wb"))
    df.to_csv(output_csv)
    print("================================================================")
    print("********* full trajectory logged")
    print("================================================================")

    dic_roadnet_len = json.load(open(os.path.join("data", args.scenario, "roadnet_len.json"), "r"))

    evaluate.evaluate(expert_dense_pkl, output_pkl, dic_roadnet_len, args.max_episode_len)

    # sparse trajectory -- for comparison: dataframe
    sparse_df = extract_sparse_from_dense(df)
    pickle.dump(sparse_df, open(output_sparse_pkl, "wb"))
    sparse_df.to_csv(output_sparse_csv)
    print("================================================================")
    print("********* sparse trajectory logged")
    print("================================================================")

    dic_roadnet_len = json.load(open(os.path.join("data", args.scenario, "roadnet_len.json"), "r"))

    evaluate.evaluate(expert_sparse_pkl, output_sparse_pkl, dic_roadnet_len, args.max_episode_len, sparse=True)
    # pay attention to the length of the lane

