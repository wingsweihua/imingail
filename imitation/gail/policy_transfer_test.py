
if __name__=="__main__":
    import os
    import argparse
    import sys

    import gym
    import gym_citycar

    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    print(os.getcwd())
    sys.path.append('.')

    from imitation.gail.cal_utils.utils import get_action, get_reward
    from imitation.gail.train_model import train_actor_critic
    from imitation.gail.model import Actor, Critic
    from imitation.base.policy_transfer_test import test_memo_scenario

    parser = argparse.ArgumentParser(description='PyTorch GAIL')
    parser.add_argument('--env_name', type=str, default="gym_citycar-v0",
                        help='name of the environment to run')
    parser.add_argument('--load_model', type=str, default=None,
                        help='path to load the saved model')
    parser.add_argument('--render', action="store_true", default=False,
                        help='if you dont want to render, set this to False')
    parser.add_argument('--gamma', type=float, default=0,
                        help='discounted factor (default: 0.99)')
    parser.add_argument('--lamda', type=float, default=0.98,
                        help='GAE hyper-parameter (default: 0.98)')
    parser.add_argument('--hidden_size', type=int, default=32,
                        help='hidden unit size of actor, critic and discrim networks (default: 100)')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='learning rate of models (default: 3e-4)')
    parser.add_argument('--l2_rate', type=float, default=1e-3,
                        help='l2 regularizer coefficient (default: 1e-3)')
    parser.add_argument('--clip_param', type=float, default=0.2,
                        help='clipping parameter for PPO (default: 0.2)')
    parser.add_argument('--discrim_update_num', type=int, default=1,
                        help='update number of discriminator (default: 2)')
    parser.add_argument('--actor_critic_update_num', type=int, default=1,
                        help='update number of actor-critic (default: 10)')
    parser.add_argument('--total_sample_size', type=int, default=2000,
                        help='total sample size to collect before PPO update (default: 2048)')
    parser.add_argument('--refresh_sample_size', type=int, default=10,
                        help='total sample size to collect before PPO update (default: 2048)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size to update (default: 64)')
    parser.add_argument('--suspend_accu_exp', type=float, default=0.8,
                        help='accuracy for suspending discriminator about expert data (default: 0.8)')
    parser.add_argument('--suspend_accu_gen', type=float, default=0.8,
                        help='accuracy for suspending discriminator about generated data (default: 0.8)')
    parser.add_argument('--max_iter_num', type=int, default=30,
                        help='maximal number of main iterations (default: 50)')
    parser.add_argument('--max_episode_len', type=int, default=500,
                        help='maximal length of episodes (default: 1000)')
    parser.add_argument('--average_running_returns', action="store_true")
    parser.add_argument('--seed', type=int, default=500,
                        help='random seed (default: 500)')
    parser.add_argument('--logdir', type=str, default='logs',
                        help='tensorboardx logs directory')
    parser.add_argument('--stat_policy', type=str, default='Beta')
    parser.add_argument('--on_policy', action="store_true")
    parser.add_argument('--on_policy_start_iter', type=int, default=0)

    parser.add_argument('--path_to_sim_conf', type=str, default="config/simulator/default.json")
    parser.add_argument('--memo', type=str, default="1x1")
    parser.add_argument('--scenario', type=str, default="hangzhou_sb_sx_1h_7_8_1671")
    parser.add_argument('--model_name', type=str, default="GAIL")
    parser.add_argument('--dynamics', type=str, default="minGap_1.5_headwayTime_1.0")
    parser.add_argument('--retrain_policy', action="store_true", default=False)

    parser.add_argument('--reward_func', type=int, default=0)
    parser.add_argument('--extra_memo', type=str, default="default")

    parser.add_argument('--select_iter_num', type=str, default="iter20")
    parser.add_argument('--ts_only', type=str, default="2019-06-01-13-17-33")

    args = parser.parse_args()

    test_memo_scenario(args, get_action, get_reward, train_actor_critic, class_actor=Actor, class_critic=Critic, iter_num=args.select_iter_num)