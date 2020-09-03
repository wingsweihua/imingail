
if __name__=="__main__":
    import os
    import argparse
    import sys
    import gym
    import gym_citycar

    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    print(os.getcwd())
    sys.path.append('.')

    from imitation.airl.cal_utils.utils import get_action
    from imitation.base.policy_test import test_memo_scenario

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
    parser.add_argument('--max_iter_num', type=int, default=5,
                        help='maximal number of main iterations (default: 50)')
    parser.add_argument('--max_episode_len', type=int, default=500,
                        help='maximal length of episodes (default: 1000)')
    parser.add_argument('--seed', type=int, default=500,
                        help='random seed (default: 500)')
    parser.add_argument('--logdir', type=str, default='logs',
                        help='tensorboardx logs directory')
    parser.add_argument('--stat_policy', type=str, default='Beta')
    parser.add_argument('--on_policy', action="store_true")
    parser.add_argument('--on_policy_start_iter', type=int, default=0)

    parser.add_argument('--path_to_sim_conf', type=str, default="config/simulator/default.json")
    parser.add_argument('--memo', type=str, default="LA")
    parser.add_argument('--scenario', type=str, default="LA")
    parser.add_argument('--model_name', type=str, default="AIRL")
    parser.add_argument('--reward_func', type=int, default=1)
    parser.add_argument('--interpolated', type=str, default="interpolated")

    args = parser.parse_args()

    test_memo_scenario(args, get_action)