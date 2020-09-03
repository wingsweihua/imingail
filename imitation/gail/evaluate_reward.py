


if __name__ == "__main__":
    import os
    import sys

    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    print(os.getcwd())
    sys.path.append('.')

    import argparse
    from imitation.base.evaluate_reward import test_memo_scenario

    parser = argparse.ArgumentParser(description='PyTorch GAIL')
    parser.add_argument('--memo', type=str, default='1x1_minGap_1.5_headwayTime_1.0')
    parser.add_argument('--model_name', type=str, default='GAIL')
    parser.add_argument('--scenario', type=str, default='hangzhou_sb_sx_1h_7_8_1671')
    arguments = parser.parse_args()

    path_to_output = os.path.join("data", "output", arguments.memo)

    test_memo_scenario(arguments)

