import numpy as np
import sys
import os
import random

# Car-following model (act like policy in RL-based simulator learning)
# Given current state (including vehicle speed and leading vehicle speed), return action (speed of in next ts)

def krauss_default(speed_max, pos_acc, neg_acc, speed_curr, speed_lead, if_leader, ts, gap):
    '''

    :param speed_max:
    :param neg_acc:
    :param pos_acc:
    :param speed_curr:
    :param speed_lead:
    :param if_lead:
    :param ts:
    :param gap:
    :param max_err:
    :return:
    '''

    max_err_ratio = 0
    rand_err_ratio = random.uniform(0, max_err_ratio)

    speed_safe = speed_lead + (gap - speed_lead * ts)/((speed_lead + speed_curr) / (2 * neg_acc) + ts)

    tmp = 100 * np.ones(len(speed_curr))
    speed_safe += tmp * (1 - if_leader)

    speed_desr = np.minimum(np.minimum(speed_max, speed_curr + pos_acc * ts), speed_safe)


    action = np.maximum(0, speed_desr - rand_err_ratio * speed_desr)

    return action

