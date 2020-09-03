import numpy as np


def check_action_equals_next_speed(ne_obs, ne_info, action, info, obs_h):

    for i in range(len(action)):
        c_speed = action[i][0]
        v_id = info["vec_id"][i]
        try:
            m_speed = ne_obs[ne_info["vec_id"].index(v_id)][obs_h.index("speed")]
        except ValueError: # vehicle has left the lanes (either entered intersection or left the system)
            continue
        if c_speed != m_speed:
            return False

    return True


def filter_unreal_speed(n_action, n_action_est): # todo why  is this needed?

    n_action_filtered = []
    for ind_action in range(len(n_action)):

        n_action_filtered.append(np.minimum(n_action[ind_action], n_action_est[ind_action]))

    return n_action_filtered
