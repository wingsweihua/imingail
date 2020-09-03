import pandas as pd
import numpy as np


def extract_abs_pos(state_action_data, len_lane):

    '''
    Extract absolute position information from current trajectory dataframe
    :param state_action_data: DataFrame - Current trajectory data
    :param len_lane: dictionary - lane_id: length
    :return: traj_df: DataFrame - Contains a column named "abs_pos"
    '''

    list_name = ['vec_id', 'ts', 'lane_id', 'pos_in_lane', 'speed']
    traj_df = state_action_data[list_name]
    traj = traj_df.values

    pos_abs = []
    lane_mark = []
    lane_mark_last = 0
    vec_id_last = state_action_data['vec_id'].values[0]
    lane_id_last = state_action_data['lane_id'].values[0]

    for ind, val in enumerate(traj):
        vec_id = val[0]
        lane_id = val[2]

        if vec_id == vec_id_last and lane_id == lane_id_last:
            lane_mark_cur = lane_mark_last
        elif vec_id == vec_id_last and lane_id != lane_id_last:
            lane_mark_cur = lane_mark_last + 1
        elif vec_id != vec_id_last:
            lane_mark_cur = 0
        lane_mark.append(lane_mark_cur)
        link_id = lane_id[0:10] # TODO : Careful, hard code the link id from lane id: 'road_2_1_2_1' -> 'road_2_1_2'
        pos_abs_cur = lane_mark_cur * len_lane[link_id] + val[3]
        pos_abs.append(pos_abs_cur)

        vec_id_last = vec_id
        lane_id_last = lane_id
        lane_mark_last = lane_mark_cur

    lane_mark = np.array(lane_mark)
    pos_abs = np.array(pos_abs)
    pos_abs = pos_abs.reshape(pos_abs.shape[0], 1)
    lane_mark = lane_mark.reshape(lane_mark.shape[0], 1)
    traj = np.hstack((traj, pos_abs))
    traj = np.hstack((traj, lane_mark))
    traj = np.delete(traj, [3], axis=1)
    traj_df = pd.DataFrame(traj, columns=['vec_id', 'ts', 'lane_id', 'speed', 'abs_pos', 'lane_order'])

    return traj_df


def traj_by_time(traj, name_ts, name_vec_id):

    '''
    Reorder trajectory data by time stamp
    :param traj: DataFrame - Trajectory data
    :param name_ts: str - Header of timestamp column
    :param name_vec_id: str - Header of vehicle ID column
    :return: traj_ts: DataFrame - Reordered trajectory
    '''

    traj.sort_values(by=[name_ts, name_vec_id], axis=0, inplace=True)
    cols = list(traj)
    cols.insert(0, cols.pop(cols.index(name_ts)))
    traj_ts = traj[cols]

    return traj_ts


def rmse_eval(traj_exp, traj_lrn, len_lane, max_episode_len):

    '''
    Calculate Root Mean Square Error (RMSE) of learned trajectory from expert trajectory.
    :param traj_exp: DataFrame - Expert trajectory
    :param traj_lrn: DataFrame - Learned trajectory
    :param len_lane: Dictionary - lane_id: length
    :return:
    '''

    dict_exp = dict()
    dict_lrn = dict()

    traj_exp.rename(columns={'speed': 'speed_exp', 'abs_pos': 'pos_exp', 'lane_id': 'lane_id_exp'}, inplace=True)
    traj_lrn.rename(columns={'speed': 'speed_lrn', 'abs_pos': 'pos_lrn', 'lane_id': 'lane_id_lrn'}, inplace=True)
    traj_overall = pd.merge(traj_exp, traj_lrn, how='outer', on=['ts', 'vec_id'])
    traj_overall = traj_by_time(traj_overall, 'ts', 'vec_id')

    rmse_pos = list()
    rmse_speed = list()
    vec_exp = list()
    vec_lrn = list()

    speedAVE_exp = list()
    speedAVE_lrn = list()

    for ts in range(max_episode_len):

        traj_cur = traj_overall[traj_overall.ts == ts]

        if traj_cur.values.shape[0] != 0:

            vec_exp_list = traj_exp[traj_exp.ts == ts]['vec_id'].values
            lane_exp_list = traj_exp[traj_exp.ts == ts]['lane_order'].values
            vec_lrn_list = traj_lrn[traj_lrn.ts == ts]['vec_id'].values
            lane_lrn_list = traj_lrn[traj_lrn.ts == ts]['lane_order'].values

            m = len(set(vec_exp_list)) # number of exp vehicles moving at current ts
            vec_exp.append(len(set(vec_exp_list)))
            vec_lrn.append(len(set(vec_lrn_list)))

            sqrErrList_pos = list()
            sqrErrList_speed = list()

            speedSUM_exp = 0
            speedSUM_lrn = 0

            for ind, row in traj_cur.iterrows():

                if row['pos_exp'] is np.NAN and row['pos_lrn'] is not np.NAN:
                    pos_lrn = row['pos_lrn']
                    if row['vec_id'] in dict_exp:
                        lane_id_exp = row['lane_id_lrn']
                        link_id_exp = lane_id_exp[0:10]
                        pos_exp = len_lane[link_id_exp] * (dict_exp[row['vec_id']] + 1)
                    else:
                        pos_exp = 0
                    speed_exp = 0
                    speed_lrn = row['speed_lrn']

                elif row['pos_exp'] is not np.NAN and row['pos_lrn'] is np.NAN:
                    pos_exp = row['pos_exp']
                    if row['vec_id'] in dict_lrn:
                        lane_id_lrn = row['lane_id_exp']
                        link_id_lrn = lane_id_lrn[0:10]
                        pos_lrn = len_lane[link_id_lrn] * (dict_lrn[row['vec_id']] + 1)
                    else:
                        pos_lrn = 0
                    speed_exp = row['speed_exp']
                    speed_lrn = 0

                else:
                    pos_exp = row['pos_exp']
                    pos_lrn = row['pos_lrn']
                    speed_exp = row['speed_exp']
                    speed_lrn = row['speed_lrn']

                sqrErrList_pos.append((pos_exp - pos_lrn)**2)
                sqrErrList_speed.append((speed_exp - speed_lrn)**2)

                speedSUM_exp = speedSUM_exp + speed_exp
                speedSUM_lrn = speedSUM_lrn + speed_lrn

            dict_exp.update(dict(zip(vec_exp_list, lane_exp_list)))
            dict_lrn.update(dict(zip(vec_lrn_list, lane_lrn_list)))

            if m == 0:
                speedAVE_exp.append(0)
                speedAVE_lrn.append(0)
                rmse_pos.append(0)
                rmse_speed.append(0)
            else:
                speedAVE_exp.append(speedSUM_exp/m)
                speedAVE_lrn.append(speedSUM_lrn/m)
                rmse_pos.append((np.array(sqrErrList_pos).sum() / m) ** 0.5)
                rmse_speed.append((np.array(sqrErrList_speed).sum() / m) ** 0.5)

        else:
            rmse_pos.append(0)
            rmse_speed.append(0)
            vec_exp.append(0)
            vec_lrn.append(0)

            speedAVE_exp.append(0)
            speedAVE_lrn.append(0)

    return rmse_pos, rmse_speed, vec_exp, vec_lrn, dict_exp, dict_lrn, speedAVE_exp, speedAVE_lrn

def rmse_eval_time(traj_exp, traj_lrn):

    '''
    Calculate Root Mean Square Error (RMSE) of learned trajectory from expert trajectory.
    :param traj_exp: DataFrame - Expert trajectory
    :param traj_lrn: DataFrame - Learned trajectory
    :param len_lane: Dictionary - lane_id: length
    :return:
    '''
    traj_exp.rename(columns={'ts': 'ts_exp', 'speed': 'speed_exp', 'abs_pos': 'pos_exp', 'lane_id': 'lane_id_exp'}, inplace=True)
    traj_lrn.rename(columns={'ts': 'ts_lrn', 'speed': 'speed_lrn', 'abs_pos': 'pos_lrn', 'lane_id': 'lane_id_lrn'}, inplace=True)
    traj_exp_group = traj_exp.groupby('vec_id')['ts_exp'].apply(list).to_dict()
    traj_lrn_group = traj_lrn.groupby('vec_id')['ts_lrn'].apply(list).to_dict()

    traj_overall_vec_id = []
    traj_overall_delta_time = []
    for key, value in enumerate(traj_exp_group):

        traj_overall_vec_id.append(value)

        time_list_exp = np.sort(np.array(traj_exp_group[value]))
        if value in traj_lrn_group.keys():
            time_list_lrn = np.sort(np.array(traj_lrn_group[value]))
        else:
            time_list_lrn = np.array([time_list_exp[0]])
        max_len = max(len(time_list_exp), len(time_list_lrn))
        traj_overall_delta_time_vehicle = []
        for i in range(max_len):
            if i < len(time_list_exp) and i < len(time_list_lrn):
                traj_overall_delta_time_vehicle.append((time_list_exp[i] - time_list_lrn[i])**2)
            elif i < len(time_list_exp) and i >= len(time_list_lrn):
                traj_overall_delta_time_vehicle.append(
                    (time_list_exp[i] - time_list_lrn[len(time_list_lrn)-1]) ** 2
                )
            else:
                traj_overall_delta_time_vehicle.append(
                    (time_list_exp[len(time_list_exp)-1] - time_list_lrn[i]) ** 2
                )
        traj_overall_delta_time.append((np.array(traj_overall_delta_time_vehicle).mean())**0.5)


    return traj_overall_delta_time

def rmse_eval_time_dense(traj_exp, traj_lrn):

    '''
    Calculate Root Mean Square Error (RMSE) of learned trajectory from expert trajectory.
    :param traj_exp: DataFrame - Expert trajectory
    :param traj_lrn: DataFrame - Learned trajectory
    :param len_lane: Dictionary - lane_id: length
    :return:
    '''
    traj_exp.rename(columns={'ts': 'ts_exp', 'speed': 'speed_exp', 'abs_pos': 'pos_exp', 'lane_id': 'lane_id_exp'}, inplace=True)
    traj_lrn.rename(columns={'ts': 'ts_lrn', 'speed': 'speed_lrn', 'abs_pos': 'pos_lrn', 'lane_id': 'lane_id_lrn'}, inplace=True)
    traj_exp_group = traj_exp.groupby('vec_id')['ts_exp'].apply(list).to_dict()
    traj_lrn_group = traj_lrn.groupby('vec_id')['ts_lrn'].apply(list).to_dict()

    traj_overall_vec_id = []
    traj_overall_delta_time = []
    for key, value in enumerate(traj_exp_group):

        traj_overall_vec_id.append(value)

        time_list_exp = np.sort(np.array(traj_exp_group[value]))
        if value in traj_lrn_group.keys():
            time_list_lrn = np.sort(np.array(traj_lrn_group[value]))
        else:
            time_list_lrn = np.array([time_list_exp[0]])
        traj_overall_delta_time_vehicle = (time_list_exp.max() - time_list_exp.min()) - \
                                           (time_list_lrn.max() - time_list_lrn.min())
        traj_overall_delta_time.append(abs(traj_overall_delta_time_vehicle))


    return np.sqrt((np.array(traj_overall_delta_time)**2).mean())

def get_rmse(traj_exp, traj_lrn, len_lane, max_episode_len):

    traj_exp = extract_abs_pos(traj_exp, len_lane=len_lane)
    traj_exp = traj_by_time(traj_exp, name_ts='ts', name_vec_id='vec_id')

    traj_lrn = extract_abs_pos(traj_lrn, len_lane=len_lane)
    traj_lrn = traj_by_time(traj_lrn, name_ts='ts', name_vec_id='vec_id')

    rmse_pos, rmse_speed, \
    num_vec, num_lrn, \
    dict_exp, dict_lrn, \
    speedAVE_exp, speedAVE_lrn = rmse_eval(traj_exp, traj_lrn, len_lane, max_episode_len)

    rmse_pos = np.array(rmse_pos).mean()
    rmse_speed = np.array(rmse_speed).mean()
    num_exp = len(dict_exp)
    num_lrn = len(dict_lrn)
    aveSpeed_exp = np.array(speedAVE_exp).mean()
    aveSpeed_lrn = np.array(speedAVE_lrn).mean()

    return rmse_pos, rmse_speed, aveSpeed_exp, aveSpeed_lrn, num_exp, num_lrn


def get_reward_from_traj(traj, max_episode_len):

    vecList = list(np.unique(traj['vec_id'].values))
    rewardMeanByVec = []
    rewardMeanByTs = []

    for vec in vecList:
        rewardArr = traj[traj.vec_id == vec]['reward'].values
        rewardMeanByVec.append(np.mean(rewardArr))

    for timestamp in range(int(max_episode_len)):
        rewardArr = traj[traj.ts == timestamp]['reward'].values
        if len(list(rewardArr)) == 0:
            rewardMeanByTs.append(0)
        else:
            rewardMeanByTs.append(np.mean(rewardArr))

    return rewardMeanByVec, rewardMeanByTs


def get_pos_from_traj(traj, max_episode_len, len_lane):

    traj = extract_abs_pos(traj, len_lane)
    meanPos = []

    for timestamp in range(int(max_episode_len)):
        rewardArr = traj[traj.ts == timestamp]['abs_pos'].values
        if len(list(rewardArr)) == 0:
            meanPos.append(0)
        else:
            meanPos.append(np.mean(rewardArr)*1000)

    return meanPos



def get_single_vehicle_traj(traj, max_episode_len, len_lane, vec_id):

    traj = extract_abs_pos(traj, len_lane)
    single_traj = traj[traj.vec_id == vec_id]
    timeStamp = list(single_traj['ts'].values)
    speed = single_traj['speed'].values
    pos = single_traj['abs_pos'].values

    speedArr = np.zeros(max_episode_len)
    posArr = np.zeros(max_episode_len)

    for i in range(max_episode_len):

        if i > max(timeStamp):
            posArr[i] = np.max(pos)
        elif min(timeStamp) <= i <= max(timeStamp):
            if i in timeStamp:
                idx = timeStamp.index(i)
                speedArr[i] = speed[idx]
                posArr[i] = pos[idx]
                i_last = i
            else:
                idx = timeStamp.index(i_last)
                speedArr[i] = speed[idx]
                posArr[i] = pos[idx]

    return speedArr, posArr, int(min(timeStamp)), int(max(timeStamp))


# def get_rmse_by_vec(traj_exp, traj_lrn, len_lane, max_episode_len):
#
#     traj_exp = extract_abs_pos(traj_exp, len_lane=len_lane)
#     traj_lrn = extract_abs_pos(traj_lrn, len_lane=len_lane)
#
#     vecListExp = list(np.unique(traj_exp['vec_id'].values))
#     vecListLrn = list(np.unique(traj_lrn['vec_id'].values))
#     vecList = list(set(vecListExp) | set(vecListLrn))
#
#     for vec in vecList:
#
#         timeExp = traj_exp[traj_exp.vec_id == vec]['ts'].values
#         speedExp = traj_exp[traj_exp.vec_id == vec]['speed'].values
#         posExp = traj_exp[traj_exp.vec_id == vec]['abs_pos'].values
#
#         timeLrn = traj_lrn[traj_lrn.vec_id == vec]['ts'].values
#         speedLrn = traj_lrn[traj_lrn.vec_id == vec]['speed'].values
#         posLrn = traj_lrn[traj_lrn.vec_id == vec]['abs_pos'].values
#
#         speedDictExp = dict(zip(timeExp, speedExp))
#         speedDictLrn = dict(zip(timeLrn, speedLrn))
#
#         start = min(timeExp[0], timeLrn[0])
#         end = max(timeExp[-1], timeLrn[-1])
#
#         time = list(range(int(start), int(end + 1)))
#
#     return rmse
#

def extract_sparse_from_dense(traj_dense):
    df = traj_dense
    df.sort_values(by=["vec_id", "ts"], axis=0, inplace=True)
    df.reset_index(drop=True, inplace=True)

    list_end_traj_ind = np.where(
        df["vec_id"][:len(df) - 1].values != df["vec_id"][1:len(df)].values
    )[0].tolist() + [len(df) - 1]
    list_start_traj_ind = [-1] + np.where(
        df["vec_id"][:len(df) - 1].values != df["vec_id"][1:len(df)].values
    )[0].tolist()
    list_start_traj_ind = np.array(list_start_traj_ind) + 1
    list_start_traj_ind = list_start_traj_ind.tolist()
    list_discontinuous_ind = \
        np.where(df["ts"][:len(df) - 1].values + df["interval"][:len(df) - 1].values != df["ts"][1:len(df)].values)[
            0].tolist()
    list_rows_start_end = np.sort(np.unique(list_end_traj_ind + list_discontinuous_ind + list_start_traj_ind))
    traj_sparse = df.iloc[list_rows_start_end.tolist()]
    print("********* drop {0}/{1} rows for sparse samples".format(df.shape[0] - len(list_rows_start_end), df.shape[0]))
    return traj_sparse
