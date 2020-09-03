
import numpy as np
import pandas as pd


class Memory:

    def __init__(self, list_headers):

        self.stack = []
        self.headers = list_headers
        for _ in self.headers:
            self.stack.append([])
        return

    def append(self, n_sample):

        for ind in range(len(self.headers)):
            self.stack[ind] += n_sample[ind]


class Stack:

    def __init__(self, obs_header, action_header):

        self.stack = []
        self.headers = [["vec_id"], ["ts"], ["lane_id"], obs_header, action_header, ["reward"], ["irl_reward"]]
        for _ in self.headers:
            self.stack.append([])
        return

    def append(self, n_sample):

        for ind in range(len(self.headers)):
            self.stack[ind] += n_sample[ind]

    def convert_to_df(self):
        df = pd.DataFrame(
            np.concatenate(
                [np.array([self.stack[0]]).transpose(),
                 np.array([self.stack[1]]).transpose(),
                 np.array([self.stack[2]]).transpose(),
                 np.array(self.stack[3]),
                 np.array(self.stack[4]),
                 np.array([self.stack[5]]).transpose(),
                 np.array([self.stack[6]]).transpose()], axis=1),
            columns=self.headers[0] + self.headers[1] + self.headers[2] + self.headers[3] + self.headers[4] + self.headers[5] + self.headers[6]
        )
        for na in df.columns.tolist():
            if na != "vec_id" and na != "lane_id":
                df[na] = pd.to_numeric(df[na])

        df.sort_values(by=["vec_id", "ts"], axis=0, inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df

def convert_trajs_to_array(trajs, obs_header, action_header):

    return trajs[obs_header + action_header].values

def convert_trajs_to_array_interpolate(trajs, obs_header, action_header):
    trajs_grouped = trajs.groupby(by=['vec_id', 'lane_id'])
    original_header = obs_header + action_header
    interpolated_header = [header+"_s" for header in original_header] + \
                          [header+"_e" for header in original_header] + ['ts']
    array_interpolate = []
    array_label = []
    for group_name, grouped in trajs_grouped:
        # iterate every line for delta = 0 samples
        for row in grouped.iterrows():
            array_row = row[1][original_header].values.tolist() + \
                        row[1][original_header].values.tolist() + [0.0]
            array_interpolate.append(array_row)
            array_label.append(row[1][original_header])
        # iterate every adjacent line for delta != 0 samples
        if len(grouped) > 1:
            row_count = 0
            for row in grouped.iterrows():
                if row_count == 0:
                    prev_row = row
                    row_count += 1
                    continue
                else:
                    array_row = grouped.iloc[0][original_header].values.tolist() + \
                                grouped.iloc[-1][original_header].values.tolist() + [row[1]['ts']-prev_row[1]['ts']]
                    array_interpolate.append(array_row)
                    array_label.append(row[1][original_header])
                    prev_row = row
                    row_count += 1
    array_interpolate = np.array(array_interpolate)
    array_interpolate = pd.DataFrame(array_interpolate, columns=interpolated_header)
    array_label = np.array(array_label)
    array_label = pd.DataFrame(array_label, columns=original_header)
    return [array_interpolate, array_label]