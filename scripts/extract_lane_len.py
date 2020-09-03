import json
from math import sqrt


def cal_length(p1, p2):

    return sqrt(pow((p1["x"] - p2["x"]), 2) + pow((p1["y"] - p2["y"]), 2))/1000


# for scenario in ["hangzhou_bc_tyc_1h_7_8_1848", "hangzhou_bc_tyc_1h_8_9_2231", "hangzhou_bc_tyc_1h_10_11_2021",
#                  "hangzhou_kn_hz_1h_7_8_827", "hangzhou_sb_sx_1h_7_8_1671", "4x4_gudang"]:
for scenario in ["hangzhou_kn_hz_1h_7_8_827"]:

    path_to_roadnet = "../data/{}/roadnet.json".format(scenario)

    dic_len = {}

    dic_road = json.load(open(path_to_roadnet, "r"))
    dic_road["roads"]

    for road in dic_road["roads"]:
        dic_len[road["id"]] = cal_length(road["points"][0], road["points"][1])
    json.dump(dic_len, open("../data/{}/roadnet_len.json".format(scenario), "w"), indent=2)
