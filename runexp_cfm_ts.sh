#!/usr/bin/env bash

memo="4x4"
model_name="CFM-TS"
max_iter=30

generate_traj_func(){
    python generate_trajs.py --scenario $1 --memo $2
}

run_ts_func(){
#    mkdir "../data/output/$2/"
    python main_ts.py --scenario $1 --memo $2 --max_iter $3
}

cd scripts

# for file in "hangzhou_bc_tyc_1h_7_8_1848" "hangzhou_bc_tyc_1h_8_9_2231" "hangzhou_bc_tyc_1h_10_11_2021" "hangzhou_kn_hz_1h_7_8_827" "hangzhou_sb_sx_1h_7_8_1671"
# do
#     generate_traj_func ${file} ${memo} &
# done

wait

cd ../imitation/cfm_calibration/

# for file in "hangzhou_bc_tyc_1h_7_8_1848" "hangzhou_bc_tyc_1h_8_9_2231" "hangzhou_bc_tyc_1h_10_11_2021" "hangzhou_kn_hz_1h_7_8_827" "hangzhou_sb_sx_1h_7_8_1671"
for file in  "4x4_gudang"
do
    run_ts_func ${file} ${memo} ${max_iter}&
done

