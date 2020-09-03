#!/usr/bin/env bash

memo="1x4"
model_name="GAIL"
interpolated="interpolated"

generate_traj_func(){
    python generate_trajs.py --scenario $1 --memo $2
}

run_gail_train_func(){
#    mkdir "../data/output/$2/"
    python main.py --scenario $1 --memo $2 --gamma 0 --model_name $3 --interpolated $4 --max_episode_len 300
}

run_gail_test_func(){
#    mkdir "../data/output/$2/"
    python policy_test.py --scenario $1 --memo $2 --model_name $3 --max_episode_len 300
}

cd scripts

# for file in "hangzhou_bc_tyc_1h_7_8_1848" "hangzhou_bc_tyc_1h_8_9_2231" "hangzhou_bc_tyc_1h_10_11_2021" "hangzhou_kn_hz_1h_7_8_827" "hangzhou_sb_sx_1h_7_8_1671"
#for file in "hangzhou_kn_hz_1h_7_8_827"
#do
#    generate_traj_func ${file} ${memo} &
#done

wait

cd ../imitation/gail/

for file in "1x4_LA"
do
    run_gail_train_func ${file} ${memo} ${model_name} ${interpolated}
done

wait

for file in "1x4_LA"
do
    run_gail_test_func ${file} ${memo} ${model_name}
done

wait

cd ../../scripts

python plot_all_training.py --memo ${memo} --model_name ${model_name}