#!/usr/bin/env bash

memo="1x4"
model_name="GAIL"
interpolated="sparse"
reward_func=1

generate_traj_func(){
    python generate_trajs.py --scenario $1 --memo $2
}

run_gail_train_func(){
#    mkdir "../data/output/$2/"
    python main.py --scenario $1 --memo $2 --gamma 0 --model_name $3 --reward_func $4 --max_episode_len 300 --interpolated $4
}

run_gail_test_func(){
#    mkdir "../data/output/$2/"
    python policy_test.py --scenario $1 --memo $2 --model_name $3 --reward_func $4 --max_episode_len 300
}

cd scripts

#for file in "4x4_gudang"
#do
#    generate_traj_func ${file} ${memo} &
#done

wait

cd ../imitation/gail/

for file in "1x4_LA"
do
    run_gail_train_func ${file} ${memo} ${model_name} ${reward_func} &
done

wait

for file in "1x4_LA"
do
    run_gail_test_func ${file} ${memo} ${model_name} ${reward_func} &
done

wait

cd ../../scripts

python plot_all_training.py --memo ${memo} --model_name ${model_name}