#!/usr/bin/env bash

memo="1x1"
model_name="BC"
batch_size=64
hidden_size=40
num_epochs=5
interpolated="interpolated"

generate_traj_func(){
    python generate_trajs.py --scenario $1 --memo $2
}

interpolate_traj_func(){
    echo interpolate/main --scenario $1 --memo $2
    python main.py --scenario $1 --memo $2
}

run_bc_func(){
#    mkdir "../data/output/$2/"
    echo imiation/bc/main -scenario $1 --memo $2 --batch_size $3 --hidden_size $4 --num_epochs $5 --interpolated $6
    python main.py --scenario $1 --memo $2 --batch_size $3 --hidden_size $4 --num_epochs $5 --interpolated $6
}

cd scripts

#for scenario in "hangzhou_bc_tyc_1h_7_8_1848" "hangzhou_bc_tyc_1h_8_9_2231" "hangzhou_bc_tyc_1h_10_11_2021" "hangzhou_kn_hz_1h_7_8_827" "hangzhou_sb_sx_1h_7_8_1671"
#for scenario in "hangzhou_kn_hz_1h_7_8_827"
#do
#    generate_traj_func ${scenario} ${memo} &
#done

wait

cd ../interpolation/
for scenario in "hangzhou_kn_hz_1h_7_8_827"
do
    interpolate_traj_func ${scenario} ${memo}&
done

wait

cd ../imitation/bc/

#for scenario in "hangzhou_bc_tyc_1h_7_8_1848" "hangzhou_bc_tyc_1h_8_9_2231" "hangzhou_bc_tyc_1h_10_11_2021" "hangzhou_kn_hz_1h_7_8_827" "hangzhou_sb_sx_1h_7_8_1671"
for scenario in "hangzhou_kn_hz_1h_7_8_827"
do
    run_bc_func ${scenario} ${memo} ${batch_size} ${hidden_size} ${num_epochs} ${interpolated}&
done

