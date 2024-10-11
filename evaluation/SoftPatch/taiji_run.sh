datapath=../../MVTec
datasets=($(ls $datapath))
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '$dataset; done))

# 对noise取0，0.1，0.2，0.3，0.4，0.5，0.6，0.7，0.8，0.9，1.0分别训练
# 对每个noise取3次，每次取不同的seed, gpu
for noise in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
    for seed in 0 1 2
    do
        python main.py --dataset mvtec --data_path $datapath --noise $noise  "${dataset_flags[@]}" \
        --gpu $seed --seed $seed --log_project MVTec --log_group softpatch-noise$noise &
    done
    wait
done

