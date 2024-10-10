mkdir -p /youtu/youtu-public && mount -t ceph 9.22.150.83:6789,9.22.150.89:6789,9.22.145.244:6789:/group/887465/public /youtu/youtu-public -o name=group887465,secret=AQCFd41dX6IXHRAAcP1xvkHFPmflXMJPNQh1WA==
ln -s /youtu/youtu-public/fx3_students/v_kokijiang /root/


source activate
source deactivate
conda activate /root/v_kokijiang/envs/softcore

cd /root/v_kokijiang/project/SoftPatch || return

datapath=../../dataset/VisA_pytorch/1cls
datasets=($(ls -F "$datapath" | grep "/$" | sed 's|/$||'))
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '$dataset; done))

for noise in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
    for seed in 0 1 2
    do
        python main.py --data_path "$datapath" --noise $noise  "${dataset_flags[@]}" \
        --gpu $seed --seed $seed --log_project VisA --log_group softpatch-noise$noise-$seed &
    done
    wait
done


#datapath=../../MVTec
#datasets=($(ls -F "$datapath" | grep "/$" | sed 's|/$||'))
#dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '$dataset; done))
#
## 对noise取0，0.1，0.2，0.3，0.4，0.5，0.6，0.7，0.8，0.9，1.0分别训练
## 对每个noise取3次，每次取不同的seed, gpu
#for noise in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
#do
#    for seed in 0 1 2
#    do
#        python main.py --dataset mvtec --data_path "$datapath" --noise $noise  "${dataset_flags[@]}" \
#        --gpu $seed --seed $seed --log_project MVTec --log_group softpatch-noise$noise-$seed &
#    done
#    wait
#done