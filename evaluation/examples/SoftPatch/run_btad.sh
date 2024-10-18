##################### BTAD
datapath=../BTAD
datasets=('01' '02' '03')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '$dataset; done))

python main.py --dataset btad --data_path ../../BTAD --noise 0  "${dataset_flags[@]}" --seed 0 \
--gpu 1 --resize 512 --imagesize 512 --sampling_ratio 0.01