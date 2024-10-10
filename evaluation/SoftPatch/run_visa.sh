datapath=../../dataset/VisA_pytorch/1cls
datasets=($(ls $datapath))
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '$dataset; done))

python main.py --data_path "$datapath" --noise 0.1  "${dataset_flags[@]}" \
--gpu 0 --log_project VisA
