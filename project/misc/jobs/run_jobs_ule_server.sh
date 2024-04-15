#!/bin/bash

DATA_HOME="/media/guru" # for computo04
#DATA_HOME="/home/guru" # for computo12

train_script="$HOME/git_code/pushpull-conv/project/train_flow.py"
logs_dir="$HOME/runtime_data/pushpull-conv-minmax_across_channels"
task="classification"

dataset_name="imagenet"
corrupted_dataset_name="imagenet-c"

if [ $dataset_name == "imagenet" ] || [ $dataset_name == "imagenet100" ] ||[ $dataset_name == "imagenet200" ]
then
    dataset_dir="${DATA_HOME}/datasets/imagenet"
    corrupted_dataset_dir="${DATA_HOME}/datasets/imagenet-c"
elif [ $dataset_name == "cifar10" ];
then
    dataset_dir="${DATA_HOME}/datasets/cifar"
    corrupted_dataset_dir="${DATA_HOME}/datasets/cifar10-c"
fi

# There are 96 CPU cores available on each node. Running 4 jobs in parallel ==> 24 cores for each job
# When augmentation is set to prime, unable to use more than 2 workers
num_workers=2

model="resnet50"
experiment_name="${model}_${dataset_name}_${task}_prime"
common_train_args="--augmentation prime --accelerator gpu --img_size 224 --model ${model} --hash_length 64 --quantization_weight 1e-4 --num_workers ${num_workers} --batch_size 64 --max_epochs 50 --weight_decay 5e-5 --lr_base 5e-2 --task ${task} --logs_dir ${logs_dir} --experiment_name ${experiment_name} --dataset_dir ${dataset_dir} --dataset_name ${dataset_name}"
base_dir="$logs_dir/$experiment_name"
predict_script="$HOME/git_code/pushpull-conv/project/predict_flow.py"
corruption_types="gaussian_noise shot_noise impulse_noise defocus_blur glass_blur  motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression"
baseline_model_logs_dir="$base_dir/${model}"
common_predict_args="--accelerator gpu --img_size 224 --model ${model} --num_workers ${num_workers} --corruption_types $corruption_types --task ${task} --dataset_dir ${dataset_dir} --dataset_name ${dataset_name} --corrupted_dataset_dir ${corrupted_dataset_dir} --corrupted_dataset_name ${corrupted_dataset_name}"

export CUDA_LAUNCH_BLOCKING=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

phase="train"
#i=0
#avg=-1
#mkdir -p "${base_dir}/version_${i}"
#python ${train_script} --no-use_push_pull --logs_version ${i} ${common_train_args}  --devices "$((i%2))," > "${base_dir}/version_${i}/logs_train.out" 2>&1
#experiment_dir="$base_dir/${model}"
#mv "${base_dir}/version_${i}" "${experiment_dir}"
#python ${predict_script} --no-use_push_pull --models_to_predict last --predict_model_logs_dir "${experiment_dir}" ${common_predict_args} --devices "$((i%2))," > "${experiment_dir}/logs_predict.out" 2>&1


i=1
avg=3
mkdir -p "${base_dir}/version_${i}"
python ${train_script} --avg_kernel_size ${avg} --trainable_pull_inhibition --logs_version ${i}  ${common_train_args} --devices "$((i%2))," > "${base_dir}/version_${i}/logs_train.out" 2>&1
experiment_dir="$base_dir/${model}_avg${avg}"
mv "${base_dir}/version_${i}" "${experiment_dir}"
python ${predict_script} --models_to_predict last --predict_model_logs_dir "${experiment_dir}" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir} --devices "$((i%2))," > "${experiment_dir}/logs_predict.out" 2>&1



#experiment_dir="$base_dir/${model}_avg${avg}"
#mv "${base_dir}/version_${i}" "${experiment_dir}"
#nohup python ${predict_script} --models_to_predict last --predict_model_logs_dir "${experiment_dir}" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir} --devices "0," > "${experiment_dir}/logs_predict.out" 2>&1

#i=0
#for avg in -1 3;
#do
#    # Training Phase
#    if ((phase == "train")); then
#        mkdir -p "${base_dir}/version_${i}"
#        if ((avg == -1)); then
#            nohup python ${train_script} --no-use_push_pull --logs_version ${i} ${common_train_args}  --devices "$((i%2))," > "${base_dir}/version_${i}/logs_train.out" 2>&1
#        else
#            nohup python ${train_script} --avg_kernel_size ${avg} --trainable_pull_inhibition --logs_version ${i}  ${common_train_args} --devices "$((i%2))," > "${base_dir}/version_${i}/logs_train.out" 2>&1
#        fi
#        sleep 30  # sleep for 30 seconds to avoid a clash between dataloader workers
#
#    # Evaluation Phase
#    elif ((phase == "eval")); then
#        if ((avg == -1)); then
#            experiment_dir="$base_dir/${model}"
#        else
#            experiment_dir="$base_dir/${model}_avg${avg}"
#        fi
#
#        mv "${base_dir}/version_${i}" "${experiment_dir}"
#
#        if ((avg == -1)); then
#            nohup python ${predict_script} --no-use_push_pull --models_to_predict last --predict_model_logs_dir "${experiment_dir}" ${common_predict_args} --devices "$((i%2))," > "${experiment_dir}/logs_predict.out" 2>&1
#        else
#            nohup python ${predict_script} --models_to_predict last --predict_model_logs_dir "${experiment_dir}" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir} --devices "$((i%2))," > "${experiment_dir}/logs_predict.out" 2>&1
#        fi
#    fi
#
#    i=$(( i + 1 ));
#done

echo "Submitted all jobs in phase ${phase}"

# Some useful commands

# Multi-hop SSH to connect to the dedicated compute nodes of ULE HPC Cluster
# ssh -J guru.swaroop@193.146.98.96 -p 10001 guru@computo04.unileon.hpc
# ssh -J guru.swaroop@193.146.98.96 -p 10001 guru@computo12.unileon.hpc

# Setting up local port forwarding
# ssh -o ExitOnForwardFailure=yes -fN -J guru.swaroop@193.146.98.96 -L 10004:localhost:22 -p 10001 guru@computo04.unileon.hpc
# ssh -o ExitOnForwardFailure=yes -fN -J guru.swaroop@193.146.98.96 -L 10012:localhost:22 -p 10001 guru@computo12.unileon.hpc

# View all the ports in use
# lsof -i -P -n | grep LISTEN
