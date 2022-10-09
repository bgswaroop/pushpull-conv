#!/bin/bash

DATA_HOME="/media/guru" # for computo04
#DATA_HOME="/home/guru" # for computo12

train_script="$HOME/git_code/pushpull-conv/project/train_flow.py"
logs_dir="$HOME/runtime_data/pushpull-conv"
task="classification"
dataset_name="imagenet100"
dataset_dir="${DATA_HOME}/datasets/imagenet"
corrupted_dataset_dir="${DATA_HOME}/datasets/imagenet/imagenet-c"
corrupted_dataset_name="imagenet100-c"
model="resnet18"
experiment_name="${model}_${dataset_name}_${task}"
common_train_args="--img_size 224 --model ${model} --hash_length 64 --quantization_weight 1e-4 --num_workers 12 --task ${task} --logs_dir ${logs_dir} --experiment_name ${experiment_name} --dataset_dir ${dataset_dir} --dataset_name ${dataset_name}"
base_dir="$logs_dir/$experiment_name"
predict_script="$HOME/git_code/pushpull-conv/project/predict_flow.py"
corruption_types="gaussian_noise shot_noise impulse_noise defocus_blur glass_blur  motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression"
baseline_model_logs_dir="$base_dir/${model}"
common_predict_args="--img_size 224 --model ${model} --num_workers 8 --corruption_types $corruption_types --task ${task} --dataset_dir ${dataset_dir} --dataset_name ${dataset_name} --corrupted_dataset_dir ${corrupted_dataset_dir} --corrupted_dataset_name ${corrupted_dataset_name}"

#mkdir -p ${base_dir}/version_0
##mkdir -p ${base_dir}/version_1
#mkdir -p ${base_dir}/version_2
##mkdir -p ${base_dir}/version_3
#mkdir -p ${base_dir}/version_4
##mkdir -p ${base_dir}/version_5
#mkdir -p ${base_dir}/version_6
#
#nohup python ${train_script} --no-use_push_pull --logs_version 0 ${common_train_args} --accelerator "gpu" --devices "0," > "${base_dir}/version_0/logs_train.out" 2>&1 &
##nohup python ${train_script} --push_kernel_size 7 --pull_kernel_size 7 --avg_kernel_size 3 --pull_inhibition_strength 4 --pupu_weight 0.0 --logs_version 1 ${common_train_args} --accelerator "gpu" --devices "0," > "${base_dir}/version_1/logs_train.out" 2>&1 &
#nohup python ${train_script} --push_kernel_size 7 --pull_kernel_size 7 --avg_kernel_size 3 --pull_inhibition_strength 4 --pupu_weight 0.33 --logs_version 2 ${common_train_args} --accelerator "gpu" --devices "1," > "${base_dir}/version_2/logs_train.out" 2>&1 &
##nohup python ${train_script} --push_kernel_size 7 --pull_kernel_size 7 --avg_kernel_size 3 --pull_inhibition_strength 4 --pupu_weight 0.4 --logs_version 3 ${common_train_args} --accelerator "gpu" --devices "0," > "${base_dir}/version_3/logs_train.out" 2>&1 &
#nohup python ${train_script} --push_kernel_size 7 --pull_kernel_size 7 --avg_kernel_size 3 --pull_inhibition_strength 4 --pupu_weight 0.66 --logs_version 4 ${common_train_args} --accelerator "gpu" --devices "0," > "${base_dir}/version_4/logs_train.out" 2>&1 &
##nohup python ${train_script} --push_kernel_size 7 --pull_kernel_size 7 --avg_kernel_size 3 --pull_inhibition_strength 4 --pupu_weight 0.8 --logs_version 5 ${common_train_args} --accelerator "gpu" --devices "0," > "${base_dir}/version_5/logs_train.out" 2>&1 &
#nohup python ${train_script} --push_kernel_size 7 --pull_kernel_size 7 --avg_kernel_size 3 --pull_inhibition_strength 4 --pupu_weight 1.0 --logs_version 6 ${common_train_args} --accelerator "gpu" --devices "1," > "${base_dir}/version_6/logs_train.out" 2>&1 &

pp_config="avg3_inh4"

mv "${base_dir}/version_0" "$base_dir/${model}"
mv "${base_dir}/version_1" "$base_dir/${model}_${pp_config}_0.0"
mv "${base_dir}/version_2" "$base_dir/${model}_${pp_config}_0.2"
mv "${base_dir}/version_3" "$base_dir/${model}_${pp_config}_0.4"
mv "${base_dir}/version_4" "$base_dir/${model}_${pp_config}_0.6"
mv "${base_dir}/version_5" "$base_dir/${model}_${pp_config}_0.8"
mv "${base_dir}/version_6" "$base_dir/${model}_${pp_config}_1.0"

nohup python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}" ${common_predict_args} --no-use_push_pull --accelerator "gpu" --devices "0," > "${base_dir}/${model}/logs_predict.out" 2>&1 &
nohup python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_${pp_config}_0.0" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir} --accelerator "gpu" --devices "1," > "${base_dir}/${model}_${pp_config}_0.0/logs_predict.out" 2>&1 &
nohup python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_${pp_config}_0.2" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir} --accelerator "gpu" --devices "0," > "${base_dir}/${model}_${pp_config}_0.2/logs_predict.out" 2>&1 &
nohup python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_${pp_config}_0.4" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir} --accelerator "gpu" --devices "1," > "${base_dir}/${model}_${pp_config}_0.4/logs_predict.out" 2>&1 &
nohup python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_${pp_config}_0.6" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir} --accelerator "gpu" --devices "0," > "${base_dir}/${model}_${pp_config}_0.6/logs_predict.out" 2>&1 &
nohup python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_${pp_config}_0.8" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir} --accelerator "gpu" --devices "1," > "${base_dir}/${model}_${pp_config}_0.8/logs_predict.out" 2>&1 &
nohup python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_${pp_config}_1.0" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir} --accelerator "gpu" --devices "0," > "${base_dir}/${model}_${pp_config}_1.0/logs_predict.out" 2>&1 &
