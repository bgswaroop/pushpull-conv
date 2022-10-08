#!/bin/bash

DATA_HOME="/media/guru" # for computo04
#DATA_HOME="/home/guru" # for computo12

train_script="$HOME/git_code/pushpull-conv/project/train_flow.py"
logs_dir="$HOME/runtime_data/pushpull-conv"
task="classification"
dataset_name="imagenet100_50pc"
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
#mkdir -p ${base_dir}/version_1
#mkdir -p ${base_dir}/version_2
#mkdir -p ${base_dir}/version_3
#mkdir -p ${base_dir}/version_4
#mkdir -p ${base_dir}/version_5
#mkdir -p ${base_dir}/version_6
#mkdir -p ${base_dir}/version_7
#mkdir -p ${base_dir}/version_8
#mkdir -p ${base_dir}/version_9
#mkdir -p ${base_dir}/version_10
#mkdir -p ${base_dir}/version_11
#mkdir -p ${base_dir}/version_12
#
#nohup python ${train_script} --no-use_push_pull --logs_version 0 ${common_train_args} --accelerator "gpu" --devices "0," > "${base_dir}/version_0/logs_train.out" 2>&1 &
#nohup python ${train_script} --push_kernel_size 7 --pull_kernel_size 7 --avg_kernel_size 3 --pull_inhibition_strength 1 --logs_version 1 ${common_train_args} --accelerator "gpu" --devices "1," > "${base_dir}/version_1/logs_train.out" 2>&1 &
#nohup python ${train_script} --push_kernel_size 7 --pull_kernel_size 7 --avg_kernel_size 3 --pull_inhibition_strength 2 --logs_version 2 ${common_train_args} --accelerator "gpu" --devices "0," > "${base_dir}/version_2/logs_train.out" 2>&1 &
#nohup python ${train_script} --push_kernel_size 7 --pull_kernel_size 7 --avg_kernel_size 3 --pull_inhibition_strength 3 --logs_version 3 ${common_train_args} --accelerator "gpu" --devices "1," > "${base_dir}/version_3/logs_train.out" 2>&1 &
#nohup python ${train_script} --push_kernel_size 7 --pull_kernel_size 7 --avg_kernel_size 3 --pull_inhibition_strength 4 --logs_version 4 ${common_train_args} --accelerator "gpu" --devices "0," > "${base_dir}/version_4/logs_train.out" 2>&1 &
#nohup python ${train_script} --push_kernel_size 7 --pull_kernel_size 7 --avg_kernel_size 3 --pull_inhibition_strength 5 --logs_version 5 ${common_train_args} --accelerator "gpu" --devices "1," > "${base_dir}/version_5/logs_train.out" 2>&1 &
#nohup python ${train_script} --push_kernel_size 7 --pull_kernel_size 7 --avg_kernel_size 3 --pull_inhibition_strength 6 --logs_version 6 ${common_train_args} --accelerator "gpu" --devices "0," > "${base_dir}/version_6/logs_train.out" 2>&1 &
#nohup python ${train_script} --push_kernel_size 7 --pull_kernel_size 7 --avg_kernel_size 5 --pull_inhibition_strength 1 --logs_version 7 ${common_train_args} --accelerator "gpu" --devices "1," > "${base_dir}/version_7/logs_train.out" 2>&1 &
#nohup python ${train_script} --push_kernel_size 7 --pull_kernel_size 7 --avg_kernel_size 5 --pull_inhibition_strength 2 --logs_version 8 ${common_train_args} --accelerator "gpu" --devices "0," > "${base_dir}/version_8/logs_train.out" 2>&1 &
#nohup python ${train_script} --push_kernel_size 7 --pull_kernel_size 7 --avg_kernel_size 5 --pull_inhibition_strength 3 --logs_version 9 ${common_train_args} --accelerator "gpu" --devices "1," > "${base_dir}/version_9/logs_train.out" 2>&1 &
#nohup python ${train_script} --push_kernel_size 7 --pull_kernel_size 7 --avg_kernel_size 5 --pull_inhibition_strength 4 --logs_version 10 ${common_train_args} --accelerator "gpu" --devices "0," > "${base_dir}/version_10/logs_train.out" 2>&1 &
#nohup python ${train_script} --push_kernel_size 7 --pull_kernel_size 7 --avg_kernel_size 5 --pull_inhibition_strength 5 --logs_version 11 ${common_train_args} --accelerator "gpu" --devices "1," > "${base_dir}/version_11/logs_train.out" 2>&1 &
#nohup python ${train_script} --push_kernel_size 7 --pull_kernel_size 7 --avg_kernel_size 5 --pull_inhibition_strength 6 --logs_version 12 ${common_train_args} --accelerator "gpu" --devices "0," > "${base_dir}/version_12/logs_train.out" 2>&1 &

mv "${base_dir}/version_0" "$base_dir/${model}"
mv "${base_dir}/version_1" "$base_dir/${model}_pp7x7_avg3_inh1"
mv "${base_dir}/version_2" "$base_dir/${model}_pp7x7_avg3_inh2"
mv "${base_dir}/version_3" "$base_dir/${model}_pp7x7_avg3_inh3"
mv "${base_dir}/version_4" "$base_dir/${model}_pp7x7_avg3_inh4"
mv "${base_dir}/version_5" "$base_dir/${model}_pp7x7_avg3_inh5"
mv "${base_dir}/version_6" "$base_dir/${model}_pp7x7_avg3_inh6"
mv "${base_dir}/version_7" "$base_dir/${model}_pp7x7_avg5_inh1"
mv "${base_dir}/version_8" "$base_dir/${model}_pp7x7_avg5_inh2"
mv "${base_dir}/version_9" "$base_dir/${model}_pp7x7_avg5_inh3"
mv "${base_dir}/version_10" "$base_dir/${model}_pp7x7_avg5_inh4"
mv "${base_dir}/version_11" "$base_dir/${model}_pp7x7_avg5_inh5"
mv "${base_dir}/version_12" "$base_dir/${model}_pp7x7_avg5_inh6"

nohup python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}" ${common_predict_args} --no-use_push_pull --accelerator "gpu" --devices "0," > "${base_dir}/${model}/logs_predict.out" 2>&1 &
nohup python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_pp7x7_avg3_inh1" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir} --accelerator "gpu" --devices "1," > "${base_dir}/${model}_pp7x7_avg3_inh1/logs_predict.out" 2>&1 &
nohup python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_pp7x7_avg3_inh2" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir} --accelerator "gpu" --devices "1," > "${base_dir}/${model}_pp7x7_avg3_inh2/logs_predict.out" 2>&1 &
nohup python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_pp7x7_avg3_inh3" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir} --accelerator "gpu" --devices "1," > "${base_dir}/${model}_pp7x7_avg3_inh3/logs_predict.out" 2>&1 &
nohup python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_pp7x7_avg3_inh4" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir} --accelerator "gpu" --devices "1," > "${base_dir}/${model}_pp7x7_avg3_inh4/logs_predict.out" 2>&1 &
nohup python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_pp7x7_avg3_inh5" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir} --accelerator "gpu" --devices "1," > "${base_dir}/${model}_pp7x7_avg3_inh5/logs_predict.out" 2>&1 &
nohup python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_pp7x7_avg3_inh6" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir} --accelerator "gpu" --devices "1," > "${base_dir}/${model}_pp7x7_avg3_inh6/logs_predict.out" 2>&1 &
nohup python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_pp7x7_avg5_inh1" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir} --accelerator "gpu" --devices "0," > "${base_dir}/${model}_pp7x7_avg5_inh1/logs_predict.out" 2>&1 &
nohup python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_pp7x7_avg5_inh2" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir} --accelerator "gpu" --devices "0," > "${base_dir}/${model}_pp7x7_avg5_inh2/logs_predict.out" 2>&1 &
nohup python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_pp7x7_avg5_inh3" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir} --accelerator "gpu" --devices "0," > "${base_dir}/${model}_pp7x7_avg5_inh3/logs_predict.out" 2>&1 &
nohup python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_pp7x7_avg5_inh4" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir} --accelerator "gpu" --devices "0," > "${base_dir}/${model}_pp7x7_avg5_inh4/logs_predict.out" 2>&1 &
nohup python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_pp7x7_avg5_inh5" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir} --accelerator "gpu" --devices "0," > "${base_dir}/${model}_pp7x7_avg5_inh5/logs_predict.out" 2>&1 &
nohup python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_pp7x7_avg5_inh6" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir} --accelerator "gpu" --devices "0," > "${base_dir}/${model}_pp7x7_avg5_inh6/logs_predict.out" 2>&1 &

