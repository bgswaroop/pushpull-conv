#!/bin/bash

train_script="$HOME/git_code/pushpull-conv/project/train_flow.py"
logs_dir="/data2/p288722/runtime_data/pushpull-conv"
task="classification"
dataset_name="cifar10"
dataset_dir="/data2/p288722/datasets/cifar"
corrupted_dataset_dir="/data2/p288722/datasets/cifar/CIFAR-10-C"
corrupted_dataset_name="cifar10-c"
model="resnet50"
experiment_name="${model}_${dataset_name}_${task}"
common_train_args="--img_size 32 --model ${model} --hash_length 64 --quantization_weight 0.0001 --no-scale_the_outputs --bias --num_workers 32 --batch_size 256 --max_epochs 50 --weight_decay 0.0005 --learning_rate 0.05 --task ${task} --logs_dir ${logs_dir} --experiment_name ${experiment_name} --dataset_dir ${dataset_dir} --dataset_name ${dataset_name}"
base_dir="$logs_dir/$experiment_name"

#python ${train_script} --no-use_push_pull --logs_version 0 ${common_train_args}
#python ${train_script} --push_kernel_size 3 --pull_kernel_size 3 --avg_kernel_size 3 --pull_inhibition_strength 1 --logs_version 1 ${common_train_args}
#python ${train_script} --push_kernel_size 3 --pull_kernel_size 3 --avg_kernel_size 3 --pull_inhibition_strength 2 --logs_version 2 ${common_train_args}
#python ${train_script} --push_kernel_size 3 --pull_kernel_size 3 --avg_kernel_size 3 --pull_inhibition_strength 3 --logs_version 3 ${common_train_args}
#python ${train_script} --push_kernel_size 3 --pull_kernel_size 3 --avg_kernel_size 3 --pull_inhibition_strength 4 --logs_version 4 ${common_train_args}
#python ${train_script} --push_kernel_size 3 --pull_kernel_size 3 --avg_kernel_size 5 --pull_inhibition_strength 1 --logs_version 5 ${common_train_args}
#python ${train_script} --push_kernel_size 3 --pull_kernel_size 3 --avg_kernel_size 5 --pull_inhibition_strength 2 --logs_version 6 ${common_train_args}
#python ${train_script} --push_kernel_size 3 --pull_kernel_size 3 --avg_kernel_size 5 --pull_inhibition_strength 3 --logs_version 7 ${common_train_args}
#python ${train_script} --push_kernel_size 3 --pull_kernel_size 3 --avg_kernel_size 5 --pull_inhibition_strength 4 --logs_version 8 ${common_train_args}
#
#mv "${base_dir}/version_0" "$base_dir/${model}"
#mv "${base_dir}/version_1" "$base_dir/${model}_pp3x3_avg3_inh1"
#mv "${base_dir}/version_2" "$base_dir/${model}_pp3x3_avg3_inh2"
#mv "${base_dir}/version_3" "$base_dir/${model}_pp3x3_avg3_inh3"
#mv "${base_dir}/version_4" "$base_dir/${model}_pp3x3_avg3_inh4"
#mv "${base_dir}/version_5" "$base_dir/${model}_pp3x3_avg5_inh1"
#mv "${base_dir}/version_6" "$base_dir/${model}_pp3x3_avg5_inh2"
#mv "${base_dir}/version_7" "$base_dir/${model}_pp3x3_avg5_inh3"
#mv "${base_dir}/version_8" "$base_dir/${model}_pp3x3_avg5_inh4"

predict_script="$HOME/git_code/pushpull-conv/project/predict_flow_update.py"
corruption_types="gaussian_noise shot_noise impulse_noise defocus_blur glass_blur  motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression"
baseline_model_logs_dir="$base_dir/${model}"
common_predict_args="--img_size 224 --model ${model} --num_workers 12 --corruption_types $corruption_types --task ${task} --dataset_dir ${dataset_dir} --dataset_name ${dataset_name} --corrupted_dataset_dir ${corrupted_dataset_dir} --corrupted_dataset_name ${corrupted_dataset_name}"

python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}" ${common_predict_args} --no-use_push_pull
python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_pp3x3_avg3_inh1" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir}
python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_pp3x3_avg3_inh2" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir}
python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_pp3x3_avg3_inh3" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir}
python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_pp3x3_avg3_inh4" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir}
python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_pp3x3_avg5_inh1" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir}
python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_pp3x3_avg5_inh2" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir}
python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_pp3x3_avg5_inh3" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir}
python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_pp3x3_avg5_inh4" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir}




model="resnet34"
experiment_name="${model}_${dataset_name}_${task}"
common_train_args="--img_size 32 --model ${model} --hash_length 64 --quantization_weight 0.0001 --no-scale_the_outputs --bias --num_workers 32 --batch_size 256 --max_epochs 50 --weight_decay 0.0005 --learning_rate 0.05 --task ${task} --logs_dir ${logs_dir} --experiment_name ${experiment_name} --dataset_dir ${dataset_dir} --dataset_name ${dataset_name}"
base_dir="$logs_dir/$experiment_name"

#python ${train_script} --no-use_push_pull --logs_version 0 ${common_train_args}
#python ${train_script} --push_kernel_size 3 --pull_kernel_size 3 --avg_kernel_size 3 --pull_inhibition_strength 1 --logs_version 1 ${common_train_args}
#python ${train_script} --push_kernel_size 3 --pull_kernel_size 3 --avg_kernel_size 3 --pull_inhibition_strength 2 --logs_version 2 ${common_train_args}
#python ${train_script} --push_kernel_size 3 --pull_kernel_size 3 --avg_kernel_size 3 --pull_inhibition_strength 3 --logs_version 3 ${common_train_args}
#python ${train_script} --push_kernel_size 3 --pull_kernel_size 3 --avg_kernel_size 3 --pull_inhibition_strength 4 --logs_version 4 ${common_train_args}
#python ${train_script} --push_kernel_size 3 --pull_kernel_size 3 --avg_kernel_size 5 --pull_inhibition_strength 1 --logs_version 5 ${common_train_args}
#python ${train_script} --push_kernel_size 3 --pull_kernel_size 3 --avg_kernel_size 5 --pull_inhibition_strength 2 --logs_version 6 ${common_train_args}
#python ${train_script} --push_kernel_size 3 --pull_kernel_size 3 --avg_kernel_size 5 --pull_inhibition_strength 3 --logs_version 7 ${common_train_args}
#python ${train_script} --push_kernel_size 3 --pull_kernel_size 3 --avg_kernel_size 5 --pull_inhibition_strength 4 --logs_version 8 ${common_train_args}
#
#mv "${base_dir}/version_0" "$base_dir/${model}"
#mv "${base_dir}/version_1" "$base_dir/${model}_pp3x3_avg3_inh1"
#mv "${base_dir}/version_2" "$base_dir/${model}_pp3x3_avg3_inh2"
#mv "${base_dir}/version_3" "$base_dir/${model}_pp3x3_avg3_inh3"
#mv "${base_dir}/version_4" "$base_dir/${model}_pp3x3_avg3_inh4"
#mv "${base_dir}/version_5" "$base_dir/${model}_pp3x3_avg5_inh1"
#mv "${base_dir}/version_6" "$base_dir/${model}_pp3x3_avg5_inh2"
#mv "${base_dir}/version_7" "$base_dir/${model}_pp3x3_avg5_inh3"
#mv "${base_dir}/version_8" "$base_dir/${model}_pp3x3_avg5_inh4"

predict_script="$HOME/git_code/pushpull-conv/project/predict_flow.py"
corruption_types="gaussian_noise shot_noise impulse_noise defocus_blur glass_blur  motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression"
baseline_model_logs_dir="$base_dir/${model}"
common_predict_args="--img_size 224 --model ${model} --num_workers 12 --corruption_types $corruption_types --task ${task} --dataset_dir ${dataset_dir} --dataset_name ${dataset_name} --corrupted_dataset_dir ${corrupted_dataset_dir} --corrupted_dataset_name ${corrupted_dataset_name}"

python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}" ${common_predict_args} --no-use_push_pull
python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_pp3x3_avg3_inh1" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir}
python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_pp3x3_avg3_inh2" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir}
python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_pp3x3_avg3_inh3" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir}
python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_pp3x3_avg3_inh4" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir}
python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_pp3x3_avg5_inh1" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir}
python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_pp3x3_avg5_inh2" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir}
python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_pp3x3_avg5_inh3" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir}
python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_pp3x3_avg5_inh4" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir}





model="resnet50"
experiment_name="${model}_${dataset_name}_${task}"
common_train_args="--img_size 32 --model ${model} --hash_length 64 --quantization_weight 0.0001 --no-scale_the_outputs --bias --num_workers 32 --batch_size 256 --max_epochs 50 --weight_decay 0.0005 --learning_rate 0.05 --task ${task} --logs_dir ${logs_dir} --experiment_name ${experiment_name} --dataset_dir ${dataset_dir} --dataset_name ${dataset_name}"
base_dir="$logs_dir/$experiment_name"

#python ${train_script} --no-use_push_pull --logs_version 0 ${common_train_args}
#python ${train_script} --push_kernel_size 3 --pull_kernel_size 3 --avg_kernel_size 3 --pull_inhibition_strength 1 --logs_version 1 ${common_train_args}
#python ${train_script} --push_kernel_size 3 --pull_kernel_size 3 --avg_kernel_size 3 --pull_inhibition_strength 2 --logs_version 2 ${common_train_args}
#python ${train_script} --push_kernel_size 3 --pull_kernel_size 3 --avg_kernel_size 3 --pull_inhibition_strength 3 --logs_version 3 ${common_train_args}
#python ${train_script} --push_kernel_size 3 --pull_kernel_size 3 --avg_kernel_size 3 --pull_inhibition_strength 4 --logs_version 4 ${common_train_args}
#python ${train_script} --push_kernel_size 3 --pull_kernel_size 3 --avg_kernel_size 5 --pull_inhibition_strength 1 --logs_version 5 ${common_train_args}
#python ${train_script} --push_kernel_size 3 --pull_kernel_size 3 --avg_kernel_size 5 --pull_inhibition_strength 2 --logs_version 6 ${common_train_args}
#python ${train_script} --push_kernel_size 3 --pull_kernel_size 3 --avg_kernel_size 5 --pull_inhibition_strength 3 --logs_version 7 ${common_train_args}
python ${train_script} --push_kernel_size 3 --pull_kernel_size 3 --avg_kernel_size 5 --pull_inhibition_strength 4 --logs_version 8 ${common_train_args}
#
mv "${base_dir}/version_0" "$base_dir/${model}"
mv "${base_dir}/version_1" "$base_dir/${model}_pp3x3_avg3_inh1"
mv "${base_dir}/version_2" "$base_dir/${model}_pp3x3_avg3_inh2"
mv "${base_dir}/version_3" "$base_dir/${model}_pp3x3_avg3_inh3"
mv "${base_dir}/version_4" "$base_dir/${model}_pp3x3_avg3_inh4"
mv "${base_dir}/version_5" "$base_dir/${model}_pp3x3_avg5_inh1"
mv "${base_dir}/version_6" "$base_dir/${model}_pp3x3_avg5_inh2"
mv "${base_dir}/version_7" "$base_dir/${model}_pp3x3_avg5_inh3"
mv "${base_dir}/version_8" "$base_dir/${model}_pp3x3_avg5_inh4"

predict_script="$HOME/git_code/pushpull-conv/project/predict_flow.py"
corruption_types="gaussian_noise shot_noise impulse_noise defocus_blur glass_blur  motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression"
baseline_model_logs_dir="$base_dir/${model}"
common_predict_args="--img_size 224 --model ${model} --num_workers 12 --corruption_types $corruption_types --task ${task} --dataset_dir ${dataset_dir} --dataset_name ${dataset_name} --corrupted_dataset_dir ${corrupted_dataset_dir} --corrupted_dataset_name ${corrupted_dataset_name}"

python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}" ${common_predict_args} --no-use_push_pull
python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_pp3x3_avg3_inh1" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir}
python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_pp3x3_avg3_inh2" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir}
python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_pp3x3_avg3_inh3" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir}
python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_pp3x3_avg3_inh4" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir}
python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_pp3x3_avg5_inh1" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir}
python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_pp3x3_avg5_inh2" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir}
python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_pp3x3_avg5_inh3" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir}
python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_pp3x3_avg5_inh4" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir}


