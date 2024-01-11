#!/bin/bash
#
#DATA_HOME="/media/guru" # for computo04
#DATA_HOME="/home/guru" # for computo12
#
#train_script="$HOME/git_code/pushpull-conv/project/train_flow.py"
#logs_dir="$HOME/runtime_data/pushpull-conv"
#task="classification"
#
#dataset_name="cifar10"
#dataset_dir="${DATA_HOME}/datasets/cifar"
#corrupted_dataset_dir="${DATA_HOME}/datasets/cifar/cifar10-c"
#corrupted_dataset_name="cifar10-c"
#
#model="resnet50"
#experiment_name="resnet50_${dataset_name}_${task}"
#common_train_args="--accelerator gpu --img_size 224 --model ${model} --hash_length 64 --quantization_weight 1e-4 --num_workers 12 --task ${task} --logs_dir ${logs_dir} --experiment_name ${experiment_name} --dataset_dir ${dataset_dir} --dataset_name ${dataset_name}"
#base_dir="$logs_dir/$experiment_name"
#predict_script="$HOME/git_code/pushpull-conv/project/predict_flow.py"
#corruption_types="gaussian_noise shot_noise impulse_noise defocus_blur glass_blur  motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression"
#teacher_ckpt="/home/guru/runtime_data/pushpull-conv/resnet18_imagenet200_classification_hparams/resnet18/checkpoints/last.ckpt"
#common_predict_args="--accelerator gpu --img_size 224 --model ${model} --num_workers 6 --corruption_types $corruption_types --task ${task} --dataset_dir ${dataset_dir} --dataset_name ${dataset_name} --corrupted_dataset_dir ${corrupted_dataset_dir} --corrupted_dataset_name ${corrupted_dataset_name}"
#
##i=3
###mkdir -p "${base_dir}/version_${i}"
###python ${train_script} --augmentation AugMix --no-use_push_pull --logs_version ${i} ${common_train_args}  --devices "$((i%2))," > "${base_dir}/version_${i}/logs_train.out"
#version_name="${model}_AugMix"
#baseline_model_logs_dir="$base_dir/${version_name}"
##mv "${base_dir}/version_${i}" "$base_dir/${version_name}"
##python ${predict_script} --predict_model_logs_dir "${base_dir}/${version_name}" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir} --devices "$((i%2))," > "$base_dir/${version_name}/logs_predict.out"
#
##i=6
##avg=3
###mkdir -p "${base_dir}/version_${i}"
###python ${train_script} --augmentation AugMix --trainable_pull_inhibition True --avg_kernel_size $avg --logs_version ${i} ${common_train_args}  --devices "$((i%2))," > "${base_dir}/version_${i}/logs_train.out"
##version_name="${model}_AugMix_avg${avg}_inh_trainable"
##mv "${base_dir}/version_${i}" "$base_dir/${version_name}"
##python ${predict_script} --predict_model_logs_dir "${base_dir}/${version_name}" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir} --devices "$((i%2))," > "$base_dir/${version_name}/logs_predict.out"
##
#i=5
#avg=5
##mkdir -p "${base_dir}/version_${i}"
##python ${train_script} --augmentation AugMix --trainable_pull_inhibition True --avg_kernel_size $avg --logs_version ${i} ${common_train_args}  --devices "$((i%2))," > "${base_dir}/version_${i}/logs_train.out"
#version_name="${model}_AugMix_avg${avg}_inh_trainable"
#mv "${base_dir}/version_${i}" "$base_dir/${version_name}"
#nohup python ${predict_script} --predict_model_logs_dir "${base_dir}/${version_name}" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir} --devices "$((i%2))," > "$base_dir/${version_name}/logs_predict.out" 2>&1 &


###!/bin/bash
#
#DATA_HOME="/media/guru" # for computo04
DATA_HOME="/home/guru" # for computo12

train_script="$HOME/git_code/pushpull-conv/project/train_flow.py"
logs_dir="$HOME/runtime_data/pushpull-conv"
task="classification"

dataset_name="cifar10"
dataset_dir="${DATA_HOME}/datasets/cifar"
corrupted_dataset_dir="${DATA_HOME}/datasets/cifar/cifar10-c"
corrupted_dataset_name="cifar10-c"

#model="zhang_resnet50"
#experiment_name="sota_resnet50_${dataset_name}_${task}"
#common_train_args="--accelerator gpu --img_size 224 --model ${model} --hash_length 64 --quantization_weight 1e-4 --num_workers 12 --task ${task} --logs_dir ${logs_dir} --experiment_name ${experiment_name} --dataset_dir ${dataset_dir} --dataset_name ${dataset_name}"
#base_dir="$logs_dir/$experiment_name"
#predict_script="$HOME/git_code/pushpull-conv/project/predict_flow.py"
#corruption_types="gaussian_noise shot_noise impulse_noise defocus_blur glass_blur  motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression"
#teacher_ckpt="/home/guru/runtime_data/pushpull-conv/resnet18_imagenet200_classification_hparams/resnet18/checkpoints/last.ckpt"
#common_predict_args="--accelerator gpu --img_size 224 --model ${model} --num_workers 6 --corruption_types $corruption_types --task ${task} --dataset_dir ${dataset_dir} --dataset_name ${dataset_name} --corrupted_dataset_dir ${corrupted_dataset_dir} --corrupted_dataset_name ${corrupted_dataset_name}"
#baseline_model_logs_dir="$base_dir/resnet50"
#
#i=9
#mkdir -p "${base_dir}/version_${i}"
#python ${train_script}  --no-use_push_pull --logs_version ${i} ${common_train_args}  --devices "$((i%2))," > "${base_dir}/version_${i}/logs_train.out"
#version_name="${model}"
#mv "${base_dir}/version_${i}" "$base_dir/${version_name}"
#python ${predict_script} --predict_model_logs_dir "${base_dir}/${version_name}" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir} --devices "$((i%2))," > "$base_dir/${version_name}/logs_predict.out"



model="vasconcelos_resnet50"
experiment_name="sota_resnet50_${dataset_name}_${task}"
common_train_args="--accelerator gpu --img_size 224 --model ${model} --hash_length 64 --quantization_weight 1e-4 --num_workers 12 --task ${task} --logs_dir ${logs_dir} --experiment_name ${experiment_name} --dataset_dir ${dataset_dir} --dataset_name ${dataset_name}"
base_dir="$logs_dir/$experiment_name"
predict_script="$HOME/git_code/pushpull-conv/project/predict_flow.py"
corruption_types="gaussian_noise shot_noise impulse_noise defocus_blur glass_blur  motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression"
teacher_ckpt="/home/guru/runtime_data/pushpull-conv/resnet18_imagenet200_classification_hparams/resnet18/checkpoints/last.ckpt"
common_predict_args="--accelerator gpu --img_size 224 --model ${model} --num_workers 6 --corruption_types $corruption_types --task ${task} --dataset_dir ${dataset_dir} --dataset_name ${dataset_name} --corrupted_dataset_dir ${corrupted_dataset_dir} --corrupted_dataset_name ${corrupted_dataset_name}"
baseline_model_logs_dir="$base_dir/resnet50"

i=10
#mkdir -p "${base_dir}/version_${i}"
#nohup python ${train_script}  --no-use_push_pull --logs_version ${i} ${common_train_args}  --devices "1," > "${base_dir}/version_${i}/logs_train.out" 2>&1 &
version_name=${model}
mv "${base_dir}/version_${i}" "$base_dir/${version_name}"
nohup python ${predict_script} --predict_model_logs_dir "${base_dir}/${version_name}" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir} --devices "$((i%2))," > "$base_dir/${version_name}/logs_predict.out" 2>&1 &


#model="strisciuglio_resnet50"
#experiment_name="sota_resnet50_${dataset_name}_${task}"
#common_train_args="--accelerator gpu --img_size 224 --model ${model} --hash_length 64 --quantization_weight 1e-4 --num_workers 12 --task ${task} --logs_dir ${logs_dir} --experiment_name ${experiment_name} --dataset_dir ${dataset_dir} --dataset_name ${dataset_name}"
#base_dir="$logs_dir/$experiment_name"
#predict_script="$HOME/git_code/pushpull-conv/project/predict_flow.py"
#corruption_types="gaussian_noise shot_noise impulse_noise defocus_blur glass_blur  motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression"
#teacher_ckpt="/home/guru/runtime_data/pushpull-conv/resnet18_imagenet200_classification_hparams/resnet18/checkpoints/last.ckpt"
#common_predict_args="--accelerator gpu --img_size 224 --model ${model} --num_workers 6 --corruption_types $corruption_types --task ${task} --dataset_dir ${dataset_dir} --dataset_name ${dataset_name} --corrupted_dataset_dir ${corrupted_dataset_dir} --corrupted_dataset_name ${corrupted_dataset_name}"
#baseline_model_logs_dir="$base_dir/resnet50"
#
#i=11
##mkdir -p "${base_dir}/version_${i}"
##python ${train_script}  --no-use_push_pull --logs_version ${i} ${common_train_args}  --devices "$((i%2))," > "${base_dir}/version_${i}/logs_train.out"
#version_name="${model}"
#mv "${base_dir}/version_${i}" "$base_dir/${version_name}"
#python ${predict_script} --predict_model_logs_dir "${base_dir}/${version_name}" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir} --devices "$((i%2))," > "$base_dir/${version_name}/logs_predict.out"

