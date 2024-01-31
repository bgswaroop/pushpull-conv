#!/bin/bash

#DATA_HOME="/media/guru" # for computo04
DATA_HOME="/home/guru" # for computo12

train_script="$HOME/git_code/pushpull-conv/project/train_flow.py"
logs_dir="$HOME/runtime_data/pushpull-conv"
task="classification"

dataset_name="imagenet"
dataset_dir="${DATA_HOME}/datasets/imagenet"
corrupted_dataset_dir="${DATA_HOME}/datasets/imagenet-c"
corrupted_dataset_name="imagenet-c"

#dataset_name="cifar10"
#dataset_dir="${DATA_HOME}/datasets/cifar"
#corrupted_dataset_dir="${DATA_HOME}/datasets/cifar/cifar10-c"
#corrupted_dataset_name="cifar10-c"

model="resnet50"
experiment_name="aug_resnet50_${dataset_name}_${task}"
common_train_args="--accelerator gpu --img_size 224 --model ${model} --hash_length 64 --quantization_weight 1e-4 --num_workers 12 --task ${task} --logs_dir ${logs_dir} --experiment_name ${experiment_name} --dataset_dir ${dataset_dir} --dataset_name ${dataset_name}"
base_dir="$logs_dir/$experiment_name"
predict_script="$HOME/git_code/pushpull-conv/project/predict_flow.py"
corruption_types="gaussian_noise shot_noise impulse_noise defocus_blur glass_blur  motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression"
baseline_model_logs_dir="$base_dir/resnet50"
teacher_ckpt="/home/guru/runtime_data/pushpull-conv/resnet18_imagenet200_classification_hparams/resnet18/checkpoints/last.ckpt"
common_predict_args="--accelerator gpu --img_size 224 --model ${model} --num_workers 6 --corruption_types $corruption_types --task ${task} --dataset_dir ${dataset_dir} --dataset_name ${dataset_name} --corrupted_dataset_dir ${corrupted_dataset_dir} --corrupted_dataset_name ${corrupted_dataset_name}"

#i=0
#mkdir -p "${base_dir}/version_${i}"
#nohup python ${train_script} --avg_kernel_size 0 --pull_inhibition_strength 1 --logs_version ${i} ${common_train_args}  --devices "$((i%2))," > "${base_dir}/version_${i}/logs_train.out" 2>&1 &
#
i=0
mkdir -p "${base_dir}/version_${i}"
python ${train_script} --augmentation TrivialAugment --no-use_push_pull --logs_version ${i} ${common_train_args}  --devices "$((i%2))," > "${base_dir}/version_${i}/logs_train.out" 2>&1 &


nohup python ${train_script} --augmentation TrivialAugment --no-use_push_pull --logs_version ${i} ${common_train_args}  --devices "$((i%2))," > "${base_dir}/version_${i}/logs_train.out" 2>&1 &
#
#i=2
#mkdir -p "${base_dir}/version_${i}"
#nohup python ${train_script} --augmentation RandAug --no-use_push_pull --logs_version ${i} ${common_train_args}  --devices "$((i%2))," > "${base_dir}/version_${i}/logs_train.out" 2>&1 &


i=1
mkdir -p "${base_dir}/version_${i}"
nohup python ${train_script} --augmentation TrivialAugment --trainable_pull_inhibition --avg_kernel_size 3 --logs_version ${i} ${common_train_args}  --devices "$((i%2)),"  > "${base_dir}/version_${i}/logs_train.out" 2>&1 &

#args.lr_initial = 5e-3
#args.lr_max = 0.2
#args.lr_end = 5e-6


#i=1
#mkdir -p "${base_dir}/version_${i}"
#nohup python ${train_script} --no-use_push_pull --logs_version ${i} ${common_train_args}  --devices "$((i%2))," > "${base_dir}/version_${i}/logs_train.out" 2>&1 &
#
#sleep 20m
#i=0
#version_name="${model}"
#mv "${base_dir}/version_${i}" "$base_dir/${version_name}"
#nohup python ${predict_script} --baseline_model_logs_dir ${baseline_model_logs_dir} --predict_model_logs_dir "${base_dir}/${version_name}" ${common_predict_args} --no-use_push_pull --accelerator "gpu" --devices "1," > "${base_dir}/${version_name}/logs_predict.out" 2>&1 &
###
#i=0
#version_name="${model}"
##mv "${base_dir}/version_${i}" "$base_dir/${version_name}"
#nohup python ${predict_script} --predict_model_logs_dir "${base_dir}/${version_name}" ${common_predict_args} --no-use_push_pull --accelerator "gpu" --devices "$((i%2))," > "${base_dir}/${version_name}/logs_predict.out" 2>&1 &

#i=1
#for avg in 3 5
#do
##  for inh in 1
##  do
##     mkdir -p "${base_dir}/version_${i}"
##     nohup python ${train_script} --avg_kernel_size $avg --pull_inhibition_strength $inh  --logs_version ${i} ${common_train_args}  --devices "$((i%2))," > "${base_dir}/version_${i}/logs_train.out" 2>&1 &
##     i=$(( $i + 1 ));
##  done
#  mkdir -p "${base_dir}/version_${i}"
#  nohup python ${train_script} --trainable_pull_inhibition True --avg_kernel_size $avg --logs_version ${i} ${common_train_args}  --devices "$((i%2))," > "${base_dir}/version_${i}/logs_train.out" 2>&1 &
#  i=$(( $i + 1 ));
#done

#i=30
#for avg in 3 5
#do
#  mkdir -p "${base_dir}/version_${i}"
#  nohup python ${train_script} --trainable_pull_inhibition True --avg_kernel_size $avg --logs_version ${i} ${common_train_args}  --devices "$((i%2))," > "${base_dir}/version_${i}/logs_train.out" 2>&1 &
#  i=$(( $i + 1 ));
#done

#sleep 4h
i=0
for avg in 0
do
#  for inh in 1
#  do
#    pp_config="w_pullrelu_avg${avg}_inh${inh}"
#    version_name="${model}_${pp_config}"
#    mv "${base_dir}/version_${i}" "$base_dir/${version_name}"
#    nohup python ${predict_script} --predict_model_logs_dir "${base_dir}/${version_name}" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir} --devices "$((i%2))," > "$base_dir/${version_name}/logs_predict.out" 2>&1 &
#    i=$(( $i + 1 ));
#  done
  pp_config="inh1"
  version_name="${model}_${pp_config}"
  mv "${base_dir}/version_${i}" "$base_dir/${version_name}"
  nohup python ${predict_script} --predict_model_logs_dir "${base_dir}/${version_name}" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir} --devices "1," > "$base_dir/${version_name}/logs_predict.out" 2>&1 &
  i=$(( $i + 1 ));
done


#alpha=0.3
#temp=3
#
#i=0
#for avg in 3 5
#do
#  for inh in 1 2 3 4 5 6
#  do
#     mkdir -p "${base_dir}/version_${i}"
#     nohup python ${train_script} --training_type student --teacher_ckpt $teacher_ckpt --distillation_loss_alpha $alpha --distillation_loss_temp $temp --avg_kernel_size $avg --pull_inhibition_strength $inh  --logs_version ${i} ${common_train_args}  --devices "$((i%2))," > "${base_dir}/version_${i}/logs_train.out" 2>&1 &
#     i=$(( $i + 1 ));
#  done
#done

#i=0
#for avg in 3 5
#do
#  for inh in 1 2 3 4 5 6
#  do
#    pp_config="avg${avg}_inh${inh}"
#    version_name="${model}_${pp_config}_temp${temp}_alpha${alpha}"
#    mv "${base_dir}/version_${i}" "$base_dir/${version_name}"
#    nohup python ${predict_script} --predict_model_logs_dir "${base_dir}/${version_name}" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir} --devices "$((i%2))," > "$base_dir/${version_name}/logs_predict.out" 2>&1 &
#    i=$(( $i + 1 ));
#  done
#done
