#!/bin/bash

# The following times are for A40 GPU with 100GB RAM and 32 CPU cores
# Dataset    | CIFAR-10 | ImageNet200
#------------|----------|-------------
# AlexNet    |     hrs  |
# ResNet18   |     hrs  |
# ResNet34   |     hrs  |
# ResNet50   |     hrs  |

module load CUDA/11.1.1-GCC-10.2.0
source /data/p288722/python_venv/pushpull-conv/bin/activate

# python /home/p288722/git_code/pushpull-conv/project/train_flow.py --model resnet18 --no-use_push_pull
# --logs_dir /data2/p288722/runtime_data/pushpull-conv --task retrieval --experiment_name resnet18_imagenet200_retrieval
# --hash_length 48 --dataset_dir /data2/p288722/datasets/imagenet --dataset_name imagenet200 --num_workers 32
# --batch_size 256 --weight_decay 0.0001 --learning_rate 0.01

train_script="$HOME/git_code/pushpull-conv/project/train_flow.py"
logs_dir="/data/p288722/runtime_data/pushpull-conv"
task="classification"
dataset_name="imagenet200"
dataset_dir="/data/p288722/datasets/imagenet"
corrupted_dataset_dir="/scratch/p288722/datasets/imagenet/imagenet-c"
corrupted_dataset_name="imagenet200-c"
model="resnet50"
experiment_name="${model}_${dataset_name}_${task}"
common_train_args="--img_size 224 --model ${model} --hash_length 64 --quantization_weight 0.0001 --no-scale_the_outputs --bias --num_workers 12 --batch_size 64 --max_epochs 60 --weight_decay 0.0001 --learning_rate 0.1 --task ${task} --logs_dir ${logs_dir} --experiment_name ${experiment_name} --dataset_dir ${dataset_dir} --dataset_name ${dataset_name}"

case ${SLURM_ARRAY_TASK_ID} in
0) python ${train_script} --no-use_push_pull --logs_version ${SLURM_ARRAY_TASK_ID} ${common_train_args} ;;
1) python ${train_script} --push_kernel_size 7 --pull_kernel_size 7 --avg_kernel_size 3 --pull_inhibition_strength 1 --logs_version ${SLURM_ARRAY_TASK_ID} ${common_train_args} ;;
2) python ${train_script} --push_kernel_size 7 --pull_kernel_size 7 --avg_kernel_size 3 --pull_inhibition_strength 2 --logs_version ${SLURM_ARRAY_TASK_ID} ${common_train_args} ;;
3) python ${train_script} --push_kernel_size 7 --pull_kernel_size 7 --avg_kernel_size 3 --pull_inhibition_strength 3 --logs_version ${SLURM_ARRAY_TASK_ID} ${common_train_args} ;;
4) python ${train_script} --push_kernel_size 7 --pull_kernel_size 7 --avg_kernel_size 3 --pull_inhibition_strength 4 --logs_version ${SLURM_ARRAY_TASK_ID} ${common_train_args} ;;
5) python ${train_script} --push_kernel_size 7 --pull_kernel_size 7 --avg_kernel_size 5 --pull_inhibition_strength 1 --logs_version ${SLURM_ARRAY_TASK_ID} ${common_train_args} ;;
6) python ${train_script} --push_kernel_size 7 --pull_kernel_size 7 --avg_kernel_size 5 --pull_inhibition_strength 2 --logs_version ${SLURM_ARRAY_TASK_ID} ${common_train_args} ;;
7) python ${train_script} --push_kernel_size 7 --pull_kernel_size 7 --avg_kernel_size 5 --pull_inhibition_strength 3 --logs_version ${SLURM_ARRAY_TASK_ID} ${common_train_args} ;;
8) python ${train_script} --push_kernel_size 7 --pull_kernel_size 7 --avg_kernel_size 5 --pull_inhibition_strength 4 --logs_version ${SLURM_ARRAY_TASK_ID} ${common_train_args} ;;
esac

base_dir="$logs_dir/$experiment_name"

case ${SLURM_ARRAY_TASK_ID} in
0) mv "${base_dir}/version_${SLURM_ARRAY_TASK_ID}" "$base_dir/${model}" ;;
1) mv "${base_dir}/version_${SLURM_ARRAY_TASK_ID}" "$base_dir/${model}_pp7x7_avg3_inh1" ;;
2) mv "${base_dir}/version_${SLURM_ARRAY_TASK_ID}" "$base_dir/${model}_pp7x7_avg3_inh2" ;;
3) mv "${base_dir}/version_${SLURM_ARRAY_TASK_ID}" "$base_dir/${model}_pp7x7_avg3_inh3" ;;
4) mv "${base_dir}/version_${SLURM_ARRAY_TASK_ID}" "$base_dir/${model}_pp7x7_avg3_inh4" ;;
5) mv "${base_dir}/version_${SLURM_ARRAY_TASK_ID}" "$base_dir/${model}_pp7x7_avg5_inh1" ;;
6) mv "${base_dir}/version_${SLURM_ARRAY_TASK_ID}" "$base_dir/${model}_pp7x7_avg5_inh2" ;;
7) mv "${base_dir}/version_${SLURM_ARRAY_TASK_ID}" "$base_dir/${model}_pp7x7_avg5_inh3" ;;
8) mv "${base_dir}/version_${SLURM_ARRAY_TASK_ID}" "$base_dir/${model}_pp7x7_avg5_inh4" ;;
esac

predict_script="$HOME/git_code/pushpull-conv/project/predict_flow.py"
# "gaussian_noise shot_noise impulse_noise defocus_blur glass_blur  motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression"
corruption_types="gaussian_noise shot_noise impulse_noise defocus_blur glass_blur  motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression"
baseline_model_logs_dir="$base_dir/${model}"
common_predict_args="--img_size 224 --model ${model} --num_workers 12 --corruption_types $corruption_types --task ${task} --dataset_dir ${dataset_dir} --dataset_name ${dataset_name} --corrupted_dataset_dir ${corrupted_dataset_dir} --corrupted_dataset_name ${corrupted_dataset_name}"

case ${SLURM_ARRAY_TASK_ID} in
0) python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}" ${common_predict_args} --no-use_push_pull ;;
1) python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_pp7x7_avg3_inh1" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir} ;;
2) python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_pp7x7_avg3_inh2" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir} ;;
3) python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_pp7x7_avg3_inh3" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir} ;;
4) python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_pp7x7_avg3_inh4" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir} ;;
5) python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_pp7x7_avg5_inh1" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir} ;;
6) python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_pp7x7_avg5_inh2" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir} ;;
7) python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_pp7x7_avg5_inh3" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir} ;;
8) python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_pp7x7_avg5_inh4" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir} ;;
esac
