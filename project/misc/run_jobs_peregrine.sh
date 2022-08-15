#!/bin/bash
#SBATCH --job-name=rn18_imgnet
#SBATCH --time=24:00:00
#SBATCH --mem=120g
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=12
#SBATCH --array=0-8
#SBATCH --mail-user=g.s.bennabhaktula@rug.nl
#SBATCH --mail-type=FAIL

# SLURM Notation used above
# %x - Name of the Job
# %A - JOB ID
# %a - TASK ID

# The following times are for V100 GPU with 100GB RAM and 12 CPU cores
# Dataset    | CIFAR-10 | ImageNet200
#------------|----------|-------------
# AlexNet    | 3.5 hrs  |
# ResNet18   | 4.5 hrs  |
# ResNet34   | 5.5 hrs  |
# ResNet50   | 6.5 hrs  | 

module load CUDA/11.1.1-GCC-10.2.0
source /data/p288722/python_venv/deep_hashing/bin/activate

train_script="$HOME/git_code/deep_hashing/project/push_pull_train.py"
logs_dir="/data/p288722/runtime_data/deep_hashing"
experiment_name="dsh_pp_ResNet18_48bit_ImageNet200_run1"
dataset_dir="/data/p288722/datasets/imagenet"
dataset_name="imagenet200"
corrupted_dataset_dir="/scratch/p288722/datasets/imagenet/imagenet-c"
corrupted_dataset_name="imagenet200-c"
common_train_args="--img_size 224 --model resnet18 --hash_length 48 --no-scale_the_outputs --bias --num_workers 12 --logs_dir ${logs_dir} --experiment_name ${experiment_name} --dataset_dir ${dataset_dir} --dataset_name ${dataset_name}"
#echo python ${train_script} --no-use_push_pull --logs_version ${SLURM_ARRAY_TASK_ID} ${common_train_args}

case ${SLURM_ARRAY_TASK_ID} in
0) python ${train_script} --no-use_push_pull --logs_version ${SLURM_ARRAY_TASK_ID} ${common_train_args} ;;
1) python ${train_script} --push_kernel_size 7 --pull_kernel_size 7 --avg_kernel_size 3 --pull_inhibition_strength 1 --logs_version ${SLURM_ARRAY_TASK_ID} ${common_train_args} ;;
2) python ${train_script} --push_kernel_size 7 --pull_kernel_size 7 --avg_kernel_size 3 --pull_inhibition_strength 2 --logs_version ${SLURM_ARRAY_TASK_ID} ${common_train_args} ;;
3) python ${train_script} --push_kernel_size 7 --pull_kernel_size 7 --avg_kernel_size 3 --pull_inhibition_strength 3 --logs_version ${SLURM_ARRAY_TASK_ID} ${common_train_args} ;;
4) python ${train_script} --push_kernel_size 7 --pull_kernel_size 7 --avg_kernel_size 5 --pull_inhibition_strength 1 --logs_version ${SLURM_ARRAY_TASK_ID} ${common_train_args} ;;
5) python ${train_script} --push_kernel_size 7 --pull_kernel_size 7 --avg_kernel_size 5 --pull_inhibition_strength 2 --logs_version ${SLURM_ARRAY_TASK_ID} ${common_train_args} ;;
6) python ${train_script} --push_kernel_size 7 --pull_kernel_size 7 --avg_kernel_size 5 --pull_inhibition_strength 3 --logs_version ${SLURM_ARRAY_TASK_ID} ${common_train_args} ;;
7) python ${train_script} --push_kernel_size 7 --pull_kernel_size 7 --avg_kernel_size 3 --pull_inhibition_strength 4 --logs_version ${SLURM_ARRAY_TASK_ID} ${common_train_args} ;;
8) python ${train_script} --push_kernel_size 7 --pull_kernel_size 7 --avg_kernel_size 5 --pull_inhibition_strength 4 --logs_version ${SLURM_ARRAY_TASK_ID} ${common_train_args} ;;
esac

base_dir="$logs_dir/$experiment_name"

case ${SLURM_ARRAY_TASK_ID} in
0) mv "${base_dir}/version_${SLURM_ARRAY_TASK_ID}" "$base_dir/ResNet18_48bit" ;;
1) mv "${base_dir}/version_${SLURM_ARRAY_TASK_ID}" "$base_dir/ResNet18_pp7x7_avg3_inh1" ;;
2) mv "${base_dir}/version_${SLURM_ARRAY_TASK_ID}" "$base_dir/ResNet18_pp7x7_avg3_inh2" ;;
3) mv "${base_dir}/version_${SLURM_ARRAY_TASK_ID}" "$base_dir/ResNet18_pp7x7_avg3_inh3" ;;
4) mv "${base_dir}/version_${SLURM_ARRAY_TASK_ID}" "$base_dir/ResNet18_pp7x7_avg5_inh1" ;;
5) mv "${base_dir}/version_${SLURM_ARRAY_TASK_ID}" "$base_dir/ResNet18_pp7x7_avg5_inh2" ;;
6) mv "${base_dir}/version_${SLURM_ARRAY_TASK_ID}" "$base_dir/ResNet18_pp7x7_avg5_inh3" ;;
7) mv "${base_dir}/version_${SLURM_ARRAY_TASK_ID}" "$base_dir/ResNet18_pp7x7_avg3_inh4" ;;
8) mv "${base_dir}/version_${SLURM_ARRAY_TASK_ID}" "$base_dir/ResNet18_pp7x7_avg5_inh4" ;;
esac
#echo mv "${base_dir}/version_${SLURM_ARRAY_TASK_ID}" "$base_dir/ResNet18_48bit"

predict_script="$HOME/git_code/deep_hashing/project/push_pull_predict.py"
# "gaussian_noise shot_noise impulse_noise defocus_blur glass_blur  motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression"
corruption_types="gaussian_noise shot_noise impulse_noise defocus_blur glass_blur  motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression"
baseline_classifier_results_dir="/data/p288722/runtime_data/deep_hashing/dsh_pp_ResNet18_48bit_ImageNet200_run1/ResNet18_48bit/results"
common_predict_args="--img_size 224 --model resnet18 --corrupted_dataset_name CIFAR-10-C-224x224 --num_workers 12 --baseline_classifier_results_dir ${baseline_classifier_results_dir} --corruption_types $corruption_types --dataset_dir ${dataset_dir} --dataset_name ${dataset_name} --corrupted_dataset_dir ${corrupted_dataset_dir} --corrupted_dataset_name ${corrupted_dataset_name}"
#echo python ${predict_script} --model_ckpt "${base_dir}/ResNet18_48bit/checkpoints/last.ckpt" ${common_predict_args} --no-use_push_pull


case ${SLURM_ARRAY_TASK_ID} in
0) python ${predict_script} --model_ckpt "${base_dir}/ResNet18_48bit/checkpoints/last.ckpt" ${common_predict_args} --no-use_push_pull ;;
1) python ${predict_script} --model_ckpt "${base_dir}/ResNet18_pp7x7_avg3_inh1/checkpoints/last.ckpt" ${common_predict_args} ;;
2) python ${predict_script} --model_ckpt "${base_dir}/ResNet18_pp7x7_avg3_inh2/checkpoints/last.ckpt" ${common_predict_args} ;;
3) python ${predict_script} --model_ckpt "${base_dir}/ResNet18_pp7x7_avg3_inh3/checkpoints/last.ckpt" ${common_predict_args} ;;
4) python ${predict_script} --model_ckpt "${base_dir}/ResNet18_pp7x7_avg5_inh1/checkpoints/last.ckpt" ${common_predict_args} ;;
5) python ${predict_script} --model_ckpt "${base_dir}/ResNet18_pp7x7_avg5_inh2/checkpoints/last.ckpt" ${common_predict_args} ;;
6) python ${predict_script} --model_ckpt "${base_dir}/ResNet18_pp7x7_avg5_inh3/checkpoints/last.ckpt" ${common_predict_args} ;;
7) python ${predict_script} --model_ckpt "${base_dir}/ResNet18_pp7x7_avg3_inh4/checkpoints/last.ckpt" ${common_predict_args} ;;
8) python ${predict_script} --model_ckpt "${base_dir}/ResNet18_pp7x7_avg5_inh4/checkpoints/last.ckpt" ${common_predict_args} ;;
esac
