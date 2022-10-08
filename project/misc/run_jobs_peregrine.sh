#!/bin/bash
#SBATCH --job-name=r18_in200_retrieval
#SBATCH --time=20:00:00
#SBATCH --mem=60g
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=12
#SBATCH --array=2
#SBATCH --mail-user=g.s.bennabhaktula@rug.nl
#SBATCH --mail-type=FAIL
#--dependency=afterok:26551770_0

# SLURM Notation used above
# %x - Name of the Job
# %A - JOB ID
# %a - TASK ID

# The following times (upperbound) are for V100 GPU with 100GB RAM and 12 CPU cores
# Training with a batch size of 64 and 45 epochs
# Dataset    | CIFAR-10 | ImageNet200 | ImageNet100 | ImageNet-1k |
#------------|----------|-------------|-------------|-------------|
# AlexNet    | 3.5 hrs  |     hrs     |             |             |
# ResNet18   | 1.5 hrs  |  14 hrs     |   12 hrs    |
# ResNet34   | 5.5 hrs  |  14 hrs     |   13 hrs    |
# ResNet50   | 6.5 hrs  |  15 hrs     |   15 hrs    |

module load CUDA/11.1.1-GCC-10.2.0
source /data/p288722/python_venv/pushpull-conv/bin/activate
#
#train_script="$HOME/git_code/pushpull-conv/project/train_flow.py"
#logs_dir="/data/p288722/runtime_data/pushpull-conv"
#task="retrieval"
#dataset_name="cifar10"
#dataset_dir="/data/p288722/datasets/cifar"
#corrupted_dataset_dir="/data/p288722/datasets/cifar/CIFAR-10-C"
#corrupted_dataset_name="cifar10-c"
#
#model="resnet50"
#experiment_name="${model}_${dataset_name}_${task}"
#common_train_args="--img_size 32 --model ${model} --hash_length 64 --quantization_weight 1e-4 --bias --num_workers 12 --batch_size 256 --max_epochs 50 --weight_decay 5e-4 --lr_base 5e-2 --task ${task} --logs_dir ${logs_dir} --experiment_name ${experiment_name} --dataset_dir ${dataset_dir} --dataset_name ${dataset_name}"
#base_dir="$logs_dir/$experiment_name"
#predict_script="$HOME/git_code/pushpull-conv/project/predict_flow.py"
#corruption_types="gaussian_noise shot_noise impulse_noise defocus_blur glass_blur  motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression"
#baseline_model_logs_dir="$base_dir/${model}"
#common_predict_args="--img_size 32 --model ${model} --num_workers 12 --corruption_types $corruption_types --task ${task} --dataset_dir ${dataset_dir} --dataset_name ${dataset_name} --corrupted_dataset_dir ${corrupted_dataset_dir} --corrupted_dataset_name ${corrupted_dataset_name}"
#
#case ${SLURM_ARRAY_TASK_ID} in
##python ${train_script} --no-use_push_pull --logs_version 0 ${common_train_args}
##python ${train_script} --push_kernel_size 3 --pull_kernel_size 3 --avg_kernel_size 3 --pull_inhibition_strength 1 --logs_version 1 ${common_train_args}
##python ${train_script} --push_kernel_size 3 --pull_kernel_size 3 --avg_kernel_size 3 --pull_inhibition_strength 2 --logs_version 2 ${common_train_args}
#3) python ${train_script} --push_kernel_size 3 --pull_kernel_size 3 --avg_kernel_size 3 --pull_inhibition_strength 3 --logs_version 3 ${common_train_args} ;;
##python ${train_script} --push_kernel_size 3 --pull_kernel_size 3 --avg_kernel_size 3 --pull_inhibition_strength 4 --logs_version 4 ${common_train_args}
#5) python ${train_script} --push_kernel_size 3 --pull_kernel_size 3 --avg_kernel_size 5 --pull_inhibition_strength 1 --logs_version 5 ${common_train_args} ;;
##python ${train_script} --push_kernel_size 3 --pull_kernel_size 3 --avg_kernel_size 5 --pull_inhibition_strength 2 --logs_version 6 ${common_train_args}
#7) python ${train_script} --push_kernel_size 3 --pull_kernel_size 3 --avg_kernel_size 5 --pull_inhibition_strength 3 --logs_version 7 ${common_train_args} ;;
#8) python ${train_script} --push_kernel_size 3 --pull_kernel_size 3 --avg_kernel_size 5 --pull_inhibition_strength 4 --logs_version 8 ${common_train_args} ;;
#esac
#
#case ${SLURM_ARRAY_TASK_ID} in
##mv "${base_dir}/version_0" "$base_dir/${model}"
##mv "${base_dir}/version_1" "$base_dir/${model}_pp3x3_avg3_inh1"
##mv "${base_dir}/version_2" "$base_dir/${model}_pp3x3_avg3_inh2"
#3) mv "${base_dir}/version_3" "$base_dir/${model}_pp3x3_avg3_inh3" ;;
##mv "${base_dir}/version_4" "$base_dir/${model}_pp3x3_avg3_inh4"
#5) mv "${base_dir}/version_5" "$base_dir/${model}_pp3x3_avg5_inh1" ;;
##mv "${base_dir}/version_6" "$base_dir/${model}_pp3x3_avg5_inh2"
#7) mv "${base_dir}/version_7" "$base_dir/${model}_pp3x3_avg5_inh3" ;;
#8) mv "${base_dir}/version_8" "$base_dir/${model}_pp3x3_avg5_inh4" ;;
#esac
#
#case ${SLURM_ARRAY_TASK_ID} in
##python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}" ${common_predict_args} --no-use_push_pull
##python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_pp3x3_avg3_inh1" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir}
##python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_pp3x3_avg3_inh2" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir}
#3) python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_pp3x3_avg3_inh3" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir} ;;
##python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_pp3x3_avg3_inh4" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir}
#5) python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_pp3x3_avg5_inh1" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir} ;;
##python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_pp3x3_avg5_inh2" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir}
#7) python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_pp3x3_avg5_inh3" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir} ;;
#8) python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_pp3x3_avg5_inh4" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir} ;;
#esac

train_script="$HOME/git_code/pushpull-conv/project/train_flow.py"
logs_dir="/data/p288722/runtime_data/pushpull-conv"
task="retrieval"
dataset_name="imagenet200"
dataset_dir="/data/p288722/datasets/imagenet"
corrupted_dataset_dir="/scratch/p288722/datasets/imagenet/imagenet-c"
corrupted_dataset_name="imagenet200-c"
model="resnet18"
experiment_name="${model}_${dataset_name}_${task}"
common_train_args="--img_size 224 --model ${model} --hash_length 64 --quantization_weight 1e-4 --no-scale_the_outputs --bias --num_workers 12 --batch_size 64 --max_epochs 50 --weight_decay 5e-5 --lr_base 5e-2 --task ${task} --logs_dir ${logs_dir} --experiment_name ${experiment_name} --dataset_dir ${dataset_dir} --dataset_name ${dataset_name}"
base_dir="$logs_dir/$experiment_name"
predict_script="$HOME/git_code/pushpull-conv/project/predict_flow.py"
corruption_types="gaussian_noise shot_noise impulse_noise defocus_blur glass_blur  motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression"
baseline_model_logs_dir="$base_dir/${model}"
common_predict_args="--img_size 224 --model ${model} --num_workers 12 --corruption_types $corruption_types --task ${task} --dataset_dir ${dataset_dir} --dataset_name ${dataset_name} --corrupted_dataset_dir ${corrupted_dataset_dir} --corrupted_dataset_name ${corrupted_dataset_name}"

 case ${SLURM_ARRAY_TASK_ID} in
 0) python ${train_script} --no-use_push_pull --logs_version ${SLURM_ARRAY_TASK_ID} ${common_train_args} --ckpt "${base_dir}/version_${SLURM_ARRAY_TASK_ID}/checkpoints/last.ckpt" ${common_train_args} ;;
 1) python ${train_script} --push_kernel_size 7 --pull_kernel_size 7 --avg_kernel_size 3 --pull_inhibition_strength 1 --logs_version ${SLURM_ARRAY_TASK_ID} --ckpt "${base_dir}/version_${SLURM_ARRAY_TASK_ID}/checkpoints/last.ckpt" ${common_train_args} ;;
 2) python ${train_script} --push_kernel_size 7 --pull_kernel_size 7 --avg_kernel_size 3 --pull_inhibition_strength 2 --logs_version ${SLURM_ARRAY_TASK_ID} --ckpt "${base_dir}/version_${SLURM_ARRAY_TASK_ID}/checkpoints/epoch=36-loss_val=0.34.ckpt" ${common_train_args} ;;
 3) python ${train_script} --push_kernel_size 7 --pull_kernel_size 7 --avg_kernel_size 3 --pull_inhibition_strength 3 --logs_version ${SLURM_ARRAY_TASK_ID} --ckpt "${base_dir}/version_${SLURM_ARRAY_TASK_ID}/checkpoints/last.ckpt" ${common_train_args} ;;
 4) python ${train_script} --push_kernel_size 7 --pull_kernel_size 7 --avg_kernel_size 3 --pull_inhibition_strength 4 --logs_version ${SLURM_ARRAY_TASK_ID} --ckpt "${base_dir}/version_${SLURM_ARRAY_TASK_ID}/checkpoints/last.ckpt" ${common_train_args} ;;
 5) python ${train_script} --push_kernel_size 7 --pull_kernel_size 7 --avg_kernel_size 5 --pull_inhibition_strength 1 --logs_version ${SLURM_ARRAY_TASK_ID} --ckpt "${base_dir}/version_${SLURM_ARRAY_TASK_ID}/checkpoints/last_old.ckpt" ${common_train_args} ;;
 6) python ${train_script} --push_kernel_size 7 --pull_kernel_size 7 --avg_kernel_size 5 --pull_inhibition_strength 2 --logs_version ${SLURM_ARRAY_TASK_ID} --ckpt "${base_dir}/version_${SLURM_ARRAY_TASK_ID}/checkpoints/last.ckpt" ${common_train_args} ;;
 7) python ${train_script} --push_kernel_size 7 --pull_kernel_size 7 --avg_kernel_size 5 --pull_inhibition_strength 3 --logs_version ${SLURM_ARRAY_TASK_ID} --ckpt "${base_dir}/version_${SLURM_ARRAY_TASK_ID}/checkpoints/last.ckpt" ${common_train_args} ;;
 8) python ${train_script} --push_kernel_size 7 --pull_kernel_size 7 --avg_kernel_size 5 --pull_inhibition_strength 4 --logs_version ${SLURM_ARRAY_TASK_ID} --ckpt "${base_dir}/version_${SLURM_ARRAY_TASK_ID}/checkpoints/last.ckpt" ${common_train_args} ;;
 esac

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


#python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_pp7x7_avg3_inh1" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir}
#python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_pp7x7_avg3_inh2" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir}
#python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_pp7x7_avg3_inh3" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir}
#python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_pp7x7_avg3_inh4" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir}
#python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_pp7x7_avg5_inh1" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir}
#python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_pp7x7_avg5_inh2" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir}
#python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_pp7x7_avg5_inh3" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir}
#python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_pp7x7_avg5_inh4" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir}
