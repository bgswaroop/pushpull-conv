#!/bin/bash
#SBATCH --job-name=train_in100
#SBATCH --time=2:00:00
#SBATCH --mem=60g
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12

# SLURM Notation used above
# %x - Name of the Job
# %A - JOB ID
# %a - TASK ID

# The following times (upperbound) are for V100 GPU with 100GB RAM and 12 CPU cores
# Training with a batch size of 64 and 45 epochs
# Dataset    | CIFAR-10 | ImageNet100 | ImageNet200 | ImageNet-1k |
#------------|----------|-------------|-------------|-------------|
# AlexNet    | 3.5 hrs  |     hrs     |             |             |
# ResNet18   | 1.5 hrs  |  12 hrs     |   14 hrs    |             |
# ResNet34   | 5.5 hrs  |  13 hrs     |   14 hrs    |             |
# ResNet50   | 6.5 hrs  |  15 hrs     |   15 hrs    |             |

# The following times (upperbound) are for A100 GPU with 60GB RAM and 12 CPU cores
# Training with a batch size of 128 and 20 epochs
# Dataset    | CIFAR-10 | ImageNet100 | ImageNet200 | ImageNet-1k |
#------------|----------|-------------|-------------|-------------|
# AlexNet    | --- hrs  |     hrs     |             |             |
# ResNet18   | --- hrs  |  -- hrs     |   -- hrs    |             |
# ResNet34   | --- hrs  |  -- hrs     |   -- hrs    |             |
# ResNet50   | --- hrs  |  -- hrs     |   -- hrs    |             |

module load CUDA
module load Python/3.11.3-GCCcore-12.3.0

which python
source "$HOME/.virtualenvs/pushpull-conv/bin/activate"

module list
which python

tar -xf /scratch/p288722/data/imagenet.tar -C "$TMPDIR" --warning=no-unknown-keyword
tar -xf /scratch/p288722/data/imagenet-c.tar -C "$TMPDIR" --warning=no-unknown-keyword
SECONDS=0;
echo deleting unwanted files from imagenet
find "$TMPDIR/imagenet" -name ".*" -delete
echo Time taken to delete $SECONDS sec
SECONDS=0;
echo deleting unwanted files from imagenet-c
find "$TMPDIR/imagenet-c" -name ".*" -delete
echo Time taken to delete $SECONDS sec

#rm /tmp/imagenet/val/n01514668/._ILSVRC2012_val_00000329.JPEG
#rm /tmp/imagenet/val/n01514668/._ILSVRC2012_val_00000911.JPEG

train_script="$HOME/git_code/pushpull-conv/project/train_flow.py"
logs_dir="/scratch/p288722/runtime_data/pushpull-conv"
task="classification"
dataset_name="imagenet100"
dataset_dir="$TMPDIR/imagenet"
corrupted_dataset_dir="$TMPDIR/imagenet-c"
corrupted_dataset_name="imagenet100-c"
model="resnet50"
experiment_name="${model}_${dataset_name}_${task}"
common_train_args="--accelerator gpu --img_size 224 --model ${model} --hash_length 64 --quantization_weight 1e-4 --num_workers 12 --batch_size 64 --max_epochs 50 --weight_decay 5e-5 --lr_base 5e-2 --task ${task} --logs_dir ${logs_dir} --experiment_name ${experiment_name} --dataset_dir ${dataset_dir} --dataset_name ${dataset_name}"
base_dir="$logs_dir/$experiment_name"
predict_script="$HOME/git_code/pushpull-conv/project/predict_flow.py"
corruption_types="gaussian_noise shot_noise impulse_noise defocus_blur glass_blur  motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression"
baseline_model_logs_dir="$base_dir/${model}"
common_predict_args="--accelerator gpu --img_size 224 --model ${model} --num_workers 12 --corruption_types $corruption_types --task ${task} --dataset_dir ${dataset_dir} --dataset_name ${dataset_name} --corrupted_dataset_dir ${corrupted_dataset_dir} --corrupted_dataset_name ${corrupted_dataset_name}"

echo python ${train_script} --avg_kernel_size 3 --pull_inhibition_strength 1 --logs_version 2 ${common_train_args}
python ${train_script} --avg_kernel_size 3 --pull_inhibition_strength 1 --logs_version 2 ${common_train_args}

#
# case ${SLURM_ARRAY_TASK_ID} in
# 0) python ${train_script} --no-use_push_pull --logs_version ${SLURM_ARRAY_TASK_ID} ${common_train_args} --ckpt "${base_dir}/version_${SLURM_ARRAY_TASK_ID}/checkpoints/last.ckpt" ${common_train_args} ;;
# 1) python ${train_script} --avg_kernel_size 3 --pull_inhibition_strength 1 --logs_version ${SLURM_ARRAY_TASK_ID} --ckpt "${base_dir}/version_${SLURM_ARRAY_TASK_ID}/checkpoints/last.ckpt" ${common_train_args} ;;
# 2) python ${train_script} --avg_kernel_size 3 --pull_inhibition_strength 2 --logs_version ${SLURM_ARRAY_TASK_ID} --ckpt "${base_dir}/version_${SLURM_ARRAY_TASK_ID}/checkpoints/epoch=36-loss_val=0.34.ckpt" ${common_train_args} ;;
# 3) python ${train_script} --avg_kernel_size 3 --pull_inhibition_strength 3 --logs_version ${SLURM_ARRAY_TASK_ID} --ckpt "${base_dir}/version_${SLURM_ARRAY_TASK_ID}/checkpoints/last.ckpt" ${common_train_args} ;;
# 4) python ${train_script} --avg_kernel_size 3 --pull_inhibition_strength 4 --logs_version ${SLURM_ARRAY_TASK_ID} --ckpt "${base_dir}/version_${SLURM_ARRAY_TASK_ID}/checkpoints/last.ckpt" ${common_train_args} ;;
# 5) python ${train_script} --avg_kernel_size 5 --pull_inhibition_strength 1 --logs_version ${SLURM_ARRAY_TASK_ID} --ckpt "${base_dir}/version_${SLURM_ARRAY_TASK_ID}/checkpoints/last_old.ckpt" ${common_train_args} ;;
# 6) python ${train_script} --avg_kernel_size 5 --pull_inhibition_strength 2 --logs_version ${SLURM_ARRAY_TASK_ID} --ckpt "${base_dir}/version_${SLURM_ARRAY_TASK_ID}/checkpoints/last.ckpt" ${common_train_args} ;;
# 7) python ${train_script} --avg_kernel_size 5 --pull_inhibition_strength 3 --logs_version ${SLURM_ARRAY_TASK_ID} --ckpt "${base_dir}/version_${SLURM_ARRAY_TASK_ID}/checkpoints/last.ckpt" ${common_train_args} ;;
# 8) python ${train_script} --avg_kernel_size 5 --pull_inhibition_strength 4 --logs_version ${SLURM_ARRAY_TASK_ID} --ckpt "${base_dir}/version_${SLURM_ARRAY_TASK_ID}/checkpoints/last.ckpt" ${common_train_args} ;;
# esac
#
# case ${SLURM_ARRAY_TASK_ID} in
# 0) mv "${base_dir}/version_${SLURM_ARRAY_TASK_ID}" "$base_dir/${model}" ;;
# 1) mv "${base_dir}/version_${SLURM_ARRAY_TASK_ID}" "$base_dir/${model}_avg3_inh1" ;;
# 2) mv "${base_dir}/version_${SLURM_ARRAY_TASK_ID}" "$base_dir/${model}_avg3_inh2" ;;
# 3) mv "${base_dir}/version_${SLURM_ARRAY_TASK_ID}" "$base_dir/${model}_avg3_inh3" ;;
# 4) mv "${base_dir}/version_${SLURM_ARRAY_TASK_ID}" "$base_dir/${model}_avg3_inh4" ;;
# 5) mv "${base_dir}/version_${SLURM_ARRAY_TASK_ID}" "$base_dir/${model}_avg5_inh1" ;;
# 6) mv "${base_dir}/version_${SLURM_ARRAY_TASK_ID}" "$base_dir/${model}_avg5_inh2" ;;
# 7) mv "${base_dir}/version_${SLURM_ARRAY_TASK_ID}" "$base_dir/${model}_avg5_inh3" ;;
# 8) mv "${base_dir}/version_${SLURM_ARRAY_TASK_ID}" "$base_dir/${model}_avg5_inh4" ;;
# esac

# case ${SLURM_ARRAY_TASK_ID} in
# 0) python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}" ${common_predict_args} --no-use_push_pull ;;
# 1) python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_avg3_inh1" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir} ;;
# 2) python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_avg3_inh2" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir} ;;
# 3) python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_avg3_inh3" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir} ;;
# 4) python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_avg3_inh4" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir} ;;
# 5) python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_avg5_inh1" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir} ;;
# 6) python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_avg5_inh2" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir} ;;
# 7) python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_avg5_inh3" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir} ;;
# 8) python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_avg5_inh4" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir} ;;
# esac
