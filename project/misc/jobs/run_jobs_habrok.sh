#!/bin/bash
#SBATCH --job-name=r50_in
#SBATCH --time=4:00:00
#SBATCH --mem=64gb
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --array=7

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

train_script="$HOME/git_code/pushpull-conv/project/train_flow.py"
logs_dir="/scratch/p288722/runtime_data/surround_suppression"
task="classification"
dataset_name="imagenet"
corrupted_dataset_name="imagenet-c"

if [ $dataset_name == "imagenet" ] || [ $dataset_name == "imagenet100" ] ||[ $dataset_name == "imagenet200" ]
then
    session_id=$(openssl rand -hex 4)
    mkdir "$TMPDIR/dataset_${session_id}"
    dataset_dir="$TMPDIR/dataset_${session_id}/imagenet"
    corrupted_dataset_dir="$TMPDIR/dataset_${session_id}/imagenet-c"
elif [ $dataset_name == "cifar10" ]; 
then
    dataset_dir="/scratch/p288722/data/cifar"
    corrupted_dataset_dir="/scratch/p288722/data/cifar10-c"
fi

model="resnet50"
experiment_name="${model}_${dataset_name}_${task}"
common_train_args="--accelerator gpu --img_size 224 --model ${model} --hash_length 64 --quantization_weight 1e-4 --num_workers $SLURM_CPUS_PER_TASK --batch_size 64 --max_epochs 20 --weight_decay 5e-5 --lr_base 5e-2 --task ${task} --logs_dir ${logs_dir} --experiment_name ${experiment_name} --dataset_dir ${dataset_dir} --dataset_name ${dataset_name}"
base_dir="$logs_dir/$experiment_name"
predict_script="$HOME/git_code/pushpull-conv/project/predict_flow.py"
corruption_types="gaussian_noise shot_noise impulse_noise defocus_blur glass_blur  motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression"
baseline_model_logs_dir="$base_dir/${model}"
common_predict_args="--accelerator gpu --img_size 224 --model ${model} --num_workers $SLURM_CPUS_PER_TASK --corruption_types $corruption_types --task ${task} --dataset_dir ${dataset_dir} --dataset_name ${dataset_name} --corrupted_dataset_dir ${corrupted_dataset_dir} --corrupted_dataset_name ${corrupted_dataset_name}"


if [ $dataset_name == "imagenet" ] || [ $dataset_name == "imagenet100" ] ||[ $dataset_name == "imagenet200" ]
then
    SECONDS=0;
    echo "extracting files from imagenet and imagenet-c"
    tar -xf /scratch/p288722/data/imagenet.tar -C "$TMPDIR/dataset_${session_id}" --warning=no-unknown-keyword
    tar -xf /scratch/p288722/data/imagenet-c.tar -C "$TMPDIR/dataset_${session_id}" --warning=no-unknown-keyword
    echo "deleting unwanted files from ${dataset_dir}"
    find $dataset_dir -name ".*" -delete
    echo "deleting unwanted files from ${corrupted_dataset_dir}"
    find $corrupted_dataset_dir -name ".*" -delete
    echo "Time taken to extract & delete unnecessary files ${SECONDS} sec"
fi

# --avg_kernel_size 3 --trainable_pull_inhibition

#ver=1
#echo python ${train_script} --no-use_push_pull --logs_version $ver --augmentation TrivialAugment ${common_train_args}
#export CUDA_LAUNCH_BLOCKING=1 python ${train_script} --no-use_push_pull --logs_version $ver --augmentation TrivialAugment ${common_train_args}
#mv "${base_dir}/version_${ver}" "$base_dir/${model}_TrivialAugment"
#export CUDA_LAUNCH_BLOCKING=1 echo python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_TrivialAugment" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir}
#python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_TrivialAugment" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir}

#ver=1
#echo python ${train_script} --avg_kernel_size 3 --logs_version $ver ${common_train_args}
#python ${train_script} --avg_kernel_size 3 --logs_version $ver ${common_train_args}
#mv "${base_dir}/version_${ver}" "$base_dir/${model}_TrivialAugment_avg3"
#export CUDA_LAUNCH_BLOCKING=1 echo python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_TrivialAugment_avg3" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir}
#python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_TrivialAugment_avg3" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir}

export CUDA_LAUNCH_BLOCKING=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

 case ${SLURM_ARRAY_TASK_ID} in
 0) python ${train_script} --no-use_push_pull --logs_version ${SLURM_ARRAY_TASK_ID}  ${common_train_args} ;;
 1) python ${train_script} --avg_kernel_size 0 --no-trainable_pull_inhibition --logs_version ${SLURM_ARRAY_TASK_ID}  ${common_train_args} ;;
 2) python ${train_script} --avg_kernel_size 3 --no-trainable_pull_inhibition --logs_version ${SLURM_ARRAY_TASK_ID}  ${common_train_args} ;;
 3) python ${train_script} --avg_kernel_size 5 --no-trainable_pull_inhibition --logs_version ${SLURM_ARRAY_TASK_ID}  ${common_train_args} ;;
 4) python ${train_script} --avg_kernel_size 0 --trainable_pull_inhibition --logs_version ${SLURM_ARRAY_TASK_ID}  ${common_train_args} ;;
 5) python ${train_script} --avg_kernel_size 3 --trainable_pull_inhibition --logs_version ${SLURM_ARRAY_TASK_ID}  ${common_train_args} ;;
 6) python ${train_script} --avg_kernel_size 5 --trainable_pull_inhibition --logs_version ${SLURM_ARRAY_TASK_ID}  ${common_train_args} ;;
 7) python ${train_script} --avg_kernel_size 0 --no-trainable_pull_inhibition --pull_inhibition_strength 1 --logs_version ${SLURM_ARRAY_TASK_ID}  ${common_train_args} ;;

 esac

 case ${SLURM_ARRAY_TASK_ID} in
 0) mv "${base_dir}/version_${SLURM_ARRAY_TASK_ID}" "$base_dir/${model}" ;;
 1) mv "${base_dir}/version_${SLURM_ARRAY_TASK_ID}" "$base_dir/${model}_avg0" ;;
 2) mv "${base_dir}/version_${SLURM_ARRAY_TASK_ID}" "$base_dir/${model}_avg3" ;;
 3) mv "${base_dir}/version_${SLURM_ARRAY_TASK_ID}" "$base_dir/${model}_avg5" ;;
 4) mv "${base_dir}/version_${SLURM_ARRAY_TASK_ID}" "$base_dir/${model}_inht_avg0" ;;
 5) mv "${base_dir}/version_${SLURM_ARRAY_TASK_ID}" "$base_dir/${model}_inht_avg3" ;;
 6) mv "${base_dir}/version_${SLURM_ARRAY_TASK_ID}" "$base_dir/${model}_inht_avg5" ;;
 7) mv "${base_dir}/version_${SLURM_ARRAY_TASK_ID}" "$base_dir/${model}_sigma2" ;;
 esac

 case ${SLURM_ARRAY_TASK_ID} in
 0) python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}" ${common_predict_args} --no-use_push_pull ;;
 1) python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_avg0" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir} ;;
 2) python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_avg3" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir} ;;
 3) python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_avg5" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir} ;;
 4) python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_inht_avg0" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir} ;;
 5) python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_inht_avg3" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir} ;;
 6) python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_inht_avg5" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir} ;;
 7) python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_sigma2" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir} ;;
 esac
