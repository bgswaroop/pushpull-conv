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
experiment_name="${model}_${dataset_name}_${task}_hparams"
common_train_args="--accelerator "gpu" --img_size 224 --model ${model} --hash_length 64 --quantization_weight 1e-4 --num_workers 8 --task ${task} --logs_dir ${logs_dir} --experiment_name ${experiment_name} --dataset_dir ${dataset_dir} --dataset_name ${dataset_name}"
base_dir="$logs_dir/$experiment_name"
predict_script="$HOME/git_code/pushpull-conv/project/predict_flow.py"
corruption_types="gaussian_noise shot_noise impulse_noise defocus_blur glass_blur  motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression"
baseline_model_logs_dir="$base_dir/${model}"
teacher_ckpt="/home/guru/runtime_data/pushpull-conv/resnet18_imagenet100_classification_hparams/resnet18/checkpoints/last.ckpt"
common_predict_args="--accelerator "gpu" --img_size 224 --model ${model} --num_workers 8 --corruption_types $corruption_types --task ${task} --dataset_dir ${dataset_dir} --dataset_name ${dataset_name} --corrupted_dataset_dir ${corrupted_dataset_dir} --corrupted_dataset_name ${corrupted_dataset_name}"


#i=0
#for alpha in 0.1 0.2 0.3 0.4
#do
#  for temp in 1 2 3
#  do
#     mkdir -p "${base_dir}/version_${i}"
#     nohup python ${train_script} --training_type student --teacher_ckpt $teacher_ckpt --distillation_loss_alpha $alpha --distillation_loss_temp $temp --push_kernel_size 7 --pull_kernel_size 7 --avg_kernel_size 3 --pull_inhibition_strength 4  --logs_version ${i} ${common_train_args}  --devices "$((i%2))," > "${base_dir}/version_${i}/logs_train.out" 2>&1 &
#     i=$(( $i + 1 ));
#  done
#done

#pp_config="avg3_inh4"
#i=0
#for alpha in 0.1 0.2 0.3 0.4
#do
#  for temp in 1 2 3
#  do
#    version_name="${model}_${pp_config}_temp${temp}_alpha${alpha}"
#    mv "${base_dir}/version_${i}" "$base_dir/${version_name}"
#    i=$(( $i + 1 ));
#  done
#done

pp_config="avg3_inh4"
i=0
for alpha in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
  for temp in 1 2 3
  do
    version_name="${model}_${pp_config}_temp${temp}_alpha${alpha}"
    #    mv "${base_dir}/version_${i}" "$base_dir/${version_name}"
    nohup python ${predict_script} --predict_model_logs_dir "${base_dir}/${version_name}" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir} --devices "$((i%2))," > "$base_dir/${version_name}/logs_predict.out" 2>&1 &
    i=$(( $i + 1 ));
  done
done


#pp_config="avg3_inh4"
#
#mv "${base_dir}/version_0" "$base_dir/${model}"
#mv "${base_dir}/version_1" "$base_dir/${model}_${pp_config}_0.0"
#mv "${base_dir}/version_2" "$base_dir/${model}_${pp_config}_0.2"
#mv "${base_dir}/version_3" "$base_dir/${model}_${pp_config}_0.4"
#mv "${base_dir}/version_4" "$base_dir/${model}_${pp_config}_0.6"
#mv "${base_dir}/version_5" "$base_dir/${model}_${pp_config}_0.8"
#mv "${base_dir}/version_6" "$base_dir/${model}_${pp_config}_1.0"
#
#nohup python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}" ${common_predict_args} --no-use_push_pull --accelerator "gpu" --devices "0," > "${base_dir}/${model}/logs_predict.out" 2>&1 &
#nohup python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_${pp_config}_0.0" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir} --accelerator "gpu" --devices "1," > "${base_dir}/${model}_${pp_config}_0.0/logs_predict.out" 2>&1 &
#nohup python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_${pp_config}_0.2" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir} --accelerator "gpu" --devices "0," > "${base_dir}/${model}_${pp_config}_0.2/logs_predict.out" 2>&1 &
#nohup python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_${pp_config}_0.4" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir} --accelerator "gpu" --devices "1," > "${base_dir}/${model}_${pp_config}_0.4/logs_predict.out" 2>&1 &
#nohup python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_${pp_config}_0.6" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir} --accelerator "gpu" --devices "0," > "${base_dir}/${model}_${pp_config}_0.6/logs_predict.out" 2>&1 &
#nohup python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_${pp_config}_0.8" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir} --accelerator "gpu" --devices "1," > "${base_dir}/${model}_${pp_config}_0.8/logs_predict.out" 2>&1 &
#nohup python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_${pp_config}_1.0" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir} --accelerator "gpu" --devices "0," > "${base_dir}/${model}_${pp_config}_1.0/logs_predict.out" 2>&1 &
