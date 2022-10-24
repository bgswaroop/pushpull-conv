#!/bin/bash

train_script="$HOME/git_code/pushpull-conv/project/train_flow.py"
logs_dir="/data2/p288722/runtime_data/pushpull-conv/from_computo04/"
task="classification"
dataset_name="imagenet100"
dataset_dir="/data2/p288722/datasets/imagenet"
corrupted_dataset_dir="/data2/p288722/datasets/imagenet/imagenet-c"
corrupted_dataset_name="imagenet100-c"
model="resnet18"
experiment_name="${model}_${dataset_name}_${task}_hparams"
common_train_args="--accelerator gpu --img_size 224 --model ${model} --hash_length 64 --quantization_weight 1e-4 --num_workers 16 --batch_size 64 --max_epochs 50 --weight_decay 5e-5 --lr_base 5e-2 --task ${task} --logs_dir ${logs_dir} --experiment_name ${experiment_name} --dataset_dir ${dataset_dir} --dataset_name ${dataset_name}"
base_dir="$logs_dir/$experiment_name"
predict_script="$HOME/git_code/pushpull-conv/project/predict_flow.py"
corruption_types="gaussian_noise shot_noise impulse_noise defocus_blur glass_blur  motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression"
baseline_model_logs_dir="$base_dir/${model}"
teacher_ckpt="/data2/p288722/runtime_data/pushpull-conv/from_computo04/resnet18_imagenet100_classification_hparams/resnet18/checkpoints/last.ckpt"
common_predict_args="--accelerator gpu --img_size 224 --model ${model} --num_workers 8 --corruption_types $corruption_types --task ${task} --dataset_dir ${dataset_dir} --dataset_name ${dataset_name} --corrupted_dataset_dir ${corrupted_dataset_dir} --corrupted_dataset_name ${corrupted_dataset_name}"

#i=0
#for alpha in 0.3
#do
#  for temp in 2 3
#  do
#     mkdir -p "${base_dir}/version_${i}"
#     nohup python ${train_script} --training_type student --teacher_ckpt $teacher_ckpt --no-use_push_pull --distillation_loss_alpha $alpha --distillation_loss_temp $temp --logs_version ${i} ${common_train_args} > "${base_dir}/version_${i}/logs_train.out" 2>&1 &
#     i=$(( $i + 1 ));
#  done
#done

i=0
for alpha in 0.3
do
  for temp in 2 3
  do
    version_name="${model}_temp${temp}_alpha${alpha}"
#    mv "${base_dir}/version_${i}" "$base_dir/${version_name}"
    nohup python ${predict_script} --predict_model_logs_dir "${base_dir}/${version_name}" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir} > "$base_dir/${version_name}/logs_predict.out" 2>&1 &
    i=$(( $i + 1 ));
  done
done


#python ${train_script} --no-use_push_pull --logs_version 0 ${common_train_args}
#python ${train_script} --avg_kernel_size 3 --pull_inhibition_strength 1 --logs_version 1 ${common_train_args}
#python ${train_script} --avg_kernel_size 3 --pull_inhibition_strength 2 --logs_version 2 ${common_train_args}
#python ${train_script} --avg_kernel_size 3 --pull_inhibition_strength 3 --logs_version 3 ${common_train_args}
#python ${train_script} --avg_kernel_size 3 --pull_inhibition_strength 4 --logs_version 4 ${common_train_args}
#python ${train_script} --avg_kernel_size 5 --pull_inhibition_strength 1 --logs_version 5 ${common_train_args}
#python ${train_script} --avg_kernel_size 5 --pull_inhibition_strength 2 --logs_version 6 ${common_train_args}
#python ${train_script} --avg_kernel_size 5 --pull_inhibition_strength 3 --logs_version 7 ${common_train_args}
#python ${train_script} --avg_kernel_size 5 --pull_inhibition_strength 4 --logs_version 8 ${common_train_args}

#mv "${base_dir}/version_0" "$base_dir/${model}"
#mv "${base_dir}/version_1" "$base_dir/${model}_avg3_inh1"
#mv "${base_dir}/version_2" "$base_dir/${model}_avg3_inh2"
#mv "${base_dir}/version_3" "$base_dir/${model}_avg3_inh3"
#mv "${base_dir}/version_4" "$base_dir/${model}_avg3_inh4"
#mv "${base_dir}/version_5" "$base_dir/${model}_avg5_inh1"
#mv "${base_dir}/version_6" "$base_dir/${model}_avg5_inh2"
#mv "${base_dir}/version_7" "$base_dir/${model}_avg5_inh3"
#mv "${base_dir}/version_8" "$base_dir/${model}_avg5_inh4"

#python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}" ${common_predict_args} --no-use_push_pull
#python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_avg3_inh1" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir}
#python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_avg3_inh2" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir}
#python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_avg3_inh3" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir}
#python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_avg3_inh4" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir}
#python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_avg5_inh1" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir}
#python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_avg5_inh2" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir}
#python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_avg5_inh3" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir}
#python ${predict_script} --predict_model_logs_dir "${base_dir}/${model}_avg5_inh4" ${common_predict_args} --baseline_model_logs_dir ${baseline_model_logs_dir}
