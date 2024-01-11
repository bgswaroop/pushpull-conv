#!/bin/bash

flow="predict"

if [[ $(hostname -s) = a40-test ]]; then
  echo a40-test
  project_dir="/home/p288722/git_code/pushpull-conv"
  logs_dir="/data2/p288722/runtime_data/pushpull-conv"
  dataset_dir="/data2/p288722/datasets"
elif [[ $(hostname -s) = 20220913_U18_2GPU_Guru ]]; then
  echo 20220913_U18_2GPU_Guru
  project_dir="/home/guru/git_code/pushpull-conv"
  logs_dir="/home/guru/runtime_data/pushpull-conv"
  dataset_dir="/home/guru/datasets"
elif [[ $(hostname -s) = 20220920_U18_2GPU_Guru ]]; then
  echo 20220920_U18_2GPU_Guru
  project_dir="/home/guru/git_code/pushpull-conv"
  logs_dir="/home/guru/runtime_data/pushpull-conv"
  dataset_dir="/home/guru/datasets"
else
  echo none of the above! exiting script.
  exit 64
fi

if [[ $flow = train ]]; then
  echo running train flow

  save_dir="/home/guru/runtime_data/pushpull-conv/dev_augmix_imagenet_sota_impl/resnet50_augmix_180epc"
  nohup python "${project_dir}/project/models/classification/augmix_ICLR2020/imagenet.py" --epochs 180 --num-workers 36 --print-freq 100 --save ${save_dir} "${dataset_dir}/imagenet/Data/CLS-LOC" "${dataset_dir}/imagenet/imagenet-c" > "${save_dir}/logs_train.out" 2>&1 &

#  experiment_name="dev_data_aug_imagenet"
#  base_dir="${logs_dir}/${experiment_name}"
#  common_args="--augmentation AugMix --accelerator gpu --img_size 224 --model resnet50 --hash_length 64 --quantization_weight 1e-4 --num_workers 24 --task classification --logs_dir ${logs_dir} --experiment_name ${experiment_name} --dataset_dir ${dataset_dir}/imagenet --dataset_name imagenet"
#
#  exp_id=0
#  mkdir -p "${base_dir}/version_${exp_id}"
#  nohup python "${project_dir}/project/train_flow.py" --no-use_push_pull --devices 0, --logs_version $exp_id $common_args> "${base_dir}/version_${exp_id}/logs.out" 2>&1 &
#
#  exp_id=1
#  mkdir -p "${base_dir}/version_${exp_id}"
#  nohup python "${project_dir}/project/train_flow.py" --use_push_pull --avg_kernel_size 3 --trainable_pull_inhibition 1 --devices 1, --logs_version $exp_id $common_args> "${base_dir}/version_${exp_id}/logs.out" 2>&1 &
#
#  exp_id=2
#  mkdir -p "${base_dir}/version_${exp_id}"
#  nohup python "${project_dir}/project/train_flow.py" --use_push_pull --avg_kernel_size 5 --trainable_pull_inhibition 1 --devices 0, --logs_version $exp_id $common_args> "${base_dir}/version_${exp_id}/logs.out" 2>&1 &

elif [[ $flow = predict ]]; then
  echo running predict flow
  baseline_model="/home/guru/runtime_data/pushpull-conv/dev_data_aug_imagenet/resnet50"
  corruption_types="gaussian_noise shot_noise impulse_noise defocus_blur glass_blur motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression "
  common_args="--baseline_model $baseline_model --accelerator gpu --img_size 224 --model resnet50 --num_workers 12 --corruption_types $corruption_types --task classification --dataset_dir ${dataset_dir}/imagenet --dataset_name imagenet --corrupted_dataset_dir ${dataset_dir}/imagenet/imagenet-c --corrupted_dataset_name imagenet-c"

  predict_model="/home/guru/runtime_data/pushpull-conv/dev_augmix_imagenet_sota_impl/resnet50_augmix_90epc"
  nohup python "${project_dir}/project/predict_flow.py" --no-use_push_pull --devices 0, --predict_model_logs_dir $predict_model $common_args > "${predict_model}/logs_predict.out" 2>&1 &

#  predict_model="/home/guru/runtime_data/pushpull-conv/dev_data_aug_imagenet/resnet50_augmix"
#  nohup python "${project_dir}/project/predict_flow.py" --no-use_push_pull --devices 0, --predict_model_logs_dir $predict_model $common_args > "${predict_model}/logs_predict.out" 2>&1 &

#  predict_model="/home/guru/runtime_data/pushpull-conv/dev_data_aug_imagenet/resnet50_augmix_avg3"
#  nohup python "${project_dir}/project/predict_flow.py" --use_push_pull --devices 1, --predict_model_logs_dir $predict_model $common_args > "${predict_model}/logs_predict.out" 2>&1 &

#  predict_model="/home/guru/runtime_data/pushpull-conv/dev_data_aug_imagenet/resnet50_augmix_avg5"
#  nohup python "${project_dir}/project/predict_flow.py" --use_push_pull --devices 0, --predict_model_logs_dir $predict_model $common_args > "${predict_model}/logs_predict.out" 2>&1 &

else
  echo Invalid flow option, Exiting the script!
  exit 64
fi
