import argparse
from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
# from ray.tune.integration.lightning.pytorch import TuneReportCallback
# from tqdm import tqdm

from data import get_dataset, get_augmentation
from models import get_classifier

torch.set_float32_matmul_precision('highest')


# class LitProgressBar(TQDMProgressBar):
#     def init_validation_tqdm(self):
#         bar = tqdm(disable=True, )
#         return bar


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', default=224, type=int, choices=[32, 224])
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--max_epochs', default=20, type=int)
    parser.add_argument('--dataset_dir', default='/home/guru/datasets/imagenet', type=str)
    parser.add_argument('--dataset_name', default='imagenet',
                        help="'cifar10', 'imagenet100', 'imagenet200', 'imagenet'"
                             "or add a suffix '_20pc' for a 20 percent stratified training subset."
                             "'_20pc' is an example, can be any float [1.0, 99.0]."
                             "Note - suffix is valid only for ImageNet variants")
    parser.add_argument('--ckpt', type=str, default=None, help='ckpt for the model to be trained based on '
                                                               '"training_type". Training to continue from this state.')
    parser.add_argument('--use-grayscale', action=argparse.BooleanOptionalAction, default=False)

    # Pytorch lightning args
    parser.add_argument('--logs_dir', required=True, type=str,
                        help='Path to save the logs/metrics during training')
    parser.add_argument('--experiment_name', default='dsh_push_pull', type=str)
    parser.add_argument('--num_workers', default=2, type=int,
                        help='how many subprocesses to use for data loading. ``0`` means that the data will be '
                             'loaded in the main process. (default: ``2``)')
    parser.add_argument('--logs_version', default=None, type=int)
    parser.add_argument('--accelerator', type=str, choices=['cpu', 'gpu', 'auto'], default='auto')
    parser.add_argument('--devices', type=str, default='auto')

    # Push Pull Convolutional Unit Params
    parser.add_argument('--use_push_pull', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--num_push_pull_layers', type=int, default=1)
    parser.add_argument('--avg_kernel_size', type=int, default=None, help='Size of the avg filter (int)')
    parser.add_argument('--pull_inhibition_strength', type=float, default=1.0)
    parser.add_argument('--trainable_pull_inhibition', action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument('--training_type', default='teacher', type=str, choices=['teacher', 'student'])
    parser.add_argument('--teacher_ckpt', type=str, default=None, help='ckpt to determine soft-targets for the student')
    parser.add_argument('--distillation_loss_temp', type=float, default=2.0)
    parser.add_argument('--distillation_loss_alpha', type=float, default=0.7)
    parser.add_argument('--model', default='resnet18', type=str, required=True)
    parser.add_argument('--task', default='classification', type=str, required=True,
                        choices=['classification', 'retrieval'])

    parser.add_argument('--lr_base', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)  # regularization
    parser.add_argument('--lr_step_size', type=int, default=30)
    parser.add_argument('--lr_gamma', type=float, default=0.1)
    parser.add_argument('--hash_length', type=int, default=64)
    parser.add_argument('--quantization_weight', type=float, default=1e-4)
    parser.add_argument('--augmentation', default='none', type=str,
                        choices=['AugMix', 'AutoAug', 'RandAug', 'none', 'TrivialAugment', 'prime'])

    # parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    assert Path(args.dataset_dir).exists(), f'dataset_dir {args.dataset_dir} does not exists!'
    Path(args.logs_dir).joinpath(args.experiment_name).mkdir(exist_ok=True, parents=True)

    # if args.use_push_pull:
    #     assert args.avg_kernel_size is not None, "Invalid config: use_push_pull=True but avg_kernel_size is not set!"

    if args.ckpt:
        args.ckpt = Path(args.ckpt).resolve()
        if not args.ckpt.exists():
            args.ckpt = None

    if args.training_type == 'student':
        assert args.teacher_ckpt is not None, 'Invalid config: teacher_ckpt is not set when training_type="student"'
        args.teacher_ckpt = args.teacher_ckpt.resolve()
        assert Path(args.teacher_ckpt).exists(), 'teacher_ckpt does not exists!'
        args.loss_type = 'distillation_loss'
    else:
        args.loss_type = 'default'  # cce for classification and CSQ for retrieval

    args.augmentation = get_augmentation(args.augmentation, args.dataset_name)

    torch.use_deterministic_algorithms(True, warn_only=True)

    if args.task == 'classification':
        if 'imagenet100' in args.dataset_name or 'imagenet200' in args.dataset_name:
            args.max_epochs = 20
            args.batch_size = 64  # use 128 for imagenet
            args.lr_scheduler = 'one_cycle'
            args.lr_three_phase = False
            args.lr_initial = 5e-3
            args.lr_max = 0.2
            args.lr_end = 5e-6
            args.weight_decay = 1e-5
        elif 'imagenet' in args.dataset_name:
            args.max_epochs = 20  # 90 (20) epochs for long (short) training
            args.batch_size = 128  # 256 (128) when num_epochs = 90 (20)
            args.lr_scheduler = 'one_cycle'
            args.lr_base = 0.1
            # The below args are only relevant when using "one_cycle" lr_schedule
            args.lr_three_phase = False
            args.lr_initial = 0.05
            args.lr_max = 1.0
            args.lr_end = 5e-5
            args.weight_decay = 1e-5
        elif args.dataset_name == 'cifar10':
            args.max_epochs = 30
            args.batch_size = 64
            args.lr_scheduler = 'one_cycle'
            args.lr_three_phase = False
            args.lr_initial = 5e-3
            args.lr_max = 0.2
            args.lr_end = 5e-6
            args.weight_decay = 1e-5

    elif args.task == 'retrieval':
        if 'imagenet100' in args.dataset_name or 'imagenet200' in args.dataset_name:
            args.max_epochs = 50
            args.batch_size = 64  # use 128 for imagenet
            args.lr_scheduler = 'one_cycle'
            args.lr_three_phase = True
            args.lr_initial = 5e-3
            args.lr_max = 0.1
            args.lr_end = 5e-6
            args.weight_decay = 1e-4
        elif 'imagenet' in args.dataset_name:
            args.max_epochs = 50  # 90 (20) epochs for long (short) training
            args.batch_size = 128  # 256 (128) when num_epochs = 90 (20)
            args.lr_scheduler = 'one_cycle'
            args.lr_three_phase = True
            args.lr_initial = 5e-3
            args.lr_max = 0.1
            args.lr_end = 5e-6
            args.weight_decay = 1e-4
        elif args.dataset_name == 'cifar10':
            args.max_epochs = 50
            args.batch_size = 128
            args.lr_scheduler = 'one_cycle'
            args.lr_three_phase = True
            args.lr_initial = 5e-3
            args.lr_max = 0.1
            args.lr_end = 5e-6
            args.weight_decay = 1e-4

    if args.model == 'AlexNet':
        args.lr_scheduler = 'step_lr'
        args.lr_base = 1e-2
        args.lr_step_size = int(args.max_epochs / 3)
        args.lr_gamma = 0.1

    return args


def train_on_clean_images(args, ray_tune=False):
    pl.seed_everything(1234)
    # ------------
    # data
    # ------------
    dataset = get_dataset(args.dataset_name, args.dataset_dir, args.augmentation, args.img_size, args.use_grayscale,
                          model=args.model)
    train_loader = dataset.get_train_dataloader(args.batch_size, args.num_workers, shuffle=True)
    val_loader = dataset.get_validation_dataloader(args.batch_size, args.num_workers, shuffle=False)
    args.num_classes = dataset.get_num_classes()
    args.steps_per_epoch = len(train_loader)

    if args.training_type == 'student':
        data_loader = dataset.get_train_dataloader(args.batch_size, args.num_workers, shuffle=False)
        trainer = pl.Trainer(max_epochs=args.max_epochs, strategy=DDPStrategy(find_unused_parameters=False), )

        args_use_push_pull = args.use_push_pull
        args.use_push_pull = False
        model = get_classifier(args)
        args.use_push_pull = args_use_push_pull

        teacher_train_predictions = trainer.predict(model=model, ckpt_path=args.teacher_ckpt, dataloaders=data_loader)
        teacher_val_predictions = trainer.predict(model=model, ckpt_path=args.teacher_ckpt, dataloaders=val_loader)
        train_loader.dataset.update_soft_targets(torch.concat([x['predictions'] for x in teacher_train_predictions]))
        val_loader.dataset.update_soft_targets(torch.concat([x['predictions'] for x in teacher_val_predictions]))

    # ------------
    # model
    # ------------
    logger = TensorBoardLogger(save_dir=args.logs_dir, name=args.experiment_name, default_hp_metric=False,
                               version=args.logs_version)
    args.logs_dir = logger.log_dir
    model = get_classifier(args)

    # ------------
    # training
    # ------------
    # ckpt_callback0 = ModelCheckpoint(filename='{epoch}-last', save_last=True)
    if args.task == 'classification':
        ckpt_callback1 = ModelCheckpoint(filename='{epoch:02d}-{val_top1_acc:.2f}-{val_loss:.2f}', every_n_epochs=1,
                                         save_top_k=-1, save_last=True)
        # ckpt_callback2 = ModelCheckpoint(mode='max', monitor='val_top1_acc', filename='{epoch}-{val_top1_acc:.2f}')
        # ckpt_callback3 = ModelCheckpoint(mode='max', monitor='val_top5_acc', filename='{epoch}-{val_top5_acc:.2f}')
        # tune_callback = TuneReportCallback(metrics={"top1_acc_val": "top1_acc_val",
        #                                             "top5_acc_val": "top5_acc_val"}, on="validation_end")
        # callbacks.extend([ckpt_callback2, ])
    elif args.task == 'retrieval':
        ckpt_callback1 = ModelCheckpoint(filename='{epoch:02d}-{top50_mAP_val:.2f}-{top200_mAP_val:.2f}',
                                         every_n_epochs=1, save_top_k=-1, save_last=True)
        # ckpt_callback2 = ModelCheckpoint(mode='max', monitor='top50_mAP_val', filename='{epoch}-{top50_mAP_val:.2f}')
        # ckpt_callback3 = ModelCheckpoint(mode='max', monitor='top200_mAP_val', filename='{epoch}-{top200_mAP_val:.2f}')
        # tune_callback = TuneReportCallback(metrics={"top200_mAP_val": "top200_mAP_val", }, on="validation_end")
    else:
        raise ValueError('Invalid task!')
    # ckpt_callback2 = ModelCheckpoint(save_last=True)
    callbacks = [ckpt_callback1, ]
    lr_monitor_callback = LearningRateMonitor(logging_interval='epoch')
    progress_bar_callback = TQDMProgressBar(refresh_rate=200)

    callbacks.extend([lr_monitor_callback, progress_bar_callback])
    # callbacks = callbacks + [tune_callback] if ray_tune else callbacks

    trainer = pl.Trainer(accelerator=args.accelerator, max_epochs=args.max_epochs, logger=logger,
                         callbacks=callbacks, devices=args.devices)  # ,
    # strategy=DDPStrategy(find_unused_parameters=False), fast_dev_run=False)

    if args.task == 'classification':
        trainer.fit(model, train_loader, ckpt_path=args.ckpt, val_dataloaders=[val_loader])
    elif args.task == 'retrieval':
        # retrieval_database = dataset.get_train_dataloader(args.batch_size, args.num_workers, shuffle=False)
        # [val_loader, retrieval_database]
        trainer.fit(model, train_loader, ckpt_path=args.ckpt, val_dataloaders=[val_loader, ])


def run_flow():
    args = parse_args()
    train_on_clean_images(args)


if __name__ == '__main__':

    # import os
    # import time
    #
    # def pid_exists(pid):
    #     try:
    #         os.kill(pid, 0)
    #     except OSError:
    #         return False
    #     return True
    #
    # def wait_pid(pid):
    #     while pid_exists(pid):
    #         time.sleep(60)
    #
    # print('Waiting for PID - 363099')
    # wait_pid(363099)
    run_flow()
