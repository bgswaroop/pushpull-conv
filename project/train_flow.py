import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from tqdm import tqdm

from data import get_dataset
from models import get_classifier


# from torchmetrics.functional import accuracy

class LitProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = tqdm(disable=True, )
        return bar


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', default=224, type=int, choices=[32, 224])
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--dataset_dir', default='/data/p288722/datasets/cifar', type=str)
    parser.add_argument('--dataset_name', default='cifar10',
                        help="'cifar10', 'imagenet100', 'imagenet200', 'imagenet'"
                             "or add a suffix '_20pc' for a 20 percent stratified training subset."
                             "'_20pc' is an example, can be any float [1.0, 99.0]")
    parser.add_argument('--finetune', action=argparse.BooleanOptionalAction, default=False)  # todo: deprecate
    parser.add_argument('--finetune_ckpt', type=str, default=None)  # todo: deprecate
    parser.add_argument('--ckpt', type=str, default=None)  # ckpt to continue training

    # Pytorch lightning args
    parser.add_argument('--logs_dir', required=True, type=str, help='Path to save the logs/metrics during training')
    parser.add_argument('--experiment_name', default='dsh_push_pull', type=str)
    parser.add_argument('--num_workers', default=2, type=int,
                        help='how many subprocesses to use for data loading. ``0`` means that the data will be '
                             'loaded in the main process. (default: ``2``)')
    parser.add_argument('--logs_version', default=None, type=int)

    # Push Pull Convolutional Unit Params
    parser.add_argument('--use_push_pull', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--num_push_pull_layers', type=int, default=1)
    parser.add_argument('--push_kernel_size', type=int, default=None, help='Size of the push filter (int)')
    parser.add_argument('--pull_kernel_size', type=int, default=None, help='Size of the pull filter (int)')
    parser.add_argument('--avg_kernel_size', type=int, default=None, help='Size of the avg filter (int)')
    parser.add_argument('--pull_inhibition_strength', type=float, default=1.0)
    parser.add_argument('--model', default='AlexNet', type=str, required=True)
    parser.add_argument('--task', default='classification', type=str, required=True,
                        choices=['classification', 'retrieval'])

    parser.add_argument('--lr_base', type=float, default=0.1)
    # parser.add_argument('--lr_start', type=float, default=1e-3)  # for three-phase lr schedule
    # parser.add_argument('--lr_peak', type=float, default=1e-1)  # for three-phase lr schedule
    # parser.add_argument('--lr_end', type=float, default=1e-4)  # for three-phase lr schedule

    parser.add_argument('--lr_multiplier', type=float, default=1e-2)  # todo: deprecate if not necessary for CSQ Hash
    parser.add_argument('--weight_decay', type=float, default=1e-4)  # regularization
    parser.add_argument('--hash_length', type=int, default=48)
    parser.add_argument('--quantization_weight', type=float, default=0.01)

    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # # Validate args
    # if not args.accelerator:
    #     args.accelerator = 'gpu'
    # if args.accelerator == 'gpu':
    #     args.device = torch.device(f'cuda:{torch.cuda.current_device()}')
    # else:
    #     args.device = torch.device(f'cpu')
    # if not args.max_epochs:
    #     args.max_epochs = 60

    assert Path(args.dataset_dir).exists(), 'dataset_dir does not exists!'
    # assert Path(args.logs_dir).exists(), 'logs_dir does not exists!'
    Path(args.logs_dir).joinpath(args.experiment_name).mkdir(exist_ok=True, parents=True)

    if args.finetune:
        assert Path(args.finetune_ckpt).exists(), 'finetune_ckpt path does not exists!'

    torch.use_deterministic_algorithms(True, warn_only=True)
    # if args.dataset_name in {'imagenet100', 'imagenet200'}:
    #     args.max_epochs = 50
    #     args.batch_size = 64
    #     args.lr_base = 5e-2
    #     args.weight_decay = 5e-5
    if 'imagenet100' in args.dataset_name or 'imagenet200' in args.dataset_name:
        args.max_epochs = 30
        args.batch_size = 64
        args.lr_base = 0.2
        args.lr_scheduler = 'step_lr'
        args.lr_step_size = 10  # epochs
        args.lr_gamma = 0.1
        args.weight_decay = 1e-4
    elif 'imagenet' in args.dataset_name:
        args.max_epochs = 90
        args.batch_size = 256
        args.lr_base = 0.1
        args.lr_scheduler = 'step_lr'
        args.lr_step_size = 30  # epochs
        args.lr_gamma = 0.1
        args.weight_decay = 1e-4
    elif args.dataset_name == 'cifar10':
        # The following config is not stable for ResNet50 + cifar10 + retrieval
        args.max_epochs = 50
        args.batch_size = 256
        args.lr_base = 5e-2
        args.weight_decay = 5e-4

    if args.use_push_pull:
        assert args.push_kernel_size is not None, "Invalid config: use_push_pull=True but push_kernel_size is not set!"
        assert args.pull_kernel_size is not None, "Invalid config: use_push_pull=True but pull_kernel_size is not set!"
        assert args.avg_kernel_size is not None, "Invalid config: use_push_pull=True but avg_kernel_size is not set!"

    return args


def train_on_clean_images(args, ray_tune=False):
    pl.seed_everything(1234)
    # ------------
    # data
    # ------------
    dataset = get_dataset(args.dataset_name, args.dataset_dir, img_size=args.img_size)
    train_loader = dataset.get_train_dataloader(args.batch_size, args.num_workers, shuffle=True)
    val_loader = dataset.get_validation_dataloader(args.batch_size, args.num_workers, shuffle=False)
    test_loader = dataset.get_test_dataloader(args.batch_size, args.num_workers, shuffle=False)
    args.num_classes = dataset.get_num_classes()
    args.steps_per_epoch = len(train_loader)

    # ------------
    # model
    # ------------
    logger = TensorBoardLogger(save_dir=args.logs_dir, name=args.experiment_name, default_hp_metric=False,
                               version=args.logs_version)
    args.logs_dir = logger.log_dir
    model = get_classifier(args)
    if args.finetune:
        raise NotImplementedError()

    # ------------
    # training
    # ------------
    ckpt_callback1 = ModelCheckpoint(mode='min', monitor='loss_val', filename='{epoch}-{loss_val:.2f}', save_last=True)
    if args.task == 'classification':
        ckpt_callback2 = ModelCheckpoint(mode='max', monitor='top1_acc_val', filename='{epoch}-{top1_acc_val:.2f}')
        ckpt_callback3 = ModelCheckpoint(mode='max', monitor='top5_acc_val', filename='{epoch}-{top5_acc_val:.2f}')
        tune_callback = TuneReportCallback(metrics={"top1_acc_val": "top1_acc_val",
                                                    "top5_acc_val": "top5_acc_val"}, on="validation_end")
        # early_stopping_cb = EarlyStopping(monitor='top1_acc_val', mode='max', patience=7, min_delta=0.002)
    elif args.task == 'retrieval':
        ckpt_callback2 = ModelCheckpoint(mode='max', monitor='top50_mAP_val', filename='{epoch}-{top50_mAP_val:.2f}')
        ckpt_callback3 = ModelCheckpoint(mode='max', monitor='top200_mAP_val', filename='{epoch}-{top200_mAP_val:.2f}')
        tune_callback = TuneReportCallback(metrics={"top1000_mAP_val": "top1000_mAP_val", }, on="validation_end")
        # early_stopping_cb = EarlyStopping(monitor='top1_acc_val', mode='max', patience=7, min_delta=0.002)
    else:
        raise ValueError('Invalid task!')
    lr_monitor_callback = LearningRateMonitor(logging_interval='epoch')
    progress_bar_callback = LitProgressBar(refresh_rate=200)

    callbacks = [ckpt_callback1, ckpt_callback2, ckpt_callback3, lr_monitor_callback, progress_bar_callback]
    callbacks = callbacks + [tune_callback] if ray_tune else callbacks

    trainer = pl.Trainer.from_argparse_args(args, logger=logger, callbacks=callbacks,
                                            # gpus=[0],
                                            # auto_select_gpus=True,
                                            strategy=DDPStrategy(find_unused_parameters=False),
                                            )

    if args.task == 'classification':
        trainer.fit(model, train_loader, ckpt_path=args.ckpt, val_dataloaders=[val_loader])
        # trainer.test(model, test_loader)
    elif args.task == 'retrieval':
        retrieval_database = dataset.get_train_dataloader(args.batch_size, args.num_workers, shuffle=False)
        trainer.fit(model, train_loader, ckpt_path=args.ckpt, val_dataloaders=[retrieval_database, val_loader])
        # trainer.test(model, dataloaders=[retrieval_database, test_loader])
    # trainer.test(model, test_loader, ckpt_path=args.ckpt)   #fixme: add logic for test loader
    # # ------------
    # # testing
    # # ------------
    # # Optionally add ckpt_path to trainer.predict()
    # train_predictions = trainer.predict(model=model, dataloaders=train_loader)
    # test_predictions = trainer.predict(model=model, dataloaders=test_loader)
    #
    # y_hat = torch.concat([x['predictions'] for x in train_predictions])
    # y = torch.concat([x['ground_truths'] for x in train_predictions])
    # acc1 = accuracy(y_hat, y)
    # print(f'Training  set accuracy: {acc1}')
    #
    # y_hat = torch.concat([x['predictions'] for x in test_predictions])
    # y = torch.concat([x['ground_truths'] for x in test_predictions])
    # acc1 = accuracy(y_hat, y)
    # print(f'Test  set accuracy: {acc1}')


def run_flow():
    args = parse_args()
    train_on_clean_images(args)


if __name__ == '__main__':
    run_flow()
