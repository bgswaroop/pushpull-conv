import copy
from pathlib import Path

from ray import tune
from ray.tune import SyncConfig, CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search import BasicVariantGenerator

from train_flow import parse_args, train_on_clean_images


def run_flow():
    config = {
        "use_pp": tune.grid_search([True]),
        "inh": tune.grid_search([4]),
        "avg": tune.grid_search([3]),
        "alpha": tune.grid_search([0.5, 0.6, 0.7, 0.8, 0.9]),
        "temp": tune.grid_search([4., 5., 6.]),
    }

    scheduler = ASHAScheduler(
        # metric='top1000_mAP_val',
        # mode='max',
        max_t=30,  # max_epochs
        grace_period=15,  # run at least 2 epochs
    )

    reporter = CLIReporter(
        parameter_columns=["avg", "inh", "alpha", "temp"],
        metric_columns=["top1_acc_val", "top5_acc_val"]
    )

    resources_per_trial = {"cpu": 6, "gpu": 0.14}

    # search_alg = BasicVariantGenerator(points_to_evaluate=[
    #     # {"use_pp": False, "avg": None, "inh": None},
    #     # {"use_pp": True, "avg": 3, "inh": 1},
    #     # {"use_pp": True, "avg": 3, "inh": 2},
    #     # {"use_pp": True, "avg": 3, "inh": 3},
    #     # {"use_pp": True, "avg": 3, "inh": 4},
    #     # {"use_pp": True, "avg": 5, "inh": 1},
    #     # {"use_pp": True, "avg": 5, "inh": 2},
    #     # {"use_pp": True, "avg": 5, "inh": 3},
    #     # {"use_pp": True, "avg": 5, "inh": 4},
    #     {"alpha": [0.5, 0.6, 0.7, 0.8, 0.9], "temp":[1., 2., 3., 4., 5., 6.]},
    # ])

    analysis = tune.run(
        hyperparameter_search,
        # search_alg=search_alg,
        resources_per_trial=resources_per_trial,
        metric='top1_acc_val',
        mode='max',
        config=config,
        num_samples=1,
        log_to_file=True,
        scheduler=scheduler,
        progress_reporter=reporter,
        name='hyperparameter_search_temp_alpha',
        local_dir='/home/guru/runtime_data/pushpull-conv/',
        sync_config=SyncConfig(),
        keep_checkpoints_num=1,
        checkpoint_score_attr="top1_acc_val",
    )
    print("Best hyperparameters found were: ", analysis.best_config)


def hyperparameter_search(config):
    args = copy.deepcopy(args_global)
    args.use_push_pull = config['use_pp']
    args.avg_kernel_size = config['avg']
    args.pull_inhibition_strength = config['inh']
    args.distillation_loss_alpha = config['alpha']
    args.distillation_loss_temp = config['temp']
    args.max_epochs = 30
    args.use_push_pull = True

    args.logs_dir = "/home/guru/runtime_data/pushpull-conv"
    args.experiment_name = "hyperparameter_search_temp_alpha"
    Path(args.logs_dir).joinpath(args.experiment_name).mkdir(exist_ok=True, parents=True)
    args.training_type = "student"
    args.teacher_ckpt = "/home/guru/runtime_data/pushpull-conv/resnet18_imagenet100_classification/resnet18/checkpoints/last.ckpt"
    args.loss_type = 'distillation_loss'

    index = 0
    logs_versions = dict()
    for alpha in [0.5, 0.6, 0.7, 0.8, 0.9]:
        for temp in [4., 5., 6.]:
            logs_versions[(alpha, temp)] = index
            index += 1

    args.logs_version = logs_versions[(args.distillation_loss_alpha, args.distillation_loss_temp)]

    # args.ckpt = f'/data2/p288722/runtime_data/pushpull-conv/resnet18_cifar10_classification/' \
    #             f'version_{args.logs_version}/checkpoints/last.ckpt'
    # if not Path(args.ckpt).exists():
    #     args.ckpt = None
    #     print('Checkpoint not found - Training from scratch!')
    # else:
    #     print(f'Resuming training from ckpt - {args.ckpt}')

    train_on_clean_images(args, ray_tune=True)


if __name__ == '__main__':
    args_global = parse_args()
    run_flow()
