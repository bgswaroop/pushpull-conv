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
        "push": tune.grid_search([7]),
        "pull": tune.grid_search([7]),
        "inh": tune.grid_search([1, 2, 3, 4]),
        "avg": tune.grid_search([3, 5]),
    }

    scheduler = ASHAScheduler(
        # metric='top1000_mAP_val',
        # mode='max',
        max_t=45,  # max_epochs
        grace_period=45,  # run at least 2 epochs

    )

    reporter = CLIReporter(
        parameter_columns=["use_pp", "push", "pull", "avg", "inh"],
        metric_columns=["top1_acc_val", "top5_acc_val"]
    )

    resources_per_trial = {"cpu": 6, "gpu": 0.19}

    search_alg = BasicVariantGenerator(points_to_evaluate=[
        # {"use_pp": False, "avg": None, "inh": None, "push": None, "pull": None},
        # {"use_pp": True, "avg": 3, "inh": 1},
        # {"use_pp": True, "avg": 3, "inh": 2},
        # {"use_pp": True, "avg": 3, "inh": 3},
        # {"use_pp": True, "avg": 3, "inh": 4},
        # {"use_pp": True, "avg": 5, "inh": 1},
        {"use_pp": True, "avg": 5, "inh": 2},
        # {"use_pp": True, "avg": 5, "inh": 3},
        # {"use_pp": True, "avg": 5, "inh": 4},
    ])

    analysis = tune.run(
        hyperparameter_search,
        search_alg=search_alg,
        resources_per_trial=resources_per_trial,
        metric='top1_acc_val',
        mode='max',
        config=config,
        num_samples=1,
        log_to_file=True,
        scheduler=scheduler,
        progress_reporter=reporter,
        name='tune_resnet18_cifar10_classification',
        local_dir='/data2/p288722/runtime_data/pushpull-conv/hyperparameter_opt/',
        sync_config=SyncConfig(),
        keep_checkpoints_num=1,
        checkpoint_score_attr="top1_acc_val",
    )
    print("Best hyperparameters found were: ", analysis.best_config)


def hyperparameter_search(config):
    args = copy.deepcopy(args_global)
    args.use_push_pull = config['use_pp']
    args.push_kernel_size = config['push']
    args.pull_kernel_size = config['pull']
    args.avg_kernel_size = config['avg']
    args.pull_inhibition_strength = config['inh']
    args.max_epochs = 45

    if not args.use_push_pull:
        args.logs_version = 0
    elif args.avg_kernel_size == 3 and args.pull_inhibition_strength == 1:
        args.logs_version = 1
    elif args.avg_kernel_size == 3 and args.pull_inhibition_strength == 2:
        args.logs_version = 2
    elif args.avg_kernel_size == 3 and args.pull_inhibition_strength == 3:
        args.logs_version = 3
    elif args.avg_kernel_size == 3 and args.pull_inhibition_strength == 4:
        args.logs_version = 4
    elif args.avg_kernel_size == 5 and args.pull_inhibition_strength == 1:
        args.logs_version = 5
    elif args.avg_kernel_size == 5 and args.pull_inhibition_strength == 2:
        args.logs_version = 6
    elif args.avg_kernel_size == 5 and args.pull_inhibition_strength == 3:
        args.logs_version = 7
    elif args.avg_kernel_size == 5 and args.pull_inhibition_strength == 4:
        args.logs_version = 8
    else:
        raise ValueError('Invalid Configuration!')

    args.ckpt = f'/data2/p288722/runtime_data/pushpull-conv/resnet18_cifar10_classification/' \
                f'version_{args.logs_version}/checkpoints/last.ckpt'
    if not Path(args.ckpt).exists():
        args.ckpt = None
        print('Checkpoint not found - Training from scratch!')
    else:
        print(f'Resuming training from ckpt - {args.ckpt}')

    train_on_clean_images(args, ray_tune=True)


if __name__ == '__main__':
    args_global = parse_args()
    run_flow()
