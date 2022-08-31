import copy
import random
import time

from ray import tune
from ray.tune import SyncConfig, CLIReporter
from ray.tune.schedulers import ASHAScheduler

from train_flow import parse_args, train_on_clean_images


def run_flow():
    config = {
        "learning_rate": tune.grid_search([1e-5, 3e-4, 1e-4, 3e-3, 1e-3, 3e-2, 1e-2, 3e-1, 1e-1]),
        "batch_size": tune.grid_search([64]),
        "weight_decay": tune.grid_search([1e-4]),
        "quantization_weight": tune.grid_search([1e-3]),
    }

    scheduler = ASHAScheduler(
        # metric='top1000_mAP_val',
        # mode='max',
        max_t=10,  # max_epochs
        grace_period=3, # run at least 2 epochs

    )

    reporter = CLIReporter(
        parameter_columns=["learning_rate", "batch_size", "weight_decay", "quantization_weight"],
        metric_columns=["top1000_mAP_val"]
    )

    resources_per_trial = {"cpu": 10, "gpu": 0.32}

    analysis = tune.run(
        hyperparameter_search,
        search_alg=None,
        resources_per_trial=resources_per_trial,
        metric='top1000_mAP_val',
        mode='max',
        config=config,
        num_samples=1,
        log_to_file=True,
        scheduler=scheduler,
        progress_reporter=reporter,
        name='tune_dsh_resnet50_imagenet100',
        local_dir='/data2/p288722/runtime_data/pushpull-conv/resnet50_imagenet100_retrieval/ray_results',
        sync_config=SyncConfig(),
        keep_checkpoints_num=2,
        checkpoint_score_attr="top1000_mAP_val",
    )

    print("Best hyperparameters found were: ", analysis.best_config)


def hyperparameter_search(config):
    args_copy = copy.deepcopy(args)
    args_copy.learning_rate = config['learning_rate']
    args_copy.batch_size = config['batch_size']
    args_copy.weight_decay = config['weight_decay']
    args_copy.quantization_weight = config['quantization_weight']
    args_copy.max_epochs = 10

    time.sleep(random.randint(2, 60))

    train_on_clean_images(args_copy)


if __name__ == '__main__':
    args = parse_args()
    run_flow()
