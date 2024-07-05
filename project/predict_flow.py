import argparse
import json
from pathlib import Path

import lightning.pytorch as pl
import pandas as pd
import torch
import yaml
from lightning.pytorch.callbacks import TQDMProgressBar
from matplotlib import pyplot as plt
from torchmetrics import Accuracy, ClasswiseWrapper

from data import get_dataset
from models import get_classifier
from models.utils import compute_map_score, accuracy


def compute_mean_corruption_error(scores):
    categories = sorted(scores.keys())
    if 'clean' in scores:
        categories.remove('clean')

    topk_ce, topk_mce = dict(), dict()
    for top_k in scores['clean']:
        # Compute the error
        error_classifier = 1 - torch.Tensor([scores[x][top_k] for x in categories])

        # Corruption Error (CE) and it's mean
        ce = torch.mean(error_classifier, dim=1)
        mce = torch.mean(ce)

        # converting all tensors to floats
        topk_ce[top_k] = {x: float(ce[idx]) for idx, x in enumerate(categories)}
        topk_mce[top_k] = float(mce)

    return topk_ce, topk_mce


def compute_model_robustness_metrics(scores, baseline_scores):
    assert scores.get('clean', False), 'results on clean images not present'
    assert baseline_scores.get('clean', False), 'results on clean images not present'

    categories = sorted(scores.keys())
    categories.remove('clean')

    topk_ce, topk_mce, topk_rce, topk_rmce = dict(), dict(), dict(), dict()
    for top_k in scores['clean']:
        # Compute the error
        error_classifier = 1 - torch.Tensor([scores[x][top_k] for x in categories])
        error_baseline = 1 - torch.Tensor([baseline_scores[x][top_k] for x in categories])

        # Corruption Error (CE) and it's mean
        ce = torch.sum(error_classifier, dim=1) / torch.sum(error_baseline, dim=1)
        mce = torch.mean(ce)

        # Relative CE and it's mean

        relative_ce = torch.sum(error_classifier - scores['clean'][top_k], dim=1) / torch.sum(
            error_baseline - baseline_scores['clean'][top_k], dim=1)
        relative_mce = torch.mean(relative_ce)

        # converting all tensors to floats
        topk_ce[top_k] = {x: float(ce[idx]) for idx, x in enumerate(categories)}
        topk_mce[top_k] = float(mce)
        topk_rce[top_k] = {x: float(relative_ce[idx]) for idx, x in enumerate(categories)}
        topk_rmce[top_k] = float(relative_mce)

    return topk_ce, topk_mce, topk_rce, topk_rmce


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', default=224, type=int, choices=[32, 224])
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--dataset_dir', default=None, type=str)
    parser.add_argument('--dataset_name', default='imagenet',
                        help="'cifar10', 'imagenet100', 'imagenet200', 'imagenet'"
                             "or add a suffix '_20pc' for a 20 percent stratified training subset."
                             "'_20pc' is an example, can be any float [1.0, 99.0]")

    parser.add_argument('--corrupted_dataset_dir', default=r'/data/p288722/datasets/cifar/CIFAR-10-C-224x224', type=str)
    parser.add_argument('--corrupted_dataset_name', default='CIFAR-10-C-224x224',
                        choices=['CIFAR-10-C-EnhancedSeverity', 'CIFAR-10-C-224x224', 'cifar10-c',
                                 'imagenet-c', 'imagenet100-c', 'imagenet200-c'])

    parser.add_argument('--accelerator', type=str, choices=['cpu', 'gpu', 'auto'])
    parser.add_argument('--devices', type=str, default='auto')
    parser.add_argument('--num_workers', default=2, type=int,
                        help='how many subprocesses to use for data loading. ``0`` means that the data will be '
                             'loaded in the main process. (default: ``2``)')
    parser.add_argument('--predict_model_logs_dir', type=str, required=True)
    parser.add_argument('--models_to_predict', default='last', choices=['all', 'last'])
    parser.add_argument('--model', default='resnet18', type=str)
    parser.add_argument('--baseline_model_logs_dir', type=str, default=None)
    parser.add_argument('--corruption_types', nargs='*', default=None, type=str,
                        choices=['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
                                 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast',
                                 'elastic_transform', 'pixelate', 'jpeg_compression',
                                 'speckle_noise', 'gaussian_blur', 'spatter', 'saturate'])
    parser.add_argument('--use_push_pull', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--num_push_pull_layers', type=int, default=1)
    parser.add_argument('--task', default='classification', type=str, required=False,
                        choices=['classification', 'retrieval'])
    args = parser.parse_args()
    args.num_severities = 5

    assert Path(args.dataset_dir).exists(), f'{args.dataset_dir} does not exists!'
    assert Path(args.corrupted_dataset_dir).exists(), f'{args.corrupted_dataset_dir} does not exists!'

    args.results_file = Path(args.predict_model_logs_dir).joinpath('results/results.yaml')
    args.results_file.parent.mkdir(exist_ok=True, parents=True)

    if args.baseline_model_logs_dir is None:
        args.baseline_model_logs_dir = args.predict_model_logs_dir
        args.baseline_results_file = Path(args.baseline_model_logs_dir).joinpath('results/results.yaml')
        args.baseline_results_file.parent.mkdir(exist_ok=True, parents=True)
    else:
        args.baseline_results_file = Path(args.baseline_model_logs_dir).joinpath('results/results.yaml')

    return args


def reconfigure_args(args):
    model_ckpt = torch.load(args.model_ckpt)
    if 'hyper_parameters' in model_ckpt:
        args.num_classes = model_ckpt['hyper_parameters'].get('num_classes', None)
        args.hash_length = model_ckpt['hyper_parameters'].get('hash_length', None)
        args.quantization_weight = model_ckpt['hyper_parameters'].get('quantization_weight', None)
        args.avg_kernel_size = model_ckpt['hyper_parameters'].get('avg_kernel_size', None)
        args.pull_inhibition_strength = model_ckpt['hyper_parameters'].get('pull_inhibition_strength', None)
        args.trainable_pull_inhibition = model_ckpt['hyper_parameters'].get('trainable_pull_inhibition', False)
        args.use_push_pull = model_ckpt['hyper_parameters'].get('use_push_pull', False)
    else:
        args.num_classes = 1000
        args.hash_length = 64
        args.quantization_weight = 1e-4
        args.avg_kernel_size = 3
        args.pull_inhibition_strength = 1.0
        args.trainable_pull_inhibition = True
        args.use_push_pull = False

    args.logs_dir = args.predict_model_logs_dir
    return args


def run_predict_flow(args):
    args = reconfigure_args(args)
    model = get_classifier(args)
    state_dict = torch.load(args.model_ckpt)['state_dict']
    epoch = torch.load(args.model_ckpt)['epoch']

    # Mapping the weights to the corresponding ones as per the ResNet names in this implementation
    state_dict = {k.removeprefix('module.'): v for k, v in state_dict.items()}
    # state_dict = {k.replace('bn1.', 'bn.') if k.startswith('bn1.') else k: v for k, v in state_dict.items()}
    state_dict = {k.replace('fc.', 'classifier.') if k.startswith('fc.') else k: v for k, v in state_dict.items()}

    progress_bar_callback = TQDMProgressBar(refresh_rate=25)
    callbacks=[progress_bar_callback,]

    model.load_state_dict(state_dict)
    trainer = pl.Trainer(accelerator=args.accelerator, fast_dev_run=False, callbacks=callbacks, devices=args.devices)
    device = trainer.strategy.root_device
    clean_dataset = get_dataset(args.dataset_name, args.dataset_dir, img_size=args.img_size,
                                grayscale=args.use_grayscale, model=args.model)

    if args.task == 'retrieval':
        train_loader = clean_dataset.get_train_dataloader(args.batch_size, args.num_workers, shuffle=False)
        predictions = trainer.predict(model=model, dataloaders=train_loader)
        train_predictions = torch.concat([x['predictions'] for x in predictions])
        train_ground_truths = torch.concat([x['ground_truths'] for x in predictions])

    scores = dict()
    # without_baseline_file = args.results_file.parent.joinpath('wout_baseline.yaml')
    if args.results_file.exists():
        with open(args.results_file) as f:
            epoch_wise_results = yaml.safe_load(f)[args.corrupted_dataset_name]['epoch_wise_results']
            if f'epoch_{epoch:03d}' in epoch_wise_results:
                scores.update(epoch_wise_results[f'epoch_{epoch:03d}']['scores'])
    # elif without_baseline_file.exists():
    #     with open(without_baseline_file) as f:
    #         epoch_wise_results = yaml.safe_load(f)[args.corrupted_dataset_name]['epoch_wise_results']
    #         if f'epoch_{epoch:03d}' in epoch_wise_results:
    #             scores.update(epoch_wise_results[f'epoch_{epoch:03d}']['scores'])

    # if 'clean' not in scores:
    test_loader = clean_dataset.get_test_dataloader(args.batch_size, args.num_workers)
    predictions = trainer.predict(model=model, dataloaders=test_loader)
    test_predictions = torch.concat([x['predictions'] for x in predictions])
    test_ground_truths = torch.concat([x['ground_truths'] for x in predictions])
    scores['clean'] = {}
    classwise_scores = {}

    num_classes = test_loader.dataset.num_classes
    labels = [test_loader.dataset.labels_num_to_txt[x] for x in range(num_classes)]
    if 'imagenet' in args.dataset_name:
        with open(Path(__file__).parent.resolve().joinpath('data/imagenet_c/LOC_synset_mapping.json')) as f:
            synset = json.load(f)
            classwise_scores['synset'] = [synset[x] for x in sorted(labels)]

    if args.task == 'classification':
        scores['clean']['top1'] = float(accuracy(test_predictions, test_ground_truths, top_k=1))
        scores['clean']['top5'] = float(accuracy(test_predictions, test_ground_truths, top_k=5))
        top1_classwise_accuracy = ClasswiseWrapper(
            Accuracy(task='multiclass', top_k=1, num_classes=num_classes, average=None), labels)
        output = top1_classwise_accuracy(test_predictions, test_ground_truths)
        classwise_index = sorted(output.keys())
        classwise_scores['Clean'] = [float(output[x].item()) for x in classwise_index]
    elif args.task == 'retrieval':
        scores['clean'] = compute_map_score(train_predictions, train_ground_truths,
                                            test_predictions, test_ground_truths,
                                            device, return_as_float=True)

    dataset = get_dataset(args.corrupted_dataset_name, args.corrupted_dataset_dir, args.num_severities,
                          grayscale=args.use_grayscale, model=args.model)
    corruption_types = args.corruption_types if args.corruption_types else dataset.test_corruption_types

    for corruption_type in corruption_types:

        if corruption_type in scores:
            continue

        dataset.corruption_type = corruption_type
        scores[corruption_type] = {x: [] for x in scores['clean']}  # reset the dictionary
        for severity_level in range(1, args.num_severities + 1):
            dataset.severity_level = severity_level

            test_loader = dataset.get_test_dataloader(args.batch_size, num_workers=args.num_workers)
            print(f'Predicting corruption - {corruption_type} with severity level {severity_level}')
            predictions = trainer.predict(model=model, dataloaders=test_loader)
            test_predictions = torch.concat([x['predictions'] for x in predictions])
            test_ground_truths = torch.concat([x['ground_truths'] for x in predictions])

            if args.task == 'classification':
                scores[corruption_type]['top1'].append(float(accuracy(test_predictions, test_ground_truths, top_k=1)))
                scores[corruption_type]['top5'].append(float(accuracy(test_predictions, test_ground_truths, top_k=5)))

                output = top1_classwise_accuracy(test_predictions, test_ground_truths)
                classwise_index = sorted(output.keys())
                classwise_scores[f'{corruption_type}_{severity_level}'] = \
                    [float(output[x].item()) for x in classwise_index]

            elif args.task == 'retrieval':
                map_score = compute_map_score(train_predictions, train_ground_truths,
                                              test_predictions, test_ground_truths,
                                              device, return_as_float=True)
                for item in scores['clean']:
                    scores[corruption_type][item].append(map_score[item])

    if args.task == 'classification':
        classwise_results = pd.DataFrame(classwise_scores, index=classwise_index)
        classwise_results = classwise_results.rename_axis(index='Classwise results')
        classwise_scores_file = args.results_file.parent.joinpath('classwise_scores.csv')
        classwise_results.to_csv(classwise_scores_file, sep=',')

    ce, mce = compute_mean_corruption_error(scores)
    corruption_errors = {'CE': ce, 'mCE': mce}
    # result = {
    #     'checkpoint': str(args.model_ckpt),
    #     'scores': scores,
    #     'errors': corruption_errors
    # }
    #
    # if without_baseline_file.exists():
    #     with open(without_baseline_file, 'r') as f:
    #         results_summary = yaml.safe_load(f)
    #         results_summary[args.corrupted_dataset_name]['epoch_wise_results'][f'epoch_{epoch:03d}'] = result
    # else:
    #     results_summary = {args.corrupted_dataset_name: {'epoch_wise_results': {f'epoch_{epoch:03d}': result}}}
    # with open(without_baseline_file, 'w+') as f:
    #     yaml.safe_dump(results_summary, f, default_flow_style=None, width=float("inf"))  # Verified

    # Compute mCE and relative-mCE scores from mAP w.r.t a baseline
    if args.baseline_results_file.exists():
        with open(args.baseline_results_file) as f:
            results_summary = yaml.safe_load(f)
        if f'epoch_{epoch:03d}' in results_summary[args.corrupted_dataset_name]['epoch_wise_results']:
            results = results_summary[args.corrupted_dataset_name]['epoch_wise_results'][f'epoch_{epoch:03d}']
            baseline_scores = results['scores']
        else:
            baseline_scores = scores
    else:  # when baseline is not provided
        baseline_scores = scores

    ce, mce, relative_ce, relative_mce = compute_model_robustness_metrics(scores, baseline_scores)
    errors_wrt_baseline = {
        'CE': ce,
        'mCE': mce,
        'relative_CE': relative_ce,
        'relative_mCE': relative_mce,
    }

    result = {
        'checkpoint': str(args.model_ckpt),
        'errors': corruption_errors,
        'errors_wrt_baseline': {args.baseline_model_logs_dir: errors_wrt_baseline},
        'scores': scores,
    }

    # Update to - results.yaml
    if args.results_file.exists():
        with open(args.results_file, 'r') as f:
            results_summary = yaml.safe_load(f)

        if args.corrupted_dataset_name in results_summary:
            epoch_wise_results = results_summary[args.corrupted_dataset_name]['epoch_wise_results']

            if f'epoch_{epoch:03d}' in epoch_wise_results:
                result = epoch_wise_results[f'epoch_{epoch:03d}']
                temp = result.get('errors_wrt_baseline', dict())
                temp[args.baseline_model_logs_dir] = errors_wrt_baseline
                result['errors_wrt_baseline'] = temp
                result['scores'] = scores
                result['errors'] = corruption_errors
                result['checkpoint'] = str(args.model_ckpt)
                results_summary[args.corrupted_dataset_name]['epoch_wise_results'][f'epoch_{epoch:03d}'] = result
            else:
                results_summary[args.corrupted_dataset_name]['epoch_wise_results'][f'epoch_{epoch:03d}'] = result
        else:
            results_summary[args.corrupted_dataset_name] = {'epoch_wise_results': {f'epoch_{epoch:03d}': result}}
    else:
        results_summary = {args.corrupted_dataset_name: {'epoch_wise_results': {f'epoch_{epoch:03d}': result}}}

    with open(args.results_file, 'w+') as f:
        yaml.safe_dump(results_summary, f, default_flow_style=None, width=float("inf"))  # Verified


def parse_args_and_run_predict_flow():
    pl.seed_everything(1234)
    args = parse_args()

    if args.models_to_predict == 'last':
        args.model_ckpt = Path(args.predict_model_logs_dir).joinpath('checkpoints/last.ckpt').resolve()
        args.model_ckpt = Path(args.predict_model_logs_dir).joinpath(f'checkpoints/{args.model_ckpt.name}')
        assert args.model_ckpt.exists(), f'{args.model_ckpt} does not exists!'
        run_predict_flow(args)

    elif args.models_to_predict == 'all':
        checkpoints = sorted(Path(args.predict_model_logs_dir).glob('checkpoints/epoch=*.ckpt'), reverse=True)
        for ckpt in checkpoints:
            args.model_ckpt = ckpt
            run_predict_flow(args)

        with open(args.results_file, 'r') as f:
            epoch_wise_results = yaml.safe_load(f)[args.corrupted_dataset_name]['epoch_wise_results']
        tradeoff = []
        for epoch, result in epoch_wise_results.items():
            acc = result['scores']['clean']['top1']
            rob = result['errors']['mCE']['top1']
            tradeoff.append(acc / rob)

        with open(args.baseline_results_file, 'r') as f:
            epoch_wise_results = yaml.safe_load(f)[args.corrupted_dataset_name]['epoch_wise_results']
        baseline_tradeoff = []
        for epoch, result in epoch_wise_results.items():
            acc = result['scores']['clean']['top1']
            rob = result['errors']['mCE']['top1']
            baseline_tradeoff.append(acc / rob)

        plt.figure(dpi=300)
        x = range(len(baseline_tradeoff))
        plt.plot(x, baseline_tradeoff, label=f'baseline {args.baseline_results_file.parent.parent.name}')
        plt.plot(x, tradeoff, label=f'{args.results_file.parent.parent.name}')
        plt.xlabel('Epoch')
        plt.ylabel('Ratio = Acc / Robustness')
        plt.title('Trade-off between Accuracy and Robustness')
        plt.legend()
        plt.savefig(f'{args.results_file.parent.joinpath("acc_vs_robustness.png")}')
        plt.show()

    print('Job finished!')


if __name__ == '__main__':
    parse_args_and_run_predict_flow()
