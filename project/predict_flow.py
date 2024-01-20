import argparse
import json
from collections import defaultdict
from pathlib import Path

import lightning.pytorch as pl
import pandas as pd
import torch
from torchmetrics import Accuracy, ClasswiseWrapper

from data import get_dataset
from models import get_classifier
from models.utils import compute_map_score, accuracy


def compute_mean_corruption_error(scores):
    categories = sorted(scores.keys())
    if 'clean' in scores:
        categories.remove('clean')

    topk_CE, topk_mCE = dict(), dict()
    for top_k in scores['clean']:
        # Compute the error
        error_classifier = 1 - torch.Tensor([scores[x][top_k] for x in categories])

        # Corruption Error (CE) and it's mean
        CE = torch.mean(error_classifier, dim=1)
        mCE = torch.mean(CE)

        # converting all tensors to floats
        topk_CE[top_k] = {x: float(CE[idx]) for idx, x in enumerate(categories)}
        topk_mCE[top_k] = float(mCE)

    return topk_CE, topk_mCE


def compute_model_robustness_metrics(scores, baseline_scores):
    assert scores.get('clean', False), 'results on clean images not present'
    assert baseline_scores.get('clean', False), 'results on clean images not present'

    categories = sorted(scores.keys())
    categories.remove('clean')

    topk_CE, topk_mCE, topk_rCE, topk_rmCE = dict(), dict(), dict(), dict()
    for top_k in scores['clean']:
        # Compute the error
        error_classifier = 1 - torch.Tensor([scores[x][top_k] for x in categories])
        error_baseline = 1 - torch.Tensor([baseline_scores[x][top_k] for x in categories])

        # Corruption Error (CE) and it's mean
        CE = torch.sum(error_classifier, dim=1) / torch.sum(error_baseline, dim=1)
        mCE = torch.mean(CE)

        # Relative CE and it's mean
        relative_CE = torch.sum(error_classifier - scores['clean'][top_k], dim=1) / \
                      torch.sum(error_baseline - baseline_scores['clean'][top_k], dim=1)
        relative_mCE = torch.mean(relative_CE)

        # converting all tensors to floats
        topk_CE[top_k] = {x: float(CE[idx]) for idx, x in enumerate(categories)}
        topk_mCE[top_k] = float(mCE)
        topk_rCE[top_k] = {x: float(relative_CE[idx]) for idx, x in enumerate(categories)}
        topk_rmCE[top_k] = float(relative_mCE)

    return topk_CE, topk_mCE, topk_rCE, topk_rmCE


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', default=224, type=int, choices=[32, 224])
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--dataset_dir', default=r'/data/p288722/datasets/cifar', type=str)
    parser.add_argument('--dataset_name', default='imagenet',
                        help="'cifar10', 'imagenet100', 'imagenet200', 'imagenet'"
                             "or add a suffix '_20pc' for a 20 percent stratified training subset."
                             "'_20pc' is an example, can be any float [1.0, 99.0]")

    parser.add_argument('--corrupted_dataset_dir', default=r'/data/p288722/datasets/cifar/CIFAR-10-C-224x224', type=str)
    parser.add_argument('--corrupted_dataset_name', default='CIFAR-10-C-224x224',
                        choices=['CIFAR-10-C-EnhancedSeverity', 'CIFAR-10-C-224x224', 'cifar10-c',
                                 'imagenet-c', 'imagenet100-c', 'imagenet200-c'])

    parser.add_argument('--num_workers', default=2, type=int,
                        help='how many subprocesses to use for data loading. ``0`` means that the data will be '
                             'loaded in the main process. (default: ``2``)')
    parser.add_argument('--predict_model_logs_dir', type=str, required=True)
    parser.add_argument('--baseline_model_logs_dir', type=str, default=None)
    parser.add_argument('--corruption_types', nargs='*', default=None, type=str,
                        choices=['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
                                 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast',
                                 'elastic_transform', 'pixelate', 'jpeg_compression',
                                 'speckle_noise', 'gaussian_blur', 'spatter', 'saturate'])
    parser.add_argument('--use_push_pull', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--num_push_pull_layers', type=int, default=1)
    parser.add_argument('--model', default='AlexNet', type=str, required=True)
    parser.add_argument('--task', default='classification', type=str, required=False,
                        choices=['classification', 'retrieval'])

    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    args.num_severities = 5

    assert Path(args.dataset_dir).exists(), f'{args.dataset_dir} does not exists!'
    assert Path(args.corrupted_dataset_dir).exists(), f'{args.corrupted_dataset_dir} does not exists!'

    args.model_ckpt = Path(args.predict_model_logs_dir).joinpath('checkpoints/last.ckpt')
    assert args.model_ckpt.exists(), f'{args.model_ckpt} does not exists!'
    args.results_file = Path(args.predict_model_logs_dir).joinpath('results/all_scores.json')
    args.results_file.parent.mkdir(exist_ok=True, parents=True)

    if args.baseline_model_logs_dir is None:
        args.baseline_model_logs_dir = args.predict_model_logs_dir
        args.baseline_results_file = Path(args.baseline_model_logs_dir).joinpath('results/all_scores.json')
        args.baseline_results_file.parent.mkdir(exist_ok=True, parents=True)
    else:
        args.baseline_results_file = Path(args.baseline_model_logs_dir).joinpath('results/all_scores.json')

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

    return args


def predict_with_noise():
    pl.seed_everything(1234)
    args = parse_args()

    args.logs_dir = args.predict_model_logs_dir
    model = get_classifier(args)
    state_dict = torch.load(args.model_ckpt)['state_dict']

    # Mapping the weights to the corresponding ones as per the ResNet names in this implementation
    state_dict = {k.removeprefix('module.'): v for k, v in state_dict.items()}
    state_dict = {k.replace('bn1.', 'bn.') if k.startswith('bn1.') else k: v for k, v in state_dict.items()}
    state_dict = {k.replace('fc.', 'classifier.') if k.startswith('fc.') else k: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)

    trainer = pl.Trainer.from_argparse_args(args)
    device = trainer.strategy.root_device  # torch.device(f'cuda:{trainer.device_ids[0]}')

    clean_dataset = get_dataset(args.dataset_name, args.dataset_dir, img_size=args.img_size)

    if args.task == 'retrieval':
        train_loader = clean_dataset.get_train_dataloader(args.batch_size, args.num_workers, shuffle=False)
        predictions = trainer.predict(model=model, dataloaders=train_loader)
        train_predictions = torch.concat([x['predictions'] for x in predictions])
        train_ground_truths = torch.concat([x['ground_truths'] for x in predictions])

    scores = defaultdict(dict)
    without_baseline_file = args.results_file.parent.joinpath('wout_baseline.json')
    if args.results_file.exists():
        with open(args.results_file) as f:
            scores.update(json.load(f)[args.corrupted_dataset_name]['scores'])
    elif without_baseline_file.exists():
        with open(without_baseline_file) as f:
            scores.update(json.load(f)[args.corrupted_dataset_name]['scores'])

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
        top1_classwise_accuracy = ClasswiseWrapper(Accuracy(top_k=1, num_classes=num_classes, average=None), labels)
        output = top1_classwise_accuracy(test_predictions, test_ground_truths)
        classwise_index = sorted(output.keys())
        classwise_scores['Clean'] = [float(output[x].item()) for x in classwise_index]
    elif args.task == 'retrieval':
        scores['clean'] = compute_map_score(train_predictions, train_ground_truths,
                                            test_predictions, test_ground_truths,
                                            device, return_as_float=True)

    dataset = get_dataset(args.corrupted_dataset_name, args.corrupted_dataset_dir, args.num_severities)
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

    CE, mCE = compute_mean_corruption_error(scores)
    corruption_errors = {'CE': CE, 'mCE': mCE}

    with open(without_baseline_file, 'w+') as f:
        results_summary = {args.corrupted_dataset_name: {
            'scores': scores,
            'errors': corruption_errors
        }}
        json.dump(results_summary, f, indent=2, sort_keys=True)

    # Compute mCE and relative-mCE scores from mAP w.r.t a baseline
    if args.baseline_results_file.exists():
        with open(args.baseline_results_file) as f:
            baseline_scores = json.load(f)[args.corrupted_dataset_name]['scores']
    else:  # when baseline is not provided
        baseline_scores = scores
    CE, mCE, relative_CE, relative_mCE = compute_model_robustness_metrics(scores, baseline_scores)
    errors_wrt_baseline = {
        'CE': CE,
        'mCE': mCE,
        'relative_CE': relative_CE,
        'relative_mCE': relative_mCE,
    }

    # Update the results to the JSON file - all_scores.json
    if args.results_file.exists():
        with open(args.results_file, 'r') as f:
            results_summary = json.load(f)
        if args.corrupted_dataset_name in results_summary:

            results_summary[args.corrupted_dataset_name]['scores'] = scores
            results_summary[args.corrupted_dataset_name]['errors'] = corruption_errors

            temp = results_summary[args.corrupted_dataset_name].get('errors_wrt_baseline', dict())
            temp[args.baseline_model_logs_dir] = errors_wrt_baseline
            results_summary[args.corrupted_dataset_name]['errors_wrt_baseline'] = temp

        else:
            results_summary[args.corrupted_dataset_name] = {
                'scores': scores,
                'errors_wrt_baseline': {args.baseline_model_logs_dir: errors_wrt_baseline},
                'errors': corruption_errors,
            }
    else:
        results_summary = {args.corrupted_dataset_name: {
            'scores': scores,
            'errors_wrt_baseline': {args.baseline_model_logs_dir: errors_wrt_baseline},
            'errors': corruption_errors
        }}

    with open(args.results_file, 'w+') as f:
        json.dump(results_summary, f, indent=2, sort_keys=True)

    print('Job finished!')


if __name__ == '__main__':
    predict_with_noise()
