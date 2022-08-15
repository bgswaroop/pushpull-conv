import argparse
import json
from collections import defaultdict
from pathlib import Path

import pytorch_lightning as pl
import torch

from data import get_dataset
from models import get_classifier
from models.utils import compute_map_score


class AddGaussianNoise(object):
    def __init__(self, mean, std, seed=None):
        self.std = std
        self.mean = mean
        self.seed = seed

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def compute_model_robustness_metrics(mAP_scores, baseline_mAP_scores):
    mCE, relative_mCE = 100, 100

    assert mAP_scores.get('clean', False), 'results on clean images not present'
    assert baseline_mAP_scores.get('clean', False), 'results on clean images not present'

    categories = sorted(mAP_scores.keys())
    categories.remove('clean')

    # Compute the error
    error_classifier = 1 - torch.Tensor([mAP_scores[x] for x in categories])
    error_baseline = 1 - torch.Tensor([baseline_mAP_scores[x] for x in categories])

    # Corruption Error (CE) and it's mean
    CE = torch.sum(error_classifier, dim=1) / torch.sum(error_baseline, dim=1)
    mCE = torch.mean(CE)

    # Relative CE and it's mean
    relative_CE = torch.sum(error_classifier - mAP_scores['clean'], dim=1) / \
                  torch.sum(error_baseline - baseline_mAP_scores['clean'], dim=1)
    relative_mCE = torch.mean(relative_CE)

    # converting all tensors to floats
    CE = {x: float(CE[idx]) for idx, x in enumerate(categories)}
    mCE = float(mCE)
    relative_CE = {x: float(relative_CE[idx]) for idx, x in enumerate(categories)}
    relative_mCE = float(relative_mCE)

    return CE, mCE, relative_CE, relative_mCE


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', default=224, type=int, choices=[32, 224])
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--dataset_dir', default=r'/data/p288722/datasets/cifar', type=str)
    parser.add_argument('--dataset_name', default='cifar10', choices=['cifar10', 'imagenet', 'imagenet200'])
    parser.add_argument('--corrupted_dataset_dir', default=r'/data/p288722/datasets/cifar/CIFAR-10-C-224x224', type=str)
    parser.add_argument('--corrupted_dataset_name', default='CIFAR-10-C-224x224',
                        choices=['CIFAR-10-C-EnhancedSeverity', 'CIFAR-10-C-224x224', 'CIFAR-10-C',
                                 'imagenet-c', 'imagenet200-c'])

    parser.add_argument('--num_workers', default=2, type=int,
                        help='how many subprocesses to use for data loading. ``0`` means that the data will be '
                             'loaded in the main process. (default: ``2``)')
    parser.add_argument('--model_ckpt', type=str, required=True)
    parser.add_argument('--corruption_types', nargs='*', default=None, type=str,
                        choices=['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
                                 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast',
                                 'elastic_transform', 'pixelate', 'jpeg_compression',
                                 'speckle_noise', 'gaussian_blur', 'spatter', 'saturate'])
    parser.add_argument('--use_push_pull', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--num_push_pull_layers', type=int, default=1)
    parser.add_argument('--baseline_classifier_results_dir', required=True, type=str)
    parser.add_argument('--model', default='AlexNet', type=str, required=True)
    parser.add_argument('--top_k', default=None, type=int)

    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.004)  # regularization

    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    if not args.accelerator:
        args.accelerator = 'gpu'
    if args.accelerator == 'gpu':
        args.device = torch.device(f'cuda:{torch.cuda.current_device()}')
    else:
        args.device = torch.device(f'cpu')

    if args.corrupted_dataset_name in {'CIFAR-10-C-EnhancedSeverity'}:
        args.num_severities = 10
    elif args.corrupted_dataset_name in {'CIFAR-10-C-224x224', 'CIFAR-10-C', 'imagenet-c', 'imagenet200-c'}:
        args.num_severities = 5
    else:
        raise NotImplementedError('num_severities for the current corrupted_dataset_name is undefined')

    assert Path(args.dataset_dir).exists(), f'{args.dataset_dir} does not exists!'
    assert Path(args.model_ckpt).exists(), f'{args.model_ckpt} path does not exists!'
    assert Path(args.baseline_classifier_results_dir).parent.exists(), \
        f'{Path(args.baseline_classifier_results_dir).parent} path does not exists!'
    Path(args.baseline_classifier_results_dir).mkdir(exist_ok=True, parents=False)

    return args


def predict_with_noise():
    pl.seed_everything(1234)
    args = parse_args()

    clean_dataset = get_dataset(args.dataset_name, args.dataset_dir, img_size=args.img_size)
    train_loader = clean_dataset.get_train_dataloader(args.batch_size, args.num_workers)

    model_ckpt = torch.load(args.model_ckpt)
    args.hash_length = model_ckpt['hyper_parameters']['hash_length']
    args.push_kernel_size = model_ckpt['hyper_parameters'].get('push_kernel_size', None)
    args.pull_kernel_size = model_ckpt['hyper_parameters'].get('pull_kernel_size', None)
    args.avg_kernel_size = model_ckpt['hyper_parameters'].get('avg_kernel_size', None)
    args.bias = model_ckpt['hyper_parameters'].get('bias', None)
    args.pull_inhibition_strength = model_ckpt['hyper_parameters'].get('pull_inhibition_strength', None)
    args.scale_the_outputs = model_ckpt['hyper_parameters'].get('scale_the_outputs', None)

    model = get_classifier(args.model)(args)
    model.load_state_dict(model_ckpt['state_dict'])

    trainer = pl.Trainer.from_argparse_args(args)
    train_predictions = trainer.predict(model=model, dataloaders=train_loader)
    train_hash_codes = torch.concat([x['hash_codes'] for x in train_predictions])
    train_ground_truths = torch.concat([x['ground_truths'] for x in train_predictions])

    results_dir = Path(args.model_ckpt).parent.parent.joinpath('results')
    results_dir.mkdir(exist_ok=True, parents=True)
    mAP_results_file = results_dir.joinpath(f'mAP_scores.json')

    mAP_scores = defaultdict(list)
    if mAP_results_file.exists():
        with open(mAP_results_file) as f:
            mAP_scores.update(json.load(f)[args.corrupted_dataset_name])

    if 'clean' not in mAP_scores:
        test_loader = clean_dataset.get_test_dataloader(args.batch_size, args.num_workers)
        predictions = trainer.predict(model=model, dataloaders=test_loader)
        test_hash_codes = torch.concat([x['hash_codes'] for x in predictions])
        test_ground_truths = torch.concat([x['ground_truths'] for x in predictions])
        map_score = compute_map_score(train_hash_codes, train_ground_truths, test_hash_codes, test_ground_truths,
                                      args.top_k, args.device)
        mAP_scores['clean'] = float(map_score)

    dataset = get_dataset(args.corrupted_dataset_name, args.corrupted_dataset_dir, args.num_severities)
    corruption_types = args.corruption_types if args.corruption_types else dataset.test_corruption_types

    for corruption_type in corruption_types:
        dataset.corruption_type = corruption_type
        mAP_scores[corruption_type] = []  # reset the list
        for severity_level in range(1, args.num_severities + 1):
            dataset.severity_level = severity_level

            test_loader = dataset.get_test_dataloader(args.batch_size, num_workers=args.num_workers)
            predictions = trainer.predict(model=model, dataloaders=test_loader)
            test_hash_codes = torch.concat([x['hash_codes'] for x in predictions])
            test_ground_truths = torch.concat([x['ground_truths'] for x in predictions])
            map_score = compute_map_score(train_hash_codes, train_ground_truths, test_hash_codes, test_ground_truths,
                                          args.top_k, args.device)
            mAP_scores[corruption_type].append(float(map_score))

    # Update the mAP scores to the JSON file - mAP_scores.json
    print(mAP_scores)
    if mAP_results_file.exists():
        with open(mAP_results_file, 'r') as f:
            mAP_results = json.load(f)
            mAP_results[args.corrupted_dataset_name] = mAP_scores
    else:
        mAP_results = {args.corrupted_dataset_name: mAP_scores}
    with open(mAP_results_file, 'w+') as f:
        json.dump(mAP_results, f, indent=2)

    # Compute mCE and relative-mCE scores from mAP w.r.t a baseline
    baseline_mAP_results_file = Path(args.baseline_classifier_results_dir).joinpath(f'mAP_scores.json')
    with open(baseline_mAP_results_file) as f:
        baseline_mAP_scores = json.load(f)[args.corrupted_dataset_name]
    CE, mCE, relative_CE, relative_mCE = compute_model_robustness_metrics(mAP_scores, baseline_mAP_scores)
    result_summary = {
        'CE': CE,
        'mCE': mCE,
        'relative_CE': relative_CE,
        'relative_mCE': relative_mCE,
        'mAP': mAP_scores,
    }

    # Update the results to the JSON file - all_scores.json
    all_results_file = results_dir.joinpath(f'all_scores.json')
    if all_results_file.exists():
        with open(all_results_file, 'r') as f:
            all_results = json.load(f)
            if args.corrupted_dataset_name in all_results:
                all_results[args.corrupted_dataset_name][str(baseline_mAP_results_file)] = result_summary
            else:
                all_results[args.corrupted_dataset_name] = {str(baseline_mAP_results_file): result_summary}
    else:
        all_results = {args.corrupted_dataset_name: {str(baseline_mAP_results_file): result_summary}}
    with open(all_results_file, 'w+') as f:
        json.dump(all_results, f, indent=2)

    print('Job finished!')


if __name__ == '__main__':
    predict_with_noise()
