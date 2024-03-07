import yaml
from pathlib import Path

import pandas as pd


def save_scores(exp_names, scores_on_clean_dataset, errors, errors_wrt_baseline, save_dir):
    row_header = ['Error', 'mCE',
                  'Gauss', 'Shot', 'Impulse',  # Noise
                  'Defocus', 'Glass', 'Motion', 'Zoom',  # Blur
                  'Snow', 'Frost', 'Fog', 'Bright',  # Weather
                  'Contrast', 'Elastic', 'Pixel', 'JPEG'  # Digital
                  ]

    summary_results = dict()
    for topk in scores_on_clean_dataset[0].keys():
        df_CE = pd.DataFrame(index=exp_names, columns=row_header)
        for idx, r, s in zip(exp_names, errors_wrt_baseline, scores_on_clean_dataset):
            df_CE['Error'][idx] = 1 - s.get(topk, None)
            df_CE['mCE'][idx] = r['mCE'].get(topk, None)
            df_CE['Gauss'][idx] = r['CE'].get(topk, {}).get('gaussian_noise', None)
            df_CE['Shot'][idx] = r['CE'].get(topk, {}).get('shot_noise', None)
            df_CE['Impulse'][idx] = r['CE'].get(topk, {}).get('impulse_noise', None)
            df_CE['Defocus'][idx] = r['CE'].get(topk, {}).get('defocus_blur', None)
            df_CE['Glass'][idx] = r['CE'].get(topk, {}).get('glass_blur', None)
            df_CE['Motion'][idx] = r['CE'].get(topk, {}).get('motion_blur', None)
            df_CE['Zoom'][idx] = r['CE'].get(topk, {}).get('zoom_blur', None)
            df_CE['Snow'][idx] = r['CE'].get(topk, {}).get('snow', None)
            df_CE['Frost'][idx] = r['CE'].get(topk, {}).get('frost', None)
            df_CE['Fog'][idx] = r['CE'].get(topk, {}).get('fog', None)
            df_CE['Bright'][idx] = r['CE'].get(topk, {}).get('brightness', None)
            df_CE['Contrast'][idx] = r['CE'].get(topk, {}).get('contrast', None)
            df_CE['Elastic'][idx] = r['CE'].get(topk, {}).get('elastic_transform', None)
            df_CE['Pixel'][idx] = r['CE'].get(topk, {}).get('pixelate', None)
            df_CE['JPEG'][idx] = r['CE'].get(topk, {}).get('jpeg_compression', None)
        summary_results[f'CE_{topk}.csv'] = df_CE

    for topk in scores_on_clean_dataset[0].keys():
        df_CE = pd.DataFrame(index=exp_names, columns=row_header)
        for idx, r, s in zip(exp_names, errors, scores_on_clean_dataset):
            df_CE['Error'][idx] = 1 - s.get(topk, None)
            df_CE['mCE'][idx] = r['mCE'].get(topk, None)
            df_CE['Gauss'][idx] = r['CE'].get(topk, {}).get('gaussian_noise', None)
            df_CE['Shot'][idx] = r['CE'].get(topk, {}).get('shot_noise', None)
            df_CE['Impulse'][idx] = r['CE'].get(topk, {}).get('impulse_noise', None)
            df_CE['Defocus'][idx] = r['CE'].get(topk, {}).get('defocus_blur', None)
            df_CE['Glass'][idx] = r['CE'].get(topk, {}).get('glass_blur', None)
            df_CE['Motion'][idx] = r['CE'].get(topk, {}).get('motion_blur', None)
            df_CE['Zoom'][idx] = r['CE'].get(topk, {}).get('zoom_blur', None)
            df_CE['Snow'][idx] = r['CE'].get(topk, {}).get('snow', None)
            df_CE['Frost'][idx] = r['CE'].get(topk, {}).get('frost', None)
            df_CE['Fog'][idx] = r['CE'].get(topk, {}).get('fog', None)
            df_CE['Bright'][idx] = r['CE'].get(topk, {}).get('brightness', None)
            df_CE['Contrast'][idx] = r['CE'].get(topk, {}).get('contrast', None)
            df_CE['Elastic'][idx] = r['CE'].get(topk, {}).get('elastic_transform', None)
            df_CE['Pixel'][idx] = r['CE'].get(topk, {}).get('pixelate', None)
            df_CE['JPEG'][idx] = r['CE'].get(topk, {}).get('jpeg_compression', None)
        summary_results[f'absolute_CE_{topk}.csv'] = df_CE

    row_header[1] = 'Rel. mCE'
    for topk in scores_on_clean_dataset[0].keys():
        df_rCE = pd.DataFrame(index=exp_names, columns=row_header)
        for idx, r, s in zip(exp_names, errors_wrt_baseline, scores_on_clean_dataset):
            df_rCE['Error'][idx] = 1 - s.get(topk, None)
            df_rCE['Rel. mCE'][idx] = r['relative_mCE'].get(topk, None)
            df_rCE['Gauss'][idx] = r['relative_CE'].get(topk, {}).get('gaussian_noise', None)
            df_rCE['Shot'][idx] = r['relative_CE'].get(topk, {}).get('shot_noise', None)
            df_rCE['Impulse'][idx] = r['relative_CE'].get(topk, {}).get('impulse_noise', None)
            df_rCE['Defocus'][idx] = r['relative_CE'].get(topk, {}).get('defocus_blur', None)
            df_rCE['Glass'][idx] = r['relative_CE'].get(topk, {}).get('glass_blur', None)
            df_rCE['Motion'][idx] = r['relative_CE'].get(topk, {}).get('motion_blur', None)
            df_rCE['Zoom'][idx] = r['relative_CE'].get(topk, {}).get('zoom_blur', None)
            df_rCE['Snow'][idx] = r['relative_CE'].get(topk, {}).get('snow', None)
            df_rCE['Frost'][idx] = r['relative_CE'].get(topk, {}).get('frost', None)
            df_rCE['Fog'][idx] = r['relative_CE'].get(topk, {}).get('fog', None)
            df_rCE['Bright'][idx] = r['relative_CE'].get(topk, {}).get('brightness', None)
            df_rCE['Contrast'][idx] = r['relative_CE'].get(topk, {}).get('contrast', None)
            df_rCE['Elastic'][idx] = r['relative_CE'].get(topk, {}).get('elastic_transform', None)
            df_rCE['Pixel'][idx] = r['relative_CE'].get(topk, {}).get('pixelate', None)
            df_rCE['JPEG'][idx] = r['relative_CE'].get(topk, {}).get('jpeg_compression', None)
        summary_results[f'rCE_{topk}.csv'] = df_rCE

    for filename, df in summary_results.items():
        df.to_csv(save_dir.joinpath(filename))


def plot_scores(experiments, all_scores, dataset_name, compile_scores_dir):
    from matplotlib import pyplot as plt
    import numpy as np

    plt.figure(dpi=300)
    for exp_name, scores in zip(experiments, all_scores):
        severity_0 = 1 - scores.pop('clean')['top1']
        severity_1to5 = np.array([k['top1'] for k in scores.values()])
        severity_1to5 = list(1 - np.mean(severity_1to5, axis=0))
        all_severities = [severity_0] + severity_1to5
        plt.plot([0, 1, 2, 3, 4, 5], all_severities, label=exp_name)

    plt.title(f'Errors vs Corruption severity on {dataset_name}')
    plt.xlabel('Severity level')
    plt.ylabel('Errors')
    plt.legend()
    plt.tight_layout()
    plt.savefig(compile_scores_dir.joinpath('top1_scores_vs_severity.png'))


def run_flow():
    compile_scores_dir = Path(r'/scratch/p288722/runtime_data/pushpull-conv/resnet50_imagenet200_classification')
    dataset_name = 'imagenet200-c'
    baseline_exp_dir = '/scratch/p288722/runtime_data/pushpull-conv/resnet50_imagenet200_classification/resnet50'
    experiments = [
        'resnet50',
        # 'resnet50_sigma2'
        'resnet50_avg0',
        'resnet50_avg3',
        'resnet50_avg5',
        # 'resnet50_inht_avg0',
        # 'resnet50_inht_avg3',
        # 'resnet50_inht_avg5',
        
        # 'strisciuglio_resnet50_inh_trainable',
        # 'vasconcelos_resnet50_avg3',
        # 'zhang_resnet50',
        # 'resnet50_AutoAug',
        # 'resnet50_AutoAug_avg3',
        # 'resnet50_avg5',
        # 'resnet50_augmix',
        # 'resnet50_augmix_avg3',
        # 'resnet50_augmix_avg5',
        # 'resnet50_autoaug',
        # 'resnet50_autoaug_avg3',
        # 'resnet50_autoaug_avg5',
    ]

    errors_wrt_baseline = []
    scores_on_clean_dataset = []
    all_scores = []
    errors = []
    for exp in experiments:
        with open(compile_scores_dir.joinpath(rf'{exp}/results/results.yaml')) as f:
            results_summary = yaml.safe_load(f)
        # Choose the results from the last epoch
        epoch = sorted(results_summary[dataset_name]['epoch_wise_results'])[-1]
        result = results_summary[dataset_name]['epoch_wise_results'][epoch]
        # Extract the necessary details
        errors_wrt_baseline.append(result['errors_wrt_baseline'][baseline_exp_dir])
        scores_on_clean_dataset.append(result['scores']['clean'])
        all_scores.append(result['scores'])
        errors.append(result['errors'])

    save_scores(experiments, scores_on_clean_dataset, errors, errors_wrt_baseline, compile_scores_dir)
    plot_scores(experiments, all_scores, dataset_name, compile_scores_dir)


if __name__ == '__main__':
    run_flow()
    print('Run finished!')
