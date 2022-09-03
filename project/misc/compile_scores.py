import json
from pathlib import Path

import pandas as pd


def save_scores(exp_names, robustness_metrics, scores_on_clean_dataset, save_dir):
    row_header = ['Error', 'mCE',
                  'Gauss', 'Shot', 'Impulse',  # Noise
                  'Defocus', 'Glass', 'Motion', 'Zoom',  # Blur
                  'Snow', 'Frost', 'Fog', 'Bright',  # Weather
                  'Contrast', 'Elastic', 'Pixel', 'JPEG'  # Digital
                  ]

    summary_results = dict()
    for topk in scores_on_clean_dataset[0].keys():
        df_CE = pd.DataFrame(index=exp_names, columns=row_header)
        for idx, r, s in zip(exp_names, robustness_metrics, scores_on_clean_dataset):
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

    row_header[1] = 'Rel. mCE'
    for topk in scores_on_clean_dataset[0].keys():
        df_rCE = pd.DataFrame(index=exp_names, columns=row_header)
        for idx, r, s in zip(exp_names, robustness_metrics, scores_on_clean_dataset):
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

def run_flow():
    compile_scores_dir = Path(r'/data/p288722/runtime_data/pushpull-conv/resnet18_imagenet200_classification')
    dataset_name = 'imagenet200-c'
    baseline_exp_dir = '/data/p288722/runtime_data/pushpull-conv/resnet18_imagenet200_classification/resnet18'
    experiments = [
        'resnet18',
        # 'resnet18_pp7x7_avg3',
        'resnet18_pp7x7_avg3_inh1',
        'resnet18_pp7x7_avg3_inh2',
        'resnet18_pp7x7_avg3_inh3',
        'resnet18_pp7x7_avg3_inh4',
        'resnet18_pp7x7_avg5_inh1',
        'resnet18_pp7x7_avg5_inh2',
        'resnet18_pp7x7_avg5_inh3',
        'resnet18_pp7x7_avg5_inh4',
    ]

    robustness_metrics = []
    scores_on_clean_dataset = []
    for exp in experiments:
        with open(compile_scores_dir.joinpath(rf'{exp}/results/all_scores.json')) as f:
            exp_data = json.load(f)
            robustness_metrics.append(exp_data[dataset_name]['scores_versus_baseline'][baseline_exp_dir])
            scores_on_clean_dataset.append(exp_data[dataset_name]['scores']['clean'])

    save_scores(experiments, robustness_metrics, scores_on_clean_dataset, compile_scores_dir)


if __name__ == '__main__':
    run_flow()
    print('Run finished!')
