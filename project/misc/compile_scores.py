import json
from pathlib import Path

import pandas as pd


def save_scores(exp_names, exp_results):
    row_header = ['Error', 'mCE',
                  'Gauss', 'Shot', 'Impulse',  # Noise
                  'Defocus', 'Glass', 'Motion', 'Zoom',  # Blur
                  'Snow', 'Frost', 'Fog', 'Bright',  # Weather
                  'Contrast', 'Elastic', 'Pixel', 'JPEG'  # Digital
                  ]
    df1 = pd.DataFrame(index=exp_names, columns=row_header)
    for idx, result in zip(exp_names, exp_results):
        df1['Error'][idx] = 1 - result['mAP']['clean']
        df1['mCE'][idx] = result['mCE']
        df1['Gauss'][idx] = result['CE'].get('gaussian_noise', None)
        df1['Shot'][idx] = result['CE'].get('shot_noise', None)
        df1['Impulse'][idx] = result['CE'].get('impulse_noise', None)
        df1['Defocus'][idx] = result['CE'].get('defocus_blur', None)
        df1['Glass'][idx] = result['CE'].get('glass_blur', None)
        df1['Motion'][idx] = result['CE'].get('motion_blur', None)
        df1['Zoom'][idx] = result['CE'].get('zoom_blur', None)
        df1['Snow'][idx] = result['CE'].get('snow', None)
        df1['Frost'][idx] = result['CE'].get('frost', None)
        df1['Fog'][idx] = result['CE'].get('fog', None)
        df1['Bright'][idx] = result['CE'].get('brightness', None)
        df1['Contrast'][idx] = result['CE'].get('contrast', None)
        df1['Elastic'][idx] = result['CE'].get('elastic_transform', None)
        df1['Pixel'][idx] = result['CE'].get('pixelate', None)
        df1['JPEG'][idx] = result['CE'].get('jpeg_compression', None)

    row_header[1] = 'Rel. mCE'
    df2 = pd.DataFrame(index=exp_names, columns=row_header)
    for idx, result in zip(exp_names, exp_results):
        df2['Error'][idx] = 1 - result['mAP']['clean']
        df2['Rel. mCE'][idx] = result['relative_mCE']
        df2['Gauss'][idx] = result['relative_CE'].get('gaussian_noise', None)
        df2['Shot'][idx] = result['relative_CE'].get('shot_noise', None)
        df2['Impulse'][idx] = result['relative_CE'].get('impulse_noise', None)
        df2['Defocus'][idx] = result['relative_CE'].get('defocus_blur', None)
        df2['Glass'][idx] = result['relative_CE'].get('glass_blur', None)
        df2['Motion'][idx] = result['relative_CE'].get('motion_blur', None)
        df2['Zoom'][idx] = result['relative_CE'].get('zoom_blur', None)
        df2['Snow'][idx] = result['relative_CE'].get('snow', None)
        df2['Frost'][idx] = result['relative_CE'].get('frost', None)
        df2['Fog'][idx] = result['relative_CE'].get('fog', None)
        df2['Bright'][idx] = result['relative_CE'].get('brightness', None)
        df2['Contrast'][idx] = result['relative_CE'].get('contrast', None)
        df2['Elastic'][idx] = result['relative_CE'].get('elastic_transform', None)
        df2['Pixel'][idx] = result['relative_CE'].get('pixelate', None)
        df2['JPEG'][idx] = result['relative_CE'].get('jpeg_compression', None)

    df1.to_csv(r'_csv_results/CE.csv')
    df2.to_csv(r'_csv_results/rel_CE.csv')


def run_flow():
    base_dir = Path(r'/data/p288722/runtime_data/deep_hashing/dsh_pp_ResNet50_48bit_step_lr_decay_run1')
    dataset_name = 'CIFAR-10-C-224x224'
    baseline_results_file = '/data/p288722/runtime_data/deep_hashing/dsh_pp_ResNet50_48bit_step_lr_decay_run1/ResNet50_48bit/results/mAP_scores.json'
    exp_names = [
        'ResNet50_48bit',
        # 'ResNet50_pp7x7_avg3',
        'ResNet50_pp7x7_avg3_inh1',
        'ResNet50_pp7x7_avg3_inh2',
        'ResNet50_pp7x7_avg3_inh3',
        'ResNet50_pp7x7_avg3_inh4',
        'ResNet50_pp7x7_avg5_inh1',
        'ResNet50_pp7x7_avg5_inh2',
        'ResNet50_pp7x7_avg5_inh3',
        'ResNet50_pp7x7_avg5_inh4',
    ]

    exp_results = []
    for exp_name in exp_names:
        with open(base_dir.joinpath(rf'{exp_name}/results/all_scores.json')) as f:
            exp_results.append(json.load(f)[dataset_name][baseline_results_file])

    save_scores(exp_names, exp_results)


if __name__ == '__main__':
    run_flow()
    print('Run finished!')
