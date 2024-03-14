import torch
from pathlib import Path
import yaml


def update_models():
    root = Path(r'/scratch/p288722/runtime_data/pushpull-conv')
    for exp_set in sorted(root.glob('*_classification')):
        for exp in sorted(exp_set.glob('*_avg*')):
            if 'inht' not in exp.name:
                hparams_file = exp.joinpath('hparams.yaml')
                ckpts = exp.glob('checkpoints/*.ckpt')

                # Fix hparams
                with open(hparams_file, 'r') as f:
                    hparams = yaml.safe_load(f)
                hparams['trainable_pull_inhibition'] = False
                with open(hparams_file, 'w+') as f:
                    yaml.safe_dump(hparams, f)

                # Fix checkpoints
                for ckpt in sorted(ckpts):

                    if ckpt.is_symlink():
                        continue

                    model = torch.load(ckpt)
                    model['hyper_parameters']['trainable_pull_inhibition'] = False
                    model['state_dict']['conv1.pull_inhibition_strength'] = 1.0
                    torch.save(model, ckpt)

                    print(ckpt)


if __name__ == '__main__':
    update_models()
