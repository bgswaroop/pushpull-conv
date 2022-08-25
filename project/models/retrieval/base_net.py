from pathlib import Path
from typing import Any

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import EPOCH_OUTPUT

from ..utils import DSHSamplingLoss, CSQLoss, compute_map_score
from ..utils.push_pull_unit import PushPullConv2DUnit


class BaseNet(pl.LightningModule):
    def __init__(self):
        super(BaseNet, self).__init__()
        self.loss = DSHSamplingLoss(
            batch_size=self.hparams.batch_size,
            margin=self.hparams.hash_length * 2,
            alpha=self.hparams.quantization_weight
        )
        # self.loss = CSQLoss(
        #     num_classes=self.hparams.num_classes,
        #     hash_length=self.hparams.hash_length,
        #     quantization_weight=self.hparams.quantization_weight,
        # )

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('loss', {'train': loss}, on_epoch=True, on_step=False)
        return loss

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        if hasattr(self, 'features'):
            first_layer = self.features._modules['0']
        elif hasattr(self, 'conv1'):
            first_layer = self.conv1
        else:
            print('\nWarning - Push Pull layer not found. Skipping saving the logs')
            return

        if type(first_layer) != PushPullConv2DUnit:
            print('\nWarning - Push Pull layer not found. Skipping saving the logs')
            return

        Path(self.logger.log_dir).mkdir(exist_ok=True, parents=True)
        log_file = Path(self.logger.log_dir).joinpath('layer0_pull_inhibition_strength.csv')

        if type(first_layer.pull_inhibition_strength) == float:
            data = torch.Tensor([first_layer.pull_inhibition_strength])
        else:
            data = first_layer.pull_inhibition_strength.cpu()
        index = [str(x) for x in range(len(data))]
        if log_file.exists():
            df = pd.read_csv(log_file)
        else:
            df = pd.DataFrame(columns=index)
        data = pd.DataFrame(data.view((1, -1)).detach(), columns=index)
        df = pd.concat([df, data], ignore_index=True)
        df.to_csv(log_file, index=False)

    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):
        x, y = batch
        y_hat = self(x)
        hash_code = torch.sign(y_hat)
        loss = self.loss(y_hat, y)

        dataloader_type = ['train', 'val', 'test'][dataloader_idx]
        if dataloader_type == 'val':
            self.log('loss', {dataloader_type: loss}, add_dataloader_idx=False, on_epoch=True, on_step=False)
            self.log(f'loss_{dataloader_type}', loss, add_dataloader_idx=False, logger=False)

        return {f'{dataloader_type}_data': {'predictions': hash_code, 'ground_truths': y}}

    def validation_epoch_end(self, outputs) -> None:
        train_hash = torch.concat([x['train_data']['predictions'] for x in outputs[0]])
        train_gt = torch.concat([x['train_data']['ground_truths'] for x in outputs[0]])
        # train_score = compute_map_score(train_hash, train_gt, train_hash, train_gt, self.device)

        val_hash = torch.concat([x['val_data']['predictions'] for x in outputs[1]])
        val_gt = torch.concat([x['val_data']['ground_truths'] for x in outputs[1]])
        val_score = compute_map_score(train_hash, train_gt, val_hash, val_gt, self.device)

        if len(outputs) == 3:  # train + val + test
            test_hash = torch.concat([x['test_data']['predictions'] for x in outputs[2]])
            test_gt = torch.concat([x['test_data']['ground_truths'] for x in outputs[2]])
            test_score = compute_map_score(train_hash, train_gt, test_hash, test_gt, self.device)

        for key in val_score.keys():
            # self.log(f'{key}_mAP', {'train': train_score[key], 'val': val_score[key]})
            self.log(f'{key}_mAP', {'val': val_score[key]})
            if key == 'top50':
                self.log(f'top50_mAP_val', val_score[key], logger=False)
            if key == 'top200':
                self.log(f'top200_mAP_val', val_score[key], logger=False)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        x, y = batch
        y_hat = self(x)
        hash_code = torch.sign(y_hat)
        return {'hash_codes': hash_code, 'ground_truths': y}

    def configure_optimizers(self):

        optimizer = torch.optim.SGD(self.parameters(),
                                    lr=self.hparams.learning_rate,
                                    momentum=0.9,
                                    weight_decay=self.hparams.weight_decay)

        # optimizer = torch.optim.Adam(self.parameters(),
        #                              lr=self.hparams.learning_rate,
        #                              weight_decay=self.hparams.weight_decay)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True,
                                                               mode='min', factor=0.1, patience=5)

        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'monitor': 'loss_val'}}
