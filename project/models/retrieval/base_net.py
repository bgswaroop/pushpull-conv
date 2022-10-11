from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import OneCycleLR

from ..utils import CSQLoss, compute_map_score
from ..utils.push_pull_unit import PushPullConv2DUnit


class BaseNet(pl.LightningModule):
    def __init__(self):
        super(BaseNet, self).__init__()
        # self.loss = DSHSamplingLoss(
        #     batch_size=self.hparams.batch_size,
        #     margin=self.hparams.hash_length * 2,
        #     alpha=self.hparams.quantization_weight
        # )
        hash_targets_file = Path(self.hparams.logs_dir).joinpath('hash_targets.pt')
        hash_targets = torch.load(hash_targets_file) if hash_targets_file.exists() else None
        self.loss = CSQLoss(
            num_classes=self.hparams.num_classes,
            hash_length=self.hparams.hash_length,
            quantization_weight=self.hparams.quantization_weight,
            hash_targets=hash_targets,
        )
        if not hash_targets_file.exists():
            Path(self.hparams.logs_dir).mkdir(exist_ok=True, parents=True)
            torch.save(self.loss.hash_targets, hash_targets_file)

    def on_train_start(self) -> None:
        self.loss.hash_targets = self.loss.hash_targets.to(self.device)

    def on_validation_start(self) -> None:
        self.loss.hash_targets = self.loss.hash_targets.to(self.device)

    def on_predict_start(self) -> None:
        self.loss.hash_targets = self.loss.hash_targets.to(self.device)

    def on_test_start(self) -> None:
        self.loss.hash_targets = self.loss.hash_targets.to(self.device)

    def evaluate_step(self, batch, batch_idx, dataloader_idx=0, *, stage=None):
        x, y, y_soft = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        if stage in {'train', 'val', 'test'}:
            self.log('loss', {stage: loss}, on_epoch=True, on_step=False, add_dataloader_idx=False, sync_dist=True)
        hash_code = torch.sign(y_hat)
        return hash_code, y, loss

    def evaluate_epoch(self, outputs, *, stage):
        database_hash = torch.concat([x['database']['predictions'] for x in outputs[0]])
        database_gt = torch.concat([x['database']['ground_truths'] for x in outputs[0]])
        # train_score = compute_map_score(train_hash, train_gt, train_hash, train_gt, self.device)

        query_hash = torch.concat([x[stage]['predictions'] for x in outputs[1]])
        query_gt = torch.concat([x[stage]['ground_truths'] for x in outputs[1]])
        query_score = compute_map_score(database_hash, database_gt, query_hash, query_gt, self.device)

        for key in query_score.keys():
            # self.log(f'{key}_mAP', {stage: train_score[key], stage: query_score[key]})
            self.log(f'{key}_mAP', {stage: query_score[key]}, add_dataloader_idx=False, sync_dist=True)
            if key in {'top50', 'top200', 'top1000'} and stage == 'val':
                self.log(f'{key}_mAP_{stage}', query_score[key], logger=False, add_dataloader_idx=False, sync_dist=True)

    def training_step(self, batch, batch_idx):
        hash_code, y, loss = self.evaluate_step(batch, batch_idx, stage='train')
        return loss

    # def training_epoch_end(self, outputs) -> None:
    #     if hasattr(self, 'features'):
    #         first_layer = self.features._modules['0']
    #     elif hasattr(self, 'conv1'):
    #         first_layer = self.conv1
    #     else:
    #         print('\nWarning - Push Pull layer not found. Skipping saving the logs')
    #         return
    #
    #     if type(first_layer) != PushPullConv2DUnit:
    #         print('\nWarning - Push Pull layer not found. Skipping saving the logs')
    #         return
    #
    #     Path(self.logger.log_dir).mkdir(exist_ok=True, parents=True)
    #     log_file = Path(self.logger.log_dir).joinpath('layer0_pull_inhibition_strength.csv')
    #
    #     if type(first_layer.pull_inhibition_strength) == float:
    #         data = torch.Tensor([first_layer.pull_inhibition_strength])
    #     else:
    #         data = first_layer.pull_inhibition_strength.cpu()
    #     index = [str(x) for x in range(len(data))]
    #     if log_file.exists():
    #         df = pd.read_csv(log_file)
    #     else:
    #         df = pd.DataFrame(columns=index)
    #     data = pd.DataFrame(data.view((1, -1)).detach(), columns=index)
    #     df = pd.concat([df, data], ignore_index=True)
    #     df.to_csv(log_file, index=False)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        stage = ['database', 'val'][dataloader_idx]
        hash_code, y, loss = self.evaluate_step(batch, batch_idx, stage=stage)
        if stage == 'val':
            self.log(f'loss_{stage}', loss, add_dataloader_idx=False, logger=False, sync_dist=True)
        return {f'{stage}': {'predictions': hash_code, 'ground_truths': y}}

    def validation_epoch_end(self, outputs) -> None:
        self.evaluate_epoch(outputs, stage='val')

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        hash_code, y, loss = self.evaluate_step(batch, batch_idx)
        return {'predictions': hash_code, 'ground_truths': y}

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        stage = ['database', 'test'][dataloader_idx]
        hash_code, y, loss = self.evaluate_step(batch, batch_idx, stage=stage)
        return {f'{stage}': {'predictions': hash_code, 'ground_truths': y}}

    def test_epoch_end(self, outputs) -> None:
        self.evaluate_epoch(outputs, stage='test')


    def configure_optimizers(self):

        optimizer = torch.optim.SGD(self.parameters(),
                                    lr=self.hparams.lr_base,
                                    momentum=0.9,
                                    weight_decay=self.hparams.weight_decay)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=0.1,
            epochs=self.trainer.max_epochs,
            steps_per_epoch=self.hparams.steps_per_epoch,
        )
        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'monitor': 'loss_val', "interval": "epoch"}}

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True,
        #                                                        mode='min', factor=0.1, patience=5)
        # return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'monitor': 'loss_val'}}
        # params_list = [
        #     {'params': self.features.parameters(), 'lr': self.hparams.lr_base * self.hparams.lr_multiplier},
        #     {'params': self.classifier.parameters()}
        # ]
        # optimizer = torch.optim.Adam(self.parameters(),
        #                              lr=self.hparams.lr_base,
        #                              weight_decay=self.hparams.weight_decay)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True,
        #                                                        mode='min', factor=0.1, patience=5)
        # return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'monitor': 'loss_val'}}
