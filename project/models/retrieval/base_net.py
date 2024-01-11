from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import OneCycleLR, StepLR

from ..utils import CSQLoss, compute_map_score


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
            self.log(f'{key}_mAP', {stage: query_score[key]}, add_dataloader_idx=False, sync_dist=True)
            if key in {'top50', 'top200'} and stage == 'val':
                self.log(f'{key}_mAP_{stage}', query_score[key], logger=False, add_dataloader_idx=False, sync_dist=True)

    def training_step(self, batch, batch_idx):
        hash_code, y, loss = self.evaluate_step(batch, batch_idx, stage='train')
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        stage = ['val', 'database'][dataloader_idx]
        hash_code, y, loss = self.evaluate_step(batch, batch_idx, stage=stage)
        if stage == 'val':
            self.log(f'loss_{stage}', loss, add_dataloader_idx=False, logger=False, sync_dist=True)
        return {f'{stage}': {'predictions': hash_code, 'ground_truths': y}}

    # def validation_epoch_end(self, outputs) -> None:
    #     self.evaluate_epoch(outputs, stage='val')

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
        # scheduler = OneCycleLR(
        #     optimizer,
        #     max_lr=0.1,
        #     epochs=self.trainer.max_epochs,
        #     steps_per_epoch=self.hparams.steps_per_epoch,
        # )
        # return {'optimizer': optimizer,
        #         'lr_scheduler': {'scheduler': scheduler, 'monitor': 'loss_val', "interval": "epoch"}}

        if self.hparams.lr_scheduler in {'step_lr'}:
            scheduler = StepLR(
                optimizer,
                step_size=self.hparams.lr_step_size,
                gamma=self.hparams.lr_gamma,
            )
            return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler}}
        elif self.hparams.lr_scheduler in {'one_cycle'}:
            scheduler = OneCycleLR(
                optimizer,
                div_factor=int(self.hparams.lr_max / self.hparams.lr_initial),
                max_lr=self.hparams.lr_max,
                final_div_factor=int(self.hparams.lr_max / self.hparams.lr_end),
                pct_start=0.3,
                three_phase=self.hparams.lr_three_phase,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=self.hparams.steps_per_epoch,
            )
            return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}
        else:
            raise ValueError('Invalid LR scheduler')
