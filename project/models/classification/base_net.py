from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, OneCycleLR
from torchmetrics.functional import accuracy
import scipy
from pathlib import Path

class BaseNet(pl.LightningModule):
    def __init__(self):
        super(BaseNet, self).__init__()

    def training_step(self, batch, batch_idx):
        loss, acc1, acc5 = self.evaluate(batch, stage='train', batch_idx=batch_idx)
        return loss

    # def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
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
    #     if type(first_layer.pull_inhibition_strength) in {float, int}:
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

    def evaluate(self, batch, stage=None, batch_idx=None):
        x, y, y_soft = batch
        y_hat = self(x)
        acc1 = accuracy(y_hat, y, top_k=1)
        acc5 = accuracy(y_hat, y, top_k=5)
        loss = None
        if stage == 'test':
            self.log('top1-accuracy', {stage: acc1}, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
            self.log('top5-accuracy', {stage: acc5}, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        elif stage in {'train', 'val'}:
            if self.hparams.loss_type == 'distillation_loss':
                # https://github.com/haitongli/knowledge-distillation-pytorch/blob/9937528f0be0efa979c745174fbcbe9621cea8b7/model/net.py#L100
                alpha = self.hparams.distillation_loss_alpha
                T = self.hparams.distillation_loss_temp  # temperature
                loss = (1. - alpha) * F.cross_entropy(y_hat, y) + \
                       alpha * (T ** 2) * F.kl_div(input=F.log_softmax(y_hat / T, dim=1),
                                                   target=F.log_softmax(y_soft / T, dim=1),
                                                   reduction='batchmean', log_target=True)
            else:
                loss = F.cross_entropy(y_hat, y)
            self.log('loss', {stage: loss}, sync_dist=True)
            self.log('top1-accuracy', {stage: acc1}, on_epoch=True, on_step=False, sync_dist=True)
            self.log('top5-accuracy', {stage: acc5}, on_epoch=True, on_step=False, sync_dist=True)

        return loss, acc1, acc5

    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):
        loss, acc1, acc5 = self.evaluate(batch, stage='val', batch_idx=batch_idx)
        self.log('loss_val', loss, add_dataloader_idx=False, logger=False, sync_dist=True)
        self.log('top1_acc_val', acc1, add_dataloader_idx=False, logger=False, sync_dist=True)
        self.log('top5_acc_val', acc5, add_dataloader_idx=False, logger=False, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, stage='test')

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        x, y, y_soft = batch
        y_hat = self(x)
        return {'predictions': y_hat, 'ground_truths': y}

    def configure_optimizers(self):

        optimizer = torch.optim.SGD(self.parameters(),
                                    lr=self.hparams.lr_base,
                                    momentum=0.9,
                                    weight_decay=self.hparams.weight_decay)

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


