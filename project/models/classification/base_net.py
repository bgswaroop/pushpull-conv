from pathlib import Path
from typing import Any

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torchmetrics.functional import accuracy

from ..utils.push_pull_unit import PushPullConv2DUnit


class BaseNet(pl.LightningModule):
    def __init__(self):
        super(BaseNet, self).__init__()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('loss', {'train': loss}, on_epoch=True, on_step=False)
        acc1 = accuracy(y_hat, y, top_k=1)
        acc5 = accuracy(y_hat, y, top_k=5)
        self.log('top1-accuracy', {'train': acc1}, on_epoch=True, on_step=False)
        self.log('top5-accuracy', {'train': acc5}, on_epoch=True, on_step=False)
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

        if type(first_layer.pull_inhibition_strength) in {float, int}:
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
        loss = F.cross_entropy(y_hat, y)
        self.log('loss', {'val': loss})
        self.log('loss_val', loss, add_dataloader_idx=False, logger=False)
        acc1 = accuracy(y_hat, y, top_k=1)
        acc5 = accuracy(y_hat, y, top_k=5)
        self.log('top1-accuracy', {'val': acc1}, on_epoch=True, on_step=False)
        self.log('top5-accuracy', {'val': acc5}, on_epoch=True, on_step=False)
        self.log('top1_acc_val', acc1, add_dataloader_idx=False, logger=False)
        self.log('top5_acc_val', acc5, add_dataloader_idx=False, logger=False)
        return loss

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        x, y = batch
        y_hat = self(x)
        # y_hat = torch.argmax(F.softmax(y_hat, dim=1), dim=1)
        return {'predictions': y_hat, 'ground_truths': y}

    def configure_optimizers(self):

        optimizer = torch.optim.SGD(self.parameters(),
                                    lr=self.hparams.learning_rate,
                                    momentum=0.9,
                                    weight_decay=self.hparams.weight_decay)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True,
                                                               mode='min', factor=0.1, patience=5)

        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'monitor': 'loss_val'}}
