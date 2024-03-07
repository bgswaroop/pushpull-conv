import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
from lets_plot import *


if __name__ == '__main__':
    model = nn.Sequential(
        nn.Linear(2, 4),
        nn.Linear(4, 1)
    )

    data = torch.randn(100, 2)
    dataset = torch.utils.data.TensorDataset(data)
    data_loader = DataLoader(dataset, batch_size=10)
    num_epochs = 100

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1.0,             # Smith et. al. (2019)
        steps_per_epoch=len(data_loader),
        epochs=num_epochs,
        pct_start=0.3,          # proportion of epochs spent in warmup
        anneal_strategy='cos',
        cycle_momentum=True,
        base_momentum=0.9,      # Based on Szegedy at al. 2017
        max_momentum=0.95,
        div_factor=20.,         # Compute initial lr = max_lr / div_factor
        final_div_factor=1e3,   # Compute final lr = initial_lr / final_div_factor
        three_phase=False,
    )

    steps = np.arange(num_epochs * len(data_loader)) / len(data_loader)
    lr = []
    for epoch in range(num_epochs):
        for batch in data_loader:
            scheduler.step()
            lr.append(scheduler.get_last_lr()[0])

    plt.figure()
    plt.plot(steps, lr)
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.ylim([-0.0001, 0.011])
    plt.show()

    print('')
