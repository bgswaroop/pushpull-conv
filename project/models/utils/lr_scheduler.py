import torch


class CustomThreePhase:
    """
    Three-phase LR schedule.
    Phase 1 - Cosine warmup from lr_start to lr_peak from step 0 to max_steps*milestones_pct[0] steps
    Phase 2 - Cosine decay from lr_peak to lr_end from max_steps*milestones_pct[0] to max_steps*milestones_pct[1] steps
    Phase 3 - Fixed LR set to lr_end from max_steps*milestones_pct[1] steps until max_steps
    """

    def __init__(self, last_epoch, milestones_pct=(0.2, 0.7), lr_start=1e-3, lr_peak=1e-1, lr_end=1e-4):

        assert len(milestones_pct) == 2, 'milestones_pct needs to be a two element list/tuple'
        assert milestones_pct[0] < milestones_pct[1], 'milestones_pct[0] < milestones_pct[1] does not hold!'
        assert lr_start <= lr_peak, 'lr_start needs to be <= than lr_peak'
        assert lr_peak >= lr_end, 'lr_peak needs to be >= than lr_end'

        self.milestone1 = round(last_epoch * milestones_pct[0])
        self.milestone2 = round(last_epoch * milestones_pct[1])
        self.lr_start = lr_start
        self.lr_peak = lr_peak
        self.lr_end = lr_end
        self.lr = lr_start

    def __call__(self, epoch):
        if epoch == 0:
            lr = self.lr_start

        elif epoch <= self.milestone1:
            nmax = self.lr_peak
            nmin = self.lr_start
            Tmax = self.milestone1
            Tcur = epoch + self.milestone1 - 1
            lr = nmin + 0.5 * (nmax - nmin) * (1 + torch.cos(torch.tensor(Tcur / Tmax) * torch.pi))

        elif epoch <= self.milestone2:
            nmax = self.lr_peak
            nmin = self.lr_end
            Tmax = self.milestone2 - self.milestone1 + 1
            Tcur = epoch - self.milestone1
            lr = nmin + 0.5 * (nmax - nmin) * (1 + torch.cos(torch.tensor(Tcur / Tmax) * torch.pi))

        else:
            lr = self.lr_end

        return lr
