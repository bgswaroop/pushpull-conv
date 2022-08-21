import math
from itertools import combinations

import torch
from torch.linalg import vector_norm


class DSHSamplingLoss(torch.nn.Module):
    # A sampling loss version of Deep Supervised Hashing (DSH) Loss
    # https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Liu_Deep_Supervised_Hashing_CVPR_2016_paper.pdf

    def __init__(self, batch_size, margin, alpha=1e-4, num_experiments=10, sampling_ratio=10):
        """
        :param batch_size: The size of the batch in the dataloader
        :param margin: regularization parameter for DSH loss - usually set as twice the hash length
        :param alpha: regularization parameter for the quantization loss
        :param num_experiments: num exp to conduct using the strategy of sampling
        :param sampling_ratio: the ratio of elements of num_same_class_pairs:num_diff_class_pairs
        expressed as 1:sampling_ratio
        """
        super().__init__()
        self.comb = torch.tensor(list(combinations(range(batch_size), 2)))
        self.batch_size = batch_size
        self.margin = margin
        self.alpha = alpha
        self._num_experiments = num_experiments
        self._sampling_ratio = sampling_ratio

    def forward(self, predictions, ground_truth_classes) -> torch.tensor:
        # Preprocess the inputs for computing the loss (simulating the inputs for a siamese network)

        if len(ground_truth_classes) == self.batch_size:
            comb = self.comb
        else:  # to handle different batch size (for instance, the last batch)
            comb = torch.tensor(list(combinations(range(len(ground_truth_classes)), 2)))
        targets = (ground_truth_classes[comb[:, 0]] == ground_truth_classes[comb[:, 1]]).int()

        same_class_indices = torch.where(targets == 1)[0].long()
        diff_class_indices = torch.where(targets == 0)[0].long()
        sampling_size = math.floor(len(same_class_indices) * 2 / 3)

        loss = torch.tensor(0., requires_grad=True, device=predictions.device)
        if sampling_size <= 0 or len(diff_class_indices) == 0:
            return loss

        # Determine the parameters for sampling
        # self._num_experiments = math.ceil((len(comb) / (2 * sampling_size)) * (3 / 2))
        sampled_same_class_indices = same_class_indices[torch.multinomial(
            torch.ones(self._num_experiments, len(same_class_indices)), sampling_size
        )]
        sampled_diff_class_indices = diff_class_indices[torch.multinomial(
            torch.ones(self._num_experiments, len(diff_class_indices)), sampling_size * self._sampling_ratio
        )]

        for exp_num in range(self._num_experiments):
            # sample a mini-batch
            sampled_indices = torch.cat((sampled_same_class_indices[exp_num], sampled_diff_class_indices[exp_num]))
            h1 = predictions[comb[sampled_indices, 0]]
            h2 = predictions[comb[sampled_indices, 1]]
            t = targets[sampled_indices]
            # compute loss
            minibatch_loss = self._dsh_loss(h1, h2, t)
            loss = loss + minibatch_loss
        loss = loss / self._num_experiments
        return loss

    def _dsh_loss(self, h1, h2, t):
        """
        :param h1: hash1
        :param h2: hash2
        :param t: targets
        :return: dsh_loss
        """
        # hash_dist == hamming_distance when h1 and h2 are perfect binary
        hash_dist = torch.square(vector_norm(h1 - h2, ord=2, dim=1))

        # Loss term for similar-pairs (i.e when target == 1)
        # It punishes similar images mapped to different binary codes
        l1 = 0.5 * t * hash_dist

        # Loss term for dissimilar-pairs (i.e when target == 0)
        # It punishes dissimilar images mapped to close binary codes
        l2 = 0.5 * (1 - t) * torch.max(self.margin - hash_dist, torch.zeros_like(hash_dist))

        # Regularization term
        l3 = self.alpha * (vector_norm(torch.abs(h1) - torch.ones_like(h1), ord=1, dim=1) +
                           vector_norm(torch.abs(h2) - torch.ones_like(h2), ord=1, dim=1))

        minibatch_loss = torch.mean(l1 + l2 + l3)
        return minibatch_loss
