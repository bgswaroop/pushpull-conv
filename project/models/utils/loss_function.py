import math
import random
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


# Code adapted from - https://github.com/swuxyj/DeepHash-pytorch/blob/993909f4e0e9f599b503ce5c380e2fcc7c6824f7/CSQ.py
class CSQLoss(torch.nn.Module):
    def __init__(self, num_classes, hash_length, quantization_weight, device='cuda:0'):
        super(CSQLoss, self).__init__()
        self.hash_targets = self._get_hash_targets(num_classes, hash_length).to(device)
        self.criterion = torch.nn.BCELoss()
        self.quantization_weight = quantization_weight  # lambda

    def forward(self, y_hat, y):
        y_hat = y_hat.tanh()
        hash_center = self.hash_targets[y]
        center_loss = self.criterion(0.5 * (y_hat + 1), 0.5 * (hash_center + 1))

        Q_loss = (y_hat.abs() - 1).pow(2).mean()
        return center_loss + self.quantization_weight * Q_loss

    def _get_hash_targets(self, num_classes, hash_length):
        H_K = self._hadamard(hash_length)
        H_2K = torch.cat((H_K, -H_K), dim=0)
        hash_targets = H_2K[:num_classes].float()

        if H_2K.shape[0] == num_classes:
            return hash_targets

        hash_targets.resize_(num_classes, hash_length)
        for _ in range(20):
            for index in range(H_2K.shape[0], num_classes):
                ones = torch.ones(hash_length)
                # Bernoulli distribution
                sa = random.sample(list(range(hash_length)), hash_length // 2)
                ones[sa] = -1
                hash_targets[index] = ones
            # to find average/min pairwise distance
            c = []
            for i in range(num_classes):
                for j in range(i + 1, num_classes):
                    TF = sum(hash_targets[i] != hash_targets[j])
                    c.append(TF)
            c = torch.tensor(c).float()

            # choose min(c) in the range of K/4 to K/3
            # see in https://github.com/yuanli2333/Hadamard-Matrix-for-hashing/issues/1
            # but, it is hard when bit is small
            if torch.min(c) > hash_length / 4 and torch.mean(c) >= hash_length / 2:
                print(torch.min(c), torch.mean(c))
                break
        return hash_targets

    @staticmethod
    def _hadamard(n, dtype=int):
        """
        Construct a Hadamard matrix.

        Constructs an n-by-n Hadamard matrix, using Sylvester's
        construction. `n` must be a power of 2.

        Parameters
        ----------
        n : int
            The order of the matrix. `n` must be a power of 2.
        dtype : dtype, optional
            The data type of the array to be constructed.

        Returns
        -------
        H : (n, n) ndarray
            The Hadamard matrix.

        Examples
        --------
        >>> CSQLoss.hadamard(2, dtype=complex)
        array([[ 1.+0.j,  1.+0.j],
               [ 1.+0.j, -1.-0.j]])
        >>> CSQLoss.hadamard(4)
        array([[ 1,  1,  1,  1],
               [ 1, -1,  1, -1],
               [ 1,  1, -1, -1],
               [ 1, -1, -1,  1]])

        """
        if n < 1:
            lg2 = 0
        else:
            lg2 = int(math.log(n, 2))
        if 2 ** lg2 != n:
            raise ValueError("n must be an positive integer and a power of 2")

        H = torch.tensor([[1]], dtype=dtype)

        # Sylvester's construction
        for _ in range(0, lg2):
            H = torch.vstack((torch.hstack((H, H)), torch.hstack((H, -H))))

        return H
