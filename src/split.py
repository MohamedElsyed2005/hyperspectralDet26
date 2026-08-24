"""Deterministic train/val index split, shared by train.py and any eval script.

Previously train.py used `random_split` directly on a single
HyperspectralDetDataset instance, then relied on that *same* instance's
`.transforms` for both the train and val Subsets -- which meant augmentation
couldn't be applied to train only. This module just produces the index lists;
callers build two dataset instances (one with augmentation, one without) and
Subset() each with the appropriate half, so the split stays identical to
before (same seed, same proportions) while train/val get independent transforms.
"""

import torch


def get_train_val_indices(n_samples: int, val_split: float, seed: int):
    """Returns (train_indices, val_indices) as plain lists of ints.

    Equivalent split logic to `torch.utils.data.random_split`, factored out so
    it can be reused by both training and any standalone evaluation script
    without having to construct a full dataset twice just to get indices.
    """
    n_val = max(1, int(n_samples * val_split))
    n_train = n_samples - n_val

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_samples, generator=generator).tolist()

    train_indices = perm[:n_train]
    val_indices = perm[n_train:]
    return train_indices, val_indices
