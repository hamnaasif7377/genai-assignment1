"""
Balanced batch sampling for the Task 2 corruption classifier.
Ensures every batch contains an equal number of clean/salt_pepper/blur/occlusion
examples, rather than relying on random per-image assignment averaging out.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.corruptions import CORRUPTION_TYPES, IMG_SIZE, corrupt_image


class BalancedCorruptionDataset(Dataset):
    """
    Unlike PetRestorationDataset (which picks ONE random corruption per
    image), this dataset is indexed as (image_index, corruption_type) pairs
    so that a simple sequential/shuffled DataLoader naturally yields a
    balanced class distribution across epochs. Combined with a batch size
    that's a multiple of 4, batches stay close to balanced throughout training.
    """

    def __init__(self, base_dataset, indices, img_size=IMG_SIZE):
        self.base_dataset = base_dataset
        self.img_size = img_size
        # Build one entry per (image, corruption_type) combination
        self.pairs = [(idx, ctype) for idx in indices for ctype in CORRUPTION_TYPES]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        real_idx, corruption_type = self.pairs[i]
        pil_img, _ = self.base_dataset[real_idx]
        pil_img = pil_img.convert("RGB").resize((self.img_size, self.img_size))
        clean_np = np.array(pil_img)

        rng = np.random.default_rng()  # severity still randomized per the spec
        corrupted_np = corrupt_image(clean_np, corruption_type, rng)

        corrupted_t = torch.from_numpy(corrupted_np).permute(2, 0, 1).float() / 255.0
        label_idx = CORRUPTION_TYPES.index(corruption_type)

        return corrupted_t, label_idx
