"""
Dataset that applies ONE fixed corruption type to every image (for
training individual Task 2 specialist autoencoders), rather than
randomly picking among all four types.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.corruptions import IMG_SIZE, corrupt_image


class SingleCorruptionDataset(Dataset):
    def __init__(self, base_dataset, indices, corruption_type, img_size=IMG_SIZE):
        self.base_dataset = base_dataset
        self.indices = indices
        self.corruption_type = corruption_type
        self.img_size = img_size

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        pil_img, _ = self.base_dataset[real_idx]
        pil_img = pil_img.convert("RGB").resize((self.img_size, self.img_size))
        clean_np = np.array(pil_img)

        rng = np.random.default_rng()
        corrupted_np = corrupt_image(clean_np, self.corruption_type, rng)

        clean_t = torch.from_numpy(clean_np).permute(2, 0, 1).float() / 255.0
        corrupted_t = torch.from_numpy(corrupted_np).permute(2, 0, 1).float() / 255.0

        return corrupted_t, clean_t
