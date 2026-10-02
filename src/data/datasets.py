"""
PyTorch Dataset classes for Pet restoration (Tasks 1-3).
- PetRestorationDataset: dynamic runtime corruption, for TRAINING.
- ManifestDataset: reads a saved manifest, reproduces EXACT corrupted
  images deterministically, for VALIDATION/TEST.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.corruptions import CORRUPTION_TYPES, IMG_SIZE, corrupt_image


class PetRestorationDataset(Dataset):
    """Applies a random corruption type dynamically every __getitem__ call."""

    def __init__(self, base_dataset, indices, img_size=IMG_SIZE):
        self.base_dataset = base_dataset
        self.indices = indices
        self.img_size = img_size

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        pil_img, _ = self.base_dataset[real_idx]
        pil_img = pil_img.convert("RGB").resize((self.img_size, self.img_size))
        clean_np = np.array(pil_img)

        rng = np.random.default_rng()
        corruption_type = rng.choice(CORRUPTION_TYPES)
        corrupted_np = corrupt_image(clean_np, corruption_type, rng)

        clean_t = torch.from_numpy(clean_np).permute(2, 0, 1).float() / 255.0
        corrupted_t = torch.from_numpy(corrupted_np).permute(2, 0, 1).float() / 255.0
        label_idx = CORRUPTION_TYPES.index(corruption_type)

        return corrupted_t, clean_t, label_idx


class ManifestDataset(Dataset):
    """Reads a saved manifest, deterministically reproduces the same
    corrupted image every time (no fresh RNG calls at read time)."""

    def __init__(self, base_dataset, manifest, img_size=IMG_SIZE):
        self.base_dataset = base_dataset
        self.manifest = manifest
        self.img_size = img_size

    def __len__(self):
        return len(self.manifest)

    def _apply_from_entry(self, img_np, entry):
        ctype = entry["corruption_type"]
        if ctype == "clean":
            return img_np.copy()

        elif ctype == "salt_pepper":
            rng = np.random.default_rng(entry["seed"] + entry["index"])
            prob = entry["prob"]
            out = img_np.copy()
            mask = rng.random(img_np.shape[:2]) < prob
            salt_mask = rng.random(img_np.shape[:2]) < 0.5
            out[mask & salt_mask] = 255
            out[mask & ~salt_mask] = 0
            return out

        elif ctype == "blur":
            import cv2
            return cv2.GaussianBlur(img_np, (entry["kernel"], entry["kernel"]), sigmaX=entry["sigma"])

        elif ctype == "occlusion":
            out = img_np.copy()
            for r in entry["rectangles"]:
                out[r["y0"]:r["y0"] + r["h"], r["x0"]:r["x0"] + r["w"]] = 0
            return out

        else:
            raise ValueError(ctype)

    def __getitem__(self, idx):
        entry = self.manifest[idx]
        pil_img, _ = self.base_dataset[entry["index"]]
        pil_img = pil_img.convert("RGB").resize((self.img_size, self.img_size))
        clean_np = np.array(pil_img)

        corrupted_np = self._apply_from_entry(clean_np, entry)

        clean_t = torch.from_numpy(clean_np).permute(2, 0, 1).float() / 255.0
        corrupted_t = torch.from_numpy(corrupted_np).permute(2, 0, 1).float() / 255.0
        label_idx = CORRUPTION_TYPES.index(entry["corruption_type"])

        return corrupted_t, clean_t, label_idx, entry


def manifest_collate_fn(batch):
    """Custom collate: entries have different keys per corruption type,
    so keep them as a plain list instead of letting default_collate merge them."""
    corrupted = torch.stack([item[0] for item in batch])
    clean = torch.stack([item[1] for item in batch])
    labels = torch.tensor([item[2] for item in batch])
    entries = [item[3] for item in batch]
    return corrupted, clean, labels, entries
