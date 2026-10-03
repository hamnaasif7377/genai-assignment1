"""
Task 4: Paired photo-sketch Dataset for FS2K.
Applies IDENTICAL random augmentation (flip, crop) to both the photo
and its paired sketch, since independent augmentation would break the
pixel-level correspondence needed for paired image-to-image training.
"""
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

IMG_SIZE = 128


class FS2KPairedDataset(Dataset):
    def __init__(self, annotations, data_root, img_size=IMG_SIZE, augment=True):
        """
        annotations: list of dicts, each with 'image_name' (e.g. 'photo1/image0110')
                     and 'style' (0, 1, or 2).
        data_root: path to the FS2K folder (contains photo/ and sketch/ subfolders).
        """
        self.annotations = annotations
        self.data_root = data_root
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.annotations)

    def _load_pair(self, image_name):
        # image_name like "photo1/image0110" -> photo at photo/photo1/image0110.jpg
        # and sketch at sketch/sketch1/image0110.jpg (folder number matches)
        folder_num = image_name.split("/")[0][-1]  # "1", "2", or "3"
        base_name = image_name.split("/")[1]

        photo_path = os.path.join(self.data_root, "photo", f"photo{folder_num}", base_name + ".jpg")
        sketch_path = os.path.join(self.data_root, "sketch", f"sketch{folder_num}", base_name + ".jpg")

        # Some FS2K files use .png for sketches; fall back if .jpg is missing
        if not os.path.exists(sketch_path):
            sketch_path = sketch_path.replace(".jpg", ".png")
        if not os.path.exists(photo_path):
            photo_path = photo_path.replace(".jpg", ".png")

        photo = Image.open(photo_path).convert("RGB")
        sketch = Image.open(sketch_path).convert("RGB")
        return photo, sketch

    def __getitem__(self, idx):
        entry = self.annotations[idx]
        photo, sketch = self._load_pair(entry["image_name"])

        # Resize both to a bit larger than target, for random-crop augmentation
        resize_dim = int(self.img_size * 1.12) if self.augment else self.img_size
        photo = photo.resize((resize_dim, resize_dim), Image.BILINEAR)
        sketch = sketch.resize((resize_dim, resize_dim), Image.BILINEAR)

        if self.augment:
            # Identical random crop for both images
            i, j, h, w = self._get_random_crop_params(resize_dim, self.img_size)
            photo = TF.crop(photo, i, j, h, w)
            sketch = TF.crop(sketch, i, j, h, w)

            # Identical random horizontal flip for both images
            if random.random() > 0.5:
                photo = TF.hflip(photo)
                sketch = TF.hflip(sketch)
        else:
            photo = photo.resize((self.img_size, self.img_size), Image.BILINEAR)
            sketch = sketch.resize((self.img_size, self.img_size), Image.BILINEAR)

        photo_t = TF.to_tensor(photo)     # [0, 1] range, shape [3, H, W]
        sketch_t = TF.to_tensor(sketch)

        # Normalize to [-1, 1] for GAN training (standard practice, matches Tanh generator output)
        photo_t = photo_t * 2 - 1
        sketch_t = sketch_t * 2 - 1

        style = entry["style"]  # 0, 1, or 2

        return photo_t, sketch_t, style

    @staticmethod
    def _get_random_crop_params(resize_dim, crop_size):
        max_offset = resize_dim - crop_size
        i = random.randint(0, max_offset)
        j = random.randint(0, max_offset)
        return i, j, crop_size, crop_size
