"""
Corruption functions for Oxford-IIIT Pet restoration tasks (1-3).
Matches the exact parameter ranges specified in the assignment.
"""
import numpy as np
import cv2

CORRUPTION_TYPES = ["clean", "salt_pepper", "blur", "occlusion"]
IMG_SIZE = 128


def apply_salt_pepper(img_np, rng):
    """img_np: HxWx3 uint8 RGB. Returns corrupted copy."""
    prob = rng.uniform(0.02, 0.15)
    out = img_np.copy()
    mask = rng.random(img_np.shape[:2]) < prob
    salt_mask = rng.random(img_np.shape[:2]) < 0.5
    out[mask & salt_mask] = 255
    out[mask & ~salt_mask] = 0
    return out


def apply_gaussian_blur(img_np, rng):
    kernel = rng.choice([3, 5, 7])
    sigma = rng.uniform(0.5, 2.5)
    return cv2.GaussianBlur(img_np, (kernel, kernel), sigmaX=sigma)


def apply_occlusion(img_np, rng):
    out = img_np.copy()
    h, w = img_np.shape[:2]
    n_rects = rng.integers(1, 4)  # 1 to 3 inclusive
    target_area_frac = rng.uniform(0.10, 0.35)
    target_area = target_area_frac * h * w
    area_per_rect = target_area / n_rects
    for _ in range(n_rects):
        aspect = rng.uniform(0.5, 2.0)
        rw = int(np.sqrt(area_per_rect * aspect))
        rh = int(np.sqrt(area_per_rect / aspect))
        rw = min(rw, w)
        rh = min(rh, h)
        x0 = rng.integers(0, max(1, w - rw))
        y0 = rng.integers(0, max(1, h - rh))
        out[y0:y0 + rh, x0:x0 + rw] = 0
    return out


def corrupt_image(img_np, corruption_type, rng):
    if corruption_type == "clean":
        return img_np.copy()
    elif corruption_type == "salt_pepper":
        return apply_salt_pepper(img_np, rng)
    elif corruption_type == "blur":
        return apply_gaussian_blur(img_np, rng)
    elif corruption_type == "occlusion":
        return apply_occlusion(img_np, rng)
    else:
        raise ValueError(corruption_type)
