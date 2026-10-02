"""
Deterministic manifest generation for validation and test sets.
Validation: random corruption type + severity per image (seeded).
Test: fixed severity levels (3 per corruption type) per the assignment spec.
"""
import json
import os
import numpy as np

from src.data.corruptions import CORRUPTION_TYPES, IMG_SIZE

SALT_PEPPER_SEVERITIES = [0.03, 0.08, 0.15]
BLUR_SEVERITIES = [(3, 0.7), (5, 1.5), (7, 2.5)]  # (kernel, sigma)
OCCLUSION_SEVERITIES = [
    (0.10, 1),  # ~10% area, 1 rectangle
    (0.20, 2),  # ~20% area, 2 rectangles
    (0.35, 3),  # ~35% area, 3 rectangles
]


def generate_manifest_entry(rng, corruption_type):
    """Generates one deterministic corruption spec (not the actual corrupted image yet)."""
    entry = {"corruption_type": corruption_type}
    if corruption_type == "salt_pepper":
        entry["prob"] = float(rng.uniform(0.02, 0.15))
    elif corruption_type == "blur":
        entry["kernel"] = int(rng.choice([3, 5, 7]))
        entry["sigma"] = float(rng.uniform(0.5, 2.5))
    elif corruption_type == "occlusion":
        h, w = IMG_SIZE, IMG_SIZE
        n_rects = int(rng.integers(1, 4))
        target_area_frac = float(rng.uniform(0.10, 0.35))
        area_per_rect = target_area_frac * h * w / n_rects
        rects = []
        for _ in range(n_rects):
            aspect = float(rng.uniform(0.5, 2.0))
            rw = int(np.sqrt(area_per_rect * aspect))
            rw = min(rw, w)
            rh = int(np.sqrt(area_per_rect / aspect))
            rh = min(rh, h)
            x0 = int(rng.integers(0, max(1, w - rw)))
            y0 = int(rng.integers(0, max(1, h - rh)))
            rects.append({"x0": x0, "y0": y0, "w": rw, "h": rh})
        entry["rectangles"] = rects
        entry["target_area_frac"] = target_area_frac
    return entry


def build_manifest(indices, seed):
    """Validation manifest: one fixed random corruption assignment per image."""
    rng = np.random.default_rng(seed)
    manifest = []
    for idx in indices:
        corruption_type = str(rng.choice(CORRUPTION_TYPES))
        entry = generate_manifest_entry(rng, corruption_type)
        entry["index"] = int(idx)
        entry["seed"] = seed
        manifest.append(entry)
    return manifest


def build_test_manifest(indices, seed):
    """
    Test manifest: EVERY image gets all 3 corruption types x 3 fixed
    severities (9 corrupted versions) PLUS 1 clean version = 10 entries/image.
    """
    rng = np.random.default_rng(seed)
    manifest = []

    for idx in indices:
        manifest.append({
            "index": int(idx), "corruption_type": "clean",
            "severity_level": None, "seed": seed
        })

        for level, prob in enumerate(SALT_PEPPER_SEVERITIES):
            manifest.append({
                "index": int(idx), "corruption_type": "salt_pepper",
                "severity_level": level, "prob": prob, "seed": seed
            })

        for level, (kernel, sigma) in enumerate(BLUR_SEVERITIES):
            manifest.append({
                "index": int(idx), "corruption_type": "blur",
                "severity_level": level, "kernel": kernel, "sigma": sigma, "seed": seed
            })

        for level, (area_frac, n_rects) in enumerate(OCCLUSION_SEVERITIES):
            h, w = IMG_SIZE, IMG_SIZE
            area_per_rect = area_frac * h * w / n_rects
            rects = []
            for _ in range(n_rects):
                aspect = float(rng.uniform(0.5, 2.0))
                rw = int(np.sqrt(area_per_rect * aspect))
                rw = min(rw, w)
                rh = int(np.sqrt(area_per_rect / aspect))
                rh = min(rh, h)
                x0 = int(rng.integers(0, max(1, w - rw)))
                y0 = int(rng.integers(0, max(1, h - rh)))
                rects.append({"x0": x0, "y0": y0, "w": rw, "h": rh})
            manifest.append({
                "index": int(idx), "corruption_type": "occlusion",
                "severity_level": level, "target_area_frac": area_frac,
                "rectangles": rects, "seed": seed
            })

    return manifest


def save_manifest(manifest, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


def load_manifest(path):
    with open(path, "r") as f:
        return json.load(f)
