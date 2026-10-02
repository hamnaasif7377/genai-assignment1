"""
Task 1 evaluation: runs the trained model against the test manifest,
reports metrics broken down by corruption type AND severity level,
and saves visual examples (including failure cases) for the report.
"""
import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import OxfordIIITPet
from pytorch_msssim import ssim
import torch.nn.functional as F

from src.data.datasets import ManifestDataset, manifest_collate_fn
from src.data.manifests import load_manifest
from src.models.task1_autoencoder import UniversalAutoencoder


def compute_metrics(pred, target):
    l1 = F.l1_loss(pred, target, reduction="none").mean(dim=[1, 2, 3])
    mse = F.mse_loss(pred, target, reduction="none").mean(dim=[1, 2, 3])
    psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
    ssim_val = ssim(pred, target, data_range=1.0, size_average=False)
    return l1, psnr, ssim_val


def run_evaluation(args):
    with open(args.best_params, encoding="utf-8-sig") as f:
        best_params = json.load(f)

    test_dataset = OxfordIIITPet(root=args.data_root, split="test", download=True)
    test_manifest = load_manifest(args.test_manifest)
    test_manifest_dataset = ManifestDataset(test_dataset, test_manifest)
    test_loader = DataLoader(test_manifest_dataset, batch_size=64, shuffle=False,
                              num_workers=2, collate_fn=manifest_collate_fn)

    model = UniversalAutoencoder(
        base_ch=best_params["base_ch"],
        bottleneck_dim=best_params["bottleneck_dim"],
        dropout=best_params["dropout"],
    ).to(args.device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    model.eval()

    results = {}
    all_records = []

    global_idx = 0
    with torch.no_grad():
        for corrupted, clean, label, entries in test_loader:
            corrupted, clean = corrupted.to(args.device), clean.to(args.device)
            pred = model(corrupted)
            l1, psnr, ssim_val = compute_metrics(pred, clean)

            for i, entry in enumerate(entries):
                key = (entry["corruption_type"], entry.get("severity_level"))
                results.setdefault(key, []).append({
                    "l1": l1[i].item(), "psnr": psnr[i].item(), "ssim": ssim_val[i].item(),
                })
                all_records.append({
                    "global_idx": global_idx, "corruption_type": entry["corruption_type"],
                    "severity_level": entry.get("severity_level"),
                    "l1": l1[i].item(), "psnr": psnr[i].item(), "ssim": ssim_val[i].item(),
                })
                global_idx += 1

    print(f"\n{'Corruption':<15} {'Severity':<10} {'N':<8} {'L1':<10} {'PSNR':<10} {'SSIM':<10}")
    print("-" * 65)
    summary = {}
    for (ctype, severity), records in sorted(results.items(), key=lambda x: (x[0][0], (x[0][1] is None, x[0][1]))):
        n = len(records)
        mean_l1 = np.mean([r["l1"] for r in records])
        mean_psnr = np.mean([r["psnr"] for r in records])
        mean_ssim = np.mean([r["ssim"] for r in records])
        sev_label = "N/A" if severity is None else f"level_{severity}"
        print(f"{ctype:<15} {sev_label:<10} {n:<8} {mean_l1:<10.4f} {mean_psnr:<10.2f} {mean_ssim:<10.4f}")
        summary[f"{ctype}_{sev_label}"] = {
            "n": n, "mean_l1": float(mean_l1), "mean_psnr": float(mean_psnr), "mean_ssim": float(mean_ssim)
        }

    os.makedirs(os.path.dirname(args.results_out), exist_ok=True)
    with open(args.results_out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary table to {args.results_out}")

    all_records_sorted = sorted(all_records, key=lambda r: r["ssim"])
    worst_cases = all_records_sorted[:args.n_failure_cases]
    print(f"\nWorst {args.n_failure_cases} cases by SSIM:")
    for case in worst_cases:
        print(f"  idx={case['global_idx']}  type={case['corruption_type']}  "
              f"severity={case['severity_level']}  ssim={case['ssim']:.4f}  psnr={case['psnr']:.2f}")

    with open(args.failure_cases_out, "w") as f:
        json.dump(worst_cases, f, indent=2)
    print(f"Saved failure case indices to {args.failure_cases_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/content/data")
    parser.add_argument("--test_manifest", default="/content/drive/MyDrive/GenAI_A1/manifests/test_manifest.json")
    parser.add_argument("--checkpoint", default="/content/drive/MyDrive/GenAI_A1/checkpoints/task1_final_best.pt")
    parser.add_argument("--best_params", default="configs/task1_best_params.json")
    parser.add_argument("--results_out", default="/content/drive/MyDrive/GenAI_A1/reports/task1_results.json")
    parser.add_argument("--failure_cases_out", default="/content/drive/MyDrive/GenAI_A1/reports/task1_failure_cases.json")
    parser.add_argument("--n_failure_cases", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    run_evaluation(args)
