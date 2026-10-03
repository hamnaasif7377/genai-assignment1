"""
Task 3 evaluation: tests the soft MoE system on the test manifest,
reports reconstruction metrics by corruption type/severity, AND
analyzes gating behavior as the assignment requires: average expert
weights per true corruption type/severity, routing distribution
examples, and detection of inactive or dominating experts.
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim
from torch.utils.data import DataLoader
from torchvision.datasets import OxfordIIITPet

from src.data.datasets import ManifestDataset, manifest_collate_fn
from src.data.manifests import load_manifest
from src.models.task3_soft_moe import SoftMoERestoration

BRANCH_NAMES = ["clean", "salt_pepper", "blur", "occlusion"]


def compute_metrics(pred, target):
    l1 = F.l1_loss(pred, target, reduction="none").mean(dim=[1, 2, 3])
    mse = F.mse_loss(pred, target, reduction="none").mean(dim=[1, 2, 3])
    psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
    ssim_val = ssim(pred, target, data_range=1.0, size_average=False)
    return l1, psnr, ssim_val


def run_evaluation(args):
    with open(args.classifier_best_params, encoding="utf-8-sig") as f:
        clf_params = json.load(f)
    with open(args.specialists_best_params, encoding="utf-8-sig") as f:
        spec_params = json.load(f)
    with open(args.task3_best_params, encoding="utf-8-sig") as f:
        moe_params = json.load(f)

    model = SoftMoERestoration(
        clf_base_ch=clf_params["base_ch"], clf_dropout=clf_params["dropout"],
        spec_base_ch=spec_params["base_ch"], spec_bottleneck_dim=spec_params["bottleneck_dim"],
        spec_dropout=spec_params["dropout"], temperature=moe_params["temperature"],
    ).to(args.device)
    model.load_state_dict(torch.load(f"{args.checkpoint_root}/task3_soft_moe_best.pt", map_location=args.device))
    model.eval()

    test_dataset = OxfordIIITPet(root=args.data_root, split="test", download=True)
    test_manifest = load_manifest(args.test_manifest)
    test_manifest_dataset = ManifestDataset(test_dataset, test_manifest)
    test_loader = DataLoader(test_manifest_dataset, batch_size=64, shuffle=False,
                              num_workers=2, collate_fn=manifest_collate_fn)

    results = {}
    weight_records = []  # per-sample routing weights, for analysis

    with torch.no_grad():
        for corrupted, clean, label, entries in test_loader:
            corrupted, clean = corrupted.to(args.device), clean.to(args.device)
            pred, weights, gate_logits = model(corrupted, return_weights=True)
            l1, psnr, ssim_val = compute_metrics(pred, clean)
            weights_np = weights.cpu().numpy()

            for i, entry in enumerate(entries):
                key = (entry["corruption_type"], entry.get("severity_level"))
                results.setdefault(key, []).append({
                    "l1": l1[i].item(), "psnr": psnr[i].item(), "ssim": ssim_val[i].item(),
                    "weights": weights_np[i].tolist(),
                })
                weight_records.append({
                    "true_type": entry["corruption_type"], "severity_level": entry.get("severity_level"),
                    "weights": weights_np[i].tolist(),
                })

    # --- Reconstruction quality table ---
    print(f"\n{'Corruption':<15} {'Severity':<10} {'N':<8} {'L1':<10} {'PSNR':<10} {'SSIM':<10}")
    print("-" * 65)
    summary = {}
    for (ctype, severity), records in sorted(results.items(), key=lambda x: (x[0][0], (x[0][1] is None, x[0][1]))):
        n = len(records)
        mean_l1 = np.mean([r["l1"] for r in records])
        mean_psnr = np.mean([r["psnr"] for r in records])
        mean_ssim = np.mean([r["ssim"] for r in records])
        mean_weights = np.mean([r["weights"] for r in records], axis=0)
        sev_label = "N/A" if severity is None else f"level_{severity}"
        print(f"{ctype:<15} {sev_label:<10} {n:<8} {mean_l1:<10.4f} {mean_psnr:<10.2f} {mean_ssim:<10.4f}")
        summary[f"{ctype}_{sev_label}"] = {
            "n": n, "mean_l1": float(mean_l1), "mean_psnr": float(mean_psnr), "mean_ssim": float(mean_ssim),
            "mean_weights": {BRANCH_NAMES[j]: float(mean_weights[j]) for j in range(4)},
        }

    # --- Routing weight analysis (required by assignment) ---
    print(f"\n{'Corruption':<15} {'Severity':<10} " + "  ".join(f"{n:<10}" for n in BRANCH_NAMES))
    print("-" * 85)
    for (ctype, severity), records in sorted(results.items(), key=lambda x: (x[0][0], (x[0][1] is None, x[0][1]))):
        mean_weights = np.mean([r["weights"] for r in records], axis=0)
        sev_label = "N/A" if severity is None else f"level_{severity}"
        weight_str = "  ".join(f"{w:<10.4f}" for w in mean_weights)
        print(f"{ctype:<15} {sev_label:<10} {weight_str}")

    # --- Check for inactive or dominating experts ---
    overall_mean_weights = np.mean([r["weights"] for r in weight_records], axis=0)
    print(f"\nOverall average routing weights: "
          f"{dict(zip(BRANCH_NAMES, [round(float(w), 4) for w in overall_mean_weights]))}")

    inactive_threshold = 0.05
    dominant_threshold = 0.7
    for i, name in enumerate(BRANCH_NAMES):
        if overall_mean_weights[i] < inactive_threshold:
            print(f"  WARNING: '{name}' branch appears INACTIVE (avg weight {overall_mean_weights[i]:.4f})")
    # Check if any expert dominates on inputs it shouldn't (e.g. blur expert on clean images)
    for (ctype, severity), records in results.items():
        mean_weights = np.mean([r["weights"] for r in records], axis=0)
        for i, name in enumerate(BRANCH_NAMES):
            if name != ctype and mean_weights[i] > dominant_threshold:
                print(f"  WARNING: '{name}' branch dominates (weight {mean_weights[i]:.4f}) "
                      f"on true type '{ctype}', which is unexpected")

    os.makedirs(os.path.dirname(args.results_out), exist_ok=True)
    with open(args.results_out, "w") as f:
        json.dump({
            "summary": summary,
            "overall_mean_weights": {BRANCH_NAMES[i]: float(overall_mean_weights[i]) for i in range(4)},
        }, f, indent=2)
    print(f"\nSaved summary to {args.results_out}")

    # Save a sample of individual routing weights for the heatmap visualization
    sample_for_heatmap = weight_records[::50]  # every 50th sample, keeps file small
    with open(args.weights_sample_out, "w") as f:
        json.dump(sample_for_heatmap, f, indent=2)
    print(f"Saved {len(sample_for_heatmap)} sample routing weights to {args.weights_sample_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/content/data")
    parser.add_argument("--test_manifest", default="/content/drive/MyDrive/GenAI_A1/manifests/test_manifest.json")
    parser.add_argument("--checkpoint_root", default="/content/drive/MyDrive/GenAI_A1/checkpoints")
    parser.add_argument("--classifier_best_params", default="configs/task2_classifier_best_params.json")
    parser.add_argument("--specialists_best_params", default="configs/task2_specialists_best_params.json")
    parser.add_argument("--task3_best_params", default="configs/task3_best_params.json")
    parser.add_argument("--results_out", default="/content/drive/MyDrive/GenAI_A1/reports/task3_results.json")
    parser.add_argument("--weights_sample_out", default="/content/drive/MyDrive/GenAI_A1/reports/task3_weights_sample.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    run_evaluation(args)
