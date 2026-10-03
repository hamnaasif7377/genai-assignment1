"""
Task 2 evaluation: hard-routed restoration system.
Tests in TWO modes:
  - oracle: uses the known corruption label from the test manifest to
    select the specialist (shows specialist restoration ability alone).
  - predicted: uses the trained classifier's own prediction to select
    the specialist (shows the full operational system's performance).
Also identifies cases where a classifier misprediction caused the
wrong specialist to be used, and reports how that affected restoration.
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

from src.data.corruptions import CORRUPTION_TYPES
from src.data.datasets import ManifestDataset, manifest_collate_fn
from src.data.manifests import load_manifest
from src.models.task1_autoencoder import UniversalAutoencoder
from src.models.task2_classifier import CorruptionClassifier
from src.models.task2_specialists import SpecialistAutoencoder


def compute_metrics(pred, target):
    l1 = F.l1_loss(pred, target, reduction="none").mean(dim=[1, 2, 3])
    mse = F.mse_loss(pred, target, reduction="none").mean(dim=[1, 2, 3])
    psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
    ssim_val = ssim(pred, target, data_range=1.0, size_average=False)
    return l1, psnr, ssim_val


def load_models(args):
    with open(args.specialists_best_params, encoding="utf-8-sig") as f:
        spec_params = json.load(f)
    with open(args.classifier_best_params, encoding="utf-8-sig") as f:
        clf_params = json.load(f)

    classifier = CorruptionClassifier(base_ch=clf_params["base_ch"], dropout=clf_params["dropout"]).to(args.device)
    classifier.load_state_dict(torch.load(f"{args.checkpoint_root}/task2_classifier_best.pt", map_location=args.device))
    classifier.eval()

    specialists = {}
    for ctype in ["salt_pepper", "blur", "occlusion"]:
        model = SpecialistAutoencoder(
            base_ch=spec_params["base_ch"],
            bottleneck_dim=spec_params["bottleneck_dim"],
            dropout=spec_params["dropout"],
        ).to(args.device)
        model.load_state_dict(torch.load(
            f"{args.checkpoint_root}/task2_specialist_{ctype}_best.pt", map_location=args.device
        ))
        model.eval()
        specialists[ctype] = model

    return classifier, specialists


def restore(corrupted, corruption_type, specialists):
    """Identity for clean, otherwise route to the matching specialist."""
    if corruption_type == "clean":
        return corrupted
    return specialists[corruption_type](corrupted)


def run_evaluation(args):
    classifier, specialists = load_models(args)

    test_dataset = OxfordIIITPet(root=args.data_root, split="test", download=True)
    test_manifest = load_manifest(args.test_manifest)
    test_manifest_dataset = ManifestDataset(test_dataset, test_manifest)
    test_loader = DataLoader(test_manifest_dataset, batch_size=64, shuffle=False,
                              num_workers=2, collate_fn=manifest_collate_fn)

    oracle_results, predicted_results = {}, {}
    misroute_cases = []  # cases where classifier prediction != true label
    global_idx = 0

    with torch.no_grad():
        for corrupted, clean, label, entries in test_loader:
            corrupted, clean = corrupted.to(args.device), clean.to(args.device)

            # --- Oracle routing: use true label from manifest ---
            oracle_preds = torch.zeros_like(clean)
            for i, entry in enumerate(entries):
                true_type = entry["corruption_type"]
                single_input = corrupted[i:i+1]
                oracle_preds[i:i+1] = restore(single_input, true_type, specialists)

            oracle_l1, oracle_psnr, oracle_ssim = compute_metrics(oracle_preds, clean)

            # --- Predicted routing: use classifier's own prediction ---
            clf_logits = classifier(corrupted)
            clf_preds = clf_logits.argmax(dim=1).cpu().numpy()

            predicted_preds = torch.zeros_like(clean)
            for i, entry in enumerate(entries):
                predicted_type = CORRUPTION_TYPES[clf_preds[i]]
                single_input = corrupted[i:i+1]
                predicted_preds[i:i+1] = restore(single_input, predicted_type, specialists)

            predicted_l1, predicted_psnr, predicted_ssim = compute_metrics(predicted_preds, clean)

            for i, entry in enumerate(entries):
                true_type = entry["corruption_type"]
                severity = entry.get("severity_level")
                key = (true_type, severity)

                oracle_results.setdefault(key, []).append({
                    "l1": oracle_l1[i].item(), "psnr": oracle_psnr[i].item(), "ssim": oracle_ssim[i].item(),
                })
                predicted_results.setdefault(key, []).append({
                    "l1": predicted_l1[i].item(), "psnr": predicted_psnr[i].item(), "ssim": predicted_ssim[i].item(),
                })

                predicted_type = CORRUPTION_TYPES[clf_preds[i]]
                if predicted_type != true_type:
                    misroute_cases.append({
                        "global_idx": global_idx, "true_type": true_type, "predicted_type": predicted_type,
                        "severity_level": severity,
                        "oracle_ssim": oracle_ssim[i].item(), "predicted_ssim": predicted_ssim[i].item(),
                        "ssim_drop": oracle_ssim[i].item() - predicted_ssim[i].item(),
                    })
                global_idx += 1

    def print_and_summarize(results, title):
        print(f"\n=== {title} ===")
        print(f"{'Corruption':<15} {'Severity':<10} {'N':<8} {'L1':<10} {'PSNR':<10} {'SSIM':<10}")
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
        return summary

    oracle_summary = print_and_summarize(oracle_results, "ORACLE ROUTING (true label used)")
    predicted_summary = print_and_summarize(predicted_results, "PREDICTED ROUTING (classifier's own prediction)")

    print(f"\n=== Misrouting Analysis ===")
    print(f"Total misrouted cases: {len(misroute_cases)} / {global_idx} ({100*len(misroute_cases)/global_idx:.2f}%)")
    if misroute_cases:
        avg_ssim_drop = np.mean([c["ssim_drop"] for c in misroute_cases])
        print(f"Average SSIM drop on misrouted cases: {avg_ssim_drop:.4f}")
        worst_misroutes = sorted(misroute_cases, key=lambda c: -c["ssim_drop"])[:10]
        print("\nTop 10 worst misrouting cases (largest SSIM drop from misrouting):")
        for c in worst_misroutes:
            print(f"  idx={c['global_idx']}  true={c['true_type']}  predicted={c['predicted_type']}  "
                  f"severity={c['severity_level']}  oracle_ssim={c['oracle_ssim']:.4f}  "
                  f"predicted_ssim={c['predicted_ssim']:.4f}  drop={c['ssim_drop']:.4f}")

    os.makedirs(os.path.dirname(args.results_out), exist_ok=True)
    with open(args.results_out, "w") as f:
        json.dump({
            "oracle_routing": oracle_summary,
            "predicted_routing": predicted_summary,
            "total_misrouted": len(misroute_cases),
            "total_evaluated": global_idx,
            "misrouting_rate": len(misroute_cases) / global_idx,
        }, f, indent=2)
    print(f"\nSaved summary to {args.results_out}")

    with open(args.misroute_out, "w") as f:
        json.dump(sorted(misroute_cases, key=lambda c: -c["ssim_drop"])[:20], f, indent=2)
    print(f"Saved top misrouting cases to {args.misroute_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/content/data")
    parser.add_argument("--test_manifest", default="/content/drive/MyDrive/GenAI_A1/manifests/test_manifest.json")
    parser.add_argument("--checkpoint_root", default="/content/drive/MyDrive/GenAI_A1/checkpoints")
    parser.add_argument("--classifier_best_params", default="configs/task2_classifier_best_params.json")
    parser.add_argument("--specialists_best_params", default="configs/task2_specialists_best_params.json")
    parser.add_argument("--results_out", default="/content/drive/MyDrive/GenAI_A1/reports/task2_hardrouting_results.json")
    parser.add_argument("--misroute_out", default="/content/drive/MyDrive/GenAI_A1/reports/task2_misroute_cases.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    run_evaluation(args)
