"""
Task 3: Joint training of the soft mixture-of-experts system.
Warm-up stage: experts frozen, only the gate is trained.
Joint stage: experts unfrozen, everything fine-tuned together with a
combined reconstruction + classification + routing-balance loss.
Includes an Optuna search over the joint fine-tuning hyperparameters.
"""
import argparse
import json
import os
import random

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim
from torch.utils.data import DataLoader
from torchvision.datasets import OxfordIIITPet

from src.data.datasets import PetRestorationDataset, ManifestDataset, manifest_collate_fn
from src.data.manifests import load_manifest, build_manifest, save_manifest
from src.models.task3_soft_moe import SoftMoERestoration

SEED = 42
CORRUPTION_TYPES = ["clean", "salt_pepper", "blur", "occlusion"]


def get_splits(data_root):
    trainval = OxfordIIITPet(root=data_root, split="trainval", download=True)
    n_total = len(trainval)
    indices = list(range(n_total))
    rng = random.Random(SEED)
    rng.shuffle(indices)
    n_train = int(0.8 * n_total)
    return trainval, indices[:n_train], indices[n_train:]


def get_val_manifest(val_indices, manifest_root):
    path = f"{manifest_root}/val_manifest.json"
    if os.path.exists(path):
        return load_manifest(path)
    manifest = build_manifest(val_indices, SEED)
    save_manifest(manifest, path)
    return manifest


def balance_loss(weights):
    """Penalizes the gate for sending nearly every input to the same
    expert: average weight per branch, in a balanced batch, should be
    close to 1/4 each."""
    avg_weights = weights.mean(dim=0)  # [4]
    target = torch.full_like(avg_weights, 1.0 / 4)
    return F.mse_loss(avg_weights, target)


def joint_loss(pred, clean, gate_logits, true_labels, weights,
                w_l1, w_ssim, w_ce, w_balance, alpha_recon=0.8):
    l1 = F.l1_loss(pred, clean)
    ssim_val = ssim(pred, clean, data_range=1.0, size_average=True)
    recon_loss = alpha_recon * l1 + (1 - alpha_recon) * (1 - ssim_val)

    ce_loss = F.cross_entropy(gate_logits, true_labels)
    bal_loss = balance_loss(weights)

    total = w_l1 * recon_loss + w_ce * ce_loss + w_balance * bal_loss
    return total, recon_loss.item(), ce_loss.item(), bal_loss.item()


def build_model(args, clf_params, spec_params, temperature):
    model = SoftMoERestoration(
        clf_base_ch=clf_params["base_ch"], clf_dropout=clf_params["dropout"],
        spec_base_ch=spec_params["base_ch"], spec_bottleneck_dim=spec_params["bottleneck_dim"],
        spec_dropout=spec_params["dropout"], temperature=temperature,
    ).to(args.device)
    model.load_pretrained(
        classifier_ckpt=f"{args.checkpoint_root}/task2_classifier_best.pt",
        salt_pepper_ckpt=f"{args.checkpoint_root}/task2_specialist_salt_pepper_best.pt",
        blur_ckpt=f"{args.checkpoint_root}/task2_specialist_blur_best.pt",
        occlusion_ckpt=f"{args.checkpoint_root}/task2_specialist_occlusion_best.pt",
        device=args.device,
    )
    return model


def warmup_phase(model, train_loader, device, epochs, lr):
    """Freeze experts, train only the gate."""
    model.freeze_experts()
    optimizer = torch.optim.Adam(model.gate.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for corrupted, clean, label in train_loader:
            corrupted, clean, label = corrupted.to(device), clean.to(device), label.to(device)
            optimizer.zero_grad()
            pred, weights, gate_logits = model(corrupted, return_weights=True)
            loss = F.cross_entropy(gate_logits, label)  # warm-up: just learn to route correctly
            loss.backward()
            optimizer.step()
    model.unfreeze_experts()


@torch.no_grad()
def evaluate(model, val_loader, device, alpha_recon=0.8):
    model.eval()
    l1_sum, n = 0.0, 0
    for corrupted, clean, label, entries in val_loader:
        corrupted, clean = corrupted.to(device), clean.to(device)
        pred = model(corrupted)
        l1_sum += F.l1_loss(pred, clean, reduction="sum").item() / (3 * 128 * 128)
        n += corrupted.size(0)
    return l1_sum / n


def run_search(args):
    trainval, train_indices, val_indices = get_splits(args.data_root)
    val_manifest = get_val_manifest(val_indices, args.manifest_root)

    train_dataset = PetRestorationDataset(trainval, train_indices)
    val_manifest_dataset = ManifestDataset(trainval, val_manifest)

    with open(args.classifier_best_params, encoding="utf-8-sig") as f:
        clf_params = json.load(f)
    with open(args.specialists_best_params, encoding="utf-8-sig") as f:
        spec_params = json.load(f)

    os.makedirs(args.optuna_root, exist_ok=True)
    optuna_db = f"sqlite:///{args.optuna_root}/task3_soft_moe.db"

    def objective(trial):
        temperature = trial.suggest_float("temperature", 0.5, 3.0)
        lr_joint = trial.suggest_float("lr_joint", 1e-5, 1e-3, log=True)
        w_ce = trial.suggest_float("w_ce", 0.01, 0.5, log=True)
        w_balance = trial.suggest_float("w_balance", 0.001, 0.1, log=True)
        alpha_recon = trial.suggest_float("alpha_recon", 0.5, 0.95)

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_manifest_dataset, batch_size=16, shuffle=False,
                                 num_workers=2, collate_fn=manifest_collate_fn)

        model = build_model(args, clf_params, spec_params, temperature)
        warmup_phase(model, train_loader, args.device, epochs=args.warmup_epochs, lr=1e-4)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr_joint)

        val_l1 = None
        for epoch in range(args.trial_epochs):
            model.train()
            for corrupted, clean, label in train_loader:
                corrupted, clean, label = corrupted.to(args.device), clean.to(args.device), label.to(args.device)
                optimizer.zero_grad()
                pred, weights, gate_logits = model(corrupted, return_weights=True)
                loss, _, _, _ = joint_loss(pred, clean, gate_logits, label, weights,
                                            w_l1=1.0, w_ssim=1.0, w_ce=w_ce, w_balance=w_balance,
                                            alpha_recon=alpha_recon)
                loss.backward()
                optimizer.step()

            val_l1 = evaluate(model, val_loader, args.device)
            trial.report(val_l1, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return val_l1

    study = optuna.create_study(
        study_name="task3_soft_moe", storage=optuna_db,
        direction="minimize", load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(objective, n_trials=args.n_trials)

    print("Best trial:", study.best_trial.params)
    print("Best val_l1:", study.best_value)

    os.makedirs(os.path.dirname(args.best_params_out), exist_ok=True)
    with open(args.best_params_out, "w") as f:
        json.dump({**study.best_trial.params, "optuna_best_val_l1": study.best_value}, f, indent=2)


def run_final(args):
    with open(args.best_params_out, encoding="utf-8-sig") as f:
        best_params = json.load(f)
    with open(args.classifier_best_params, encoding="utf-8-sig") as f:
        clf_params = json.load(f)
    with open(args.specialists_best_params, encoding="utf-8-sig") as f:
        spec_params = json.load(f)

    trainval, train_indices, val_indices = get_splits(args.data_root)
    val_manifest = get_val_manifest(val_indices, args.manifest_root)

    train_dataset = PetRestorationDataset(trainval, train_indices)
    val_manifest_dataset = ManifestDataset(trainval, val_manifest)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_manifest_dataset, batch_size=16, shuffle=False,
                             num_workers=2, collate_fn=manifest_collate_fn)

    model = build_model(args, clf_params, spec_params, best_params["temperature"])

    print("Warm-up phase: training gate only, experts frozen...")
    warmup_phase(model, train_loader, args.device, epochs=args.warmup_epochs, lr=1e-4)
    print("Warm-up complete.")

    optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr_joint"])

    os.makedirs(args.checkpoint_root, exist_ok=True)
    best_ckpt = f"{args.checkpoint_root}/task3_soft_moe_best.pt"
    best_val_l1 = float("inf")

    for epoch in range(args.epochs):
        model.train()
        recon_sum, ce_sum, bal_sum = 0.0, 0.0, 0.0
        for corrupted, clean, label in train_loader:
            corrupted, clean, label = corrupted.to(args.device), clean.to(args.device), label.to(args.device)
            optimizer.zero_grad()
            pred, weights, gate_logits = model(corrupted, return_weights=True)
            loss, recon_l, ce_l, bal_l = joint_loss(
                pred, clean, gate_logits, label, weights,
                w_l1=1.0, w_ssim=1.0, w_ce=best_params["w_ce"], w_balance=best_params["w_balance"],
                alpha_recon=best_params["alpha_recon"]
            )
            loss.backward()
            optimizer.step()
            recon_sum += recon_l * corrupted.size(0)
            ce_sum += ce_l * corrupted.size(0)
            bal_sum += bal_l * corrupted.size(0)

        n = len(train_dataset)
        val_l1 = evaluate(model, val_loader, args.device)

        print(f"Epoch {epoch+1}/{args.epochs}  recon={recon_sum/n:.4f}  "
              f"ce={ce_sum/n:.4f}  balance={bal_sum/n:.4f}  val_l1={val_l1:.4f}")

        if val_l1 < best_val_l1:
            best_val_l1 = val_l1
            torch.save(model.state_dict(), best_ckpt)
            print(f"  -> New best! Saved to {best_ckpt}")

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"{args.checkpoint_root}/task3_soft_moe_epoch{epoch+1}.pt")

    print(f"\nTraining complete. Best val_l1: {best_val_l1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["search", "final"], required=True)
    parser.add_argument("--data_root", default="/content/data")
    parser.add_argument("--manifest_root", default="/content/drive/MyDrive/GenAI_A1/manifests")
    parser.add_argument("--optuna_root", default="/content/drive/MyDrive/GenAI_A1/optuna")
    parser.add_argument("--checkpoint_root", default="/content/drive/MyDrive/GenAI_A1/checkpoints")
    parser.add_argument("--classifier_best_params", default="configs/task2_classifier_best_params.json")
    parser.add_argument("--specialists_best_params", default="configs/task2_specialists_best_params.json")
    parser.add_argument("--best_params_out", default="configs/task3_best_params.json")
    parser.add_argument("--n_trials", type=int, default=15)
    parser.add_argument("--trial_epochs", type=int, default=3)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.mode == "search":
        run_search(args)
    else:
        run_final(args)
