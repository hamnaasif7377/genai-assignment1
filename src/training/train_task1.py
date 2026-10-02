"""
Task 1 training script: Optuna hyperparameter search + final training run.

Usage (from repo root, with data already downloaded to DATA_ROOT):
    python -m src.training.train_task1 --mode search --n_trials 30
    python -m src.training.train_task1 --mode final --epochs 60
"""
import argparse
import json
import os
import random

import numpy as np
import optuna
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import OxfordIIITPet

from src.data.corruptions import IMG_SIZE
from src.data.datasets import PetRestorationDataset, ManifestDataset, manifest_collate_fn
from src.data.manifests import build_manifest, load_manifest, save_manifest
from src.models.task1_autoencoder import UniversalAutoencoder
from src.training.losses import combined_loss

SEED = 42


def get_splits(data_root):
    trainval = OxfordIIITPet(root=data_root, split="trainval", download=True)
    n_total = len(trainval)
    indices = list(range(n_total))
    rng = random.Random(SEED)
    rng.shuffle(indices)
    n_train = int(0.8 * n_total)
    return trainval, indices[:n_train], indices[n_train:]


def get_val_manifest(trainval, val_indices, manifest_root):
    manifest_path = f"{manifest_root}/val_manifest.json"
    if os.path.exists(manifest_path):
        return load_manifest(manifest_path)
    manifest = build_manifest(val_indices, SEED)
    save_manifest(manifest, manifest_path)
    return manifest


def run_search(args):
    trainval, train_indices, val_indices = get_splits(args.data_root)
    val_manifest = get_val_manifest(trainval, val_indices, args.manifest_root)

    train_dataset = PetRestorationDataset(trainval, train_indices)
    val_manifest_dataset = ManifestDataset(trainval, val_manifest)

    os.makedirs(args.optuna_root, exist_ok=True)
    optuna_db = f"sqlite:///{args.optuna_root}/task1.db"

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        bottleneck_dim = trial.suggest_categorical("bottleneck_dim", [128, 256, 512])
        base_ch = trial.suggest_categorical("base_ch", [16, 32, 64])
        dropout = trial.suggest_float("dropout", 0.0, 0.3)
        alpha = trial.suggest_float("alpha", 0.5, 0.95)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_manifest_dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=2, collate_fn=manifest_collate_fn)

        model = UniversalAutoencoder(base_ch=base_ch, bottleneck_dim=bottleneck_dim, dropout=dropout).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        val_loss = None
        for epoch in range(args.trial_epochs):
            model.train()
            for corrupted, clean, label in train_loader:
                corrupted, clean = corrupted.to(args.device), clean.to(args.device)
                optimizer.zero_grad()
                pred = model(corrupted)
                loss, _, _ = combined_loss(pred, clean, alpha)
                loss.backward()
                optimizer.step()

            model.eval()
            val_loss_sum = 0
            with torch.no_grad():
                for corrupted, clean, label, entry in val_loader:
                    corrupted, clean = corrupted.to(args.device), clean.to(args.device)
                    pred = model(corrupted)
                    loss, _, _ = combined_loss(pred, clean, alpha)
                    val_loss_sum += loss.item() * corrupted.size(0)
            val_loss = val_loss_sum / len(val_manifest_dataset)

            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return val_loss

    study = optuna.create_study(
        study_name="task1_universal_autoencoder",
        storage=optuna_db,
        direction="minimize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(objective, n_trials=args.n_trials)

    print("Best trial:", study.best_trial.params)
    print("Best val_loss:", study.best_value)

    os.makedirs(os.path.dirname(args.best_params_out), exist_ok=True)
    with open(args.best_params_out, "w") as f:
        json.dump({**study.best_trial.params, "optuna_best_val_loss": study.best_value}, f, indent=2)


def run_final(args):
    with open(args.best_params_out) as f:
        best_params = json.load(f)

    trainval, train_indices, val_indices = get_splits(args.data_root)
    val_manifest = get_val_manifest(trainval, val_indices, args.manifest_root)

    train_dataset = PetRestorationDataset(trainval, train_indices)
    val_manifest_dataset = ManifestDataset(trainval, val_manifest)

    train_loader = DataLoader(train_dataset, batch_size=best_params["batch_size"], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_manifest_dataset, batch_size=best_params["batch_size"], shuffle=False,
                             num_workers=2, collate_fn=manifest_collate_fn)

    model = UniversalAutoencoder(
        base_ch=best_params["base_ch"],
        bottleneck_dim=best_params["bottleneck_dim"],
        dropout=best_params["dropout"],
    ).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.checkpoint_root, exist_ok=True)
    latest_ckpt = f"{args.checkpoint_root}/task1_final_latest.pt"
    best_ckpt = f"{args.checkpoint_root}/task1_final_best.pt"

    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume and os.path.exists(latest_ckpt):
        ckpt = torch.load(latest_ckpt)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt["best_val_loss"]
        print(f"Resumed from epoch {start_epoch}, best_val_loss so far: {best_val_loss:.4f}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss_sum = 0
        for corrupted, clean, label in train_loader:
            corrupted, clean = corrupted.to(args.device), clean.to(args.device)
            optimizer.zero_grad()
            pred = model(corrupted)
            loss, _, _ = combined_loss(pred, clean, best_params["alpha"])
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * corrupted.size(0)
        train_loss = train_loss_sum / len(train_dataset)
        scheduler.step()

        model.eval()
        val_loss_sum = 0
        with torch.no_grad():
            for corrupted, clean, label, entry in val_loader:
                corrupted, clean = corrupted.to(args.device), clean.to(args.device)
                pred = model(corrupted)
                loss, _, _ = combined_loss(pred, clean, best_params["alpha"])
                val_loss_sum += loss.item() * corrupted.size(0)
        val_loss = val_loss_sum / len(val_manifest_dataset)

        print(f"Epoch {epoch + 1}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        torch.save({
            "epoch": epoch, "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
            "best_val_loss": best_val_loss, "val_loss": val_loss,
        }, latest_ckpt)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_ckpt)
            print(f"  -> New best! Saved to {best_ckpt}")

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"{args.checkpoint_root}/task1_final_epoch{epoch + 1}.pt")

    print(f"\nTraining complete. Best val_loss achieved: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["search", "final"], required=True)
    parser.add_argument("--data_root", default="/content/data")
    parser.add_argument("--manifest_root", default="/content/drive/MyDrive/GenAI_A1/manifests")
    parser.add_argument("--optuna_root", default="/content/drive/MyDrive/GenAI_A1/optuna")
    parser.add_argument("--checkpoint_root", default="/content/drive/MyDrive/GenAI_A1/checkpoints")
    parser.add_argument("--best_params_out", default="configs/task1_best_params.json")
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--trial_epochs", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.mode == "search":
        run_search(args)
    else:
        run_final(args)
