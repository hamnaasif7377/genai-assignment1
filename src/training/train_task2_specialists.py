"""
Task 2, Part B: train the three specialist autoencoders.

Per the assignment, a SHARED Optuna search (using salt_pepper as the
representative corruption) finds one common architecture, which is
then used to independently train three specialists (salt_pepper,
blur, occlusion), each with its own trained parameters.

Usage:
    python -m src.training.train_task2_specialists --mode search --n_trials 20
    python -m src.training.train_task2_specialists --mode final --corruption_type salt_pepper --epochs 40
    python -m src.training.train_task2_specialists --mode final --corruption_type blur --epochs 40
    python -m src.training.train_task2_specialists --mode final --corruption_type occlusion --epochs 40
"""
import argparse
import json
import os
import random

import optuna
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import OxfordIIITPet

from src.data.single_corruption_dataset import SingleCorruptionDataset
from src.models.task2_specialists import SpecialistAutoencoder
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


def run_search(args):
    """Shared search using salt_pepper as the representative corruption type."""
    trainval, train_indices, val_indices = get_splits(args.data_root)
    train_dataset = SingleCorruptionDataset(trainval, train_indices, "salt_pepper")
    val_dataset = SingleCorruptionDataset(trainval, val_indices, "salt_pepper")

    os.makedirs(args.optuna_root, exist_ok=True)
    optuna_db = f"sqlite:///{args.optuna_root}/task2_specialists.db"

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        bottleneck_dim = trial.suggest_categorical("bottleneck_dim", [128, 256, 512])
        base_ch = trial.suggest_categorical("base_ch", [16, 32, 64])
        dropout = trial.suggest_float("dropout", 0.0, 0.3)
        alpha = trial.suggest_float("alpha", 0.5, 0.95)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        model = SpecialistAutoencoder(base_ch=base_ch, bottleneck_dim=bottleneck_dim, dropout=dropout).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        val_loss = None
        for epoch in range(args.trial_epochs):
            model.train()
            for corrupted, clean in train_loader:
                corrupted, clean = corrupted.to(args.device), clean.to(args.device)
                optimizer.zero_grad()
                pred = model(corrupted)
                loss, _, _ = combined_loss(pred, clean, alpha)
                loss.backward()
                optimizer.step()

            model.eval()
            val_loss_sum = 0
            with torch.no_grad():
                for corrupted, clean in val_loader:
                    corrupted, clean = corrupted.to(args.device), clean.to(args.device)
                    pred = model(corrupted)
                    loss, _, _ = combined_loss(pred, clean, alpha)
                    val_loss_sum += loss.item() * corrupted.size(0)
            val_loss = val_loss_sum / len(val_dataset)

            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return val_loss

    study = optuna.create_study(
        study_name="task2_specialists_shared", storage=optuna_db,
        direction="minimize", load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(objective, n_trials=args.n_trials)

    print("Best trial:", study.best_trial.params)
    print("Best val_loss:", study.best_value)

    os.makedirs(os.path.dirname(args.best_params_out), exist_ok=True)
    with open(args.best_params_out, "w") as f:
        json.dump({**study.best_trial.params, "optuna_best_val_loss": study.best_value}, f, indent=2)


def run_final(args):
    """Train ONE specialist for args.corruption_type using the shared architecture."""
    with open(args.best_params_out, encoding="utf-8-sig") as f:
        best_params = json.load(f)

    trainval, train_indices, val_indices = get_splits(args.data_root)
    train_dataset = SingleCorruptionDataset(trainval, train_indices, args.corruption_type)
    val_dataset = SingleCorruptionDataset(trainval, val_indices, args.corruption_type)

    train_loader = DataLoader(train_dataset, batch_size=best_params["batch_size"], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=best_params["batch_size"], shuffle=False, num_workers=2)

    model = SpecialistAutoencoder(
        base_ch=best_params["base_ch"],
        bottleneck_dim=best_params["bottleneck_dim"],
        dropout=best_params["dropout"],
    ).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.checkpoint_root, exist_ok=True)
    best_ckpt = f"{args.checkpoint_root}/task2_specialist_{args.corruption_type}_best.pt"
    latest_ckpt = f"{args.checkpoint_root}/task2_specialist_{args.corruption_type}_latest.pt"

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
        for corrupted, clean in train_loader:
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
            for corrupted, clean in val_loader:
                corrupted, clean = corrupted.to(args.device), clean.to(args.device)
                pred = model(corrupted)
                loss, _, _ = combined_loss(pred, clean, best_params["alpha"])
                val_loss_sum += loss.item() * corrupted.size(0)
        val_loss = val_loss_sum / len(val_dataset)

        print(f"[{args.corruption_type}] Epoch {epoch+1}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        torch.save({
            "epoch": epoch, "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
            "best_val_loss": best_val_loss, "val_loss": val_loss,
        }, latest_ckpt)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_ckpt)
            print(f"  -> New best! Saved to {best_ckpt}")

    print(f"\n[{args.corruption_type}] Training complete. Best val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["search", "final"], required=True)
    parser.add_argument("--corruption_type", choices=["salt_pepper", "blur", "occlusion"])
    parser.add_argument("--data_root", default="/content/data")
    parser.add_argument("--optuna_root", default="/content/drive/MyDrive/GenAI_A1/optuna")
    parser.add_argument("--checkpoint_root", default="/content/drive/MyDrive/GenAI_A1/checkpoints")
    parser.add_argument("--best_params_out", default="configs/task2_specialists_best_params.json")
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--trial_epochs", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.mode == "search":
        run_search(args)
    else:
        if not args.corruption_type:
            parser.error("--corruption_type is required for --mode final")
        run_final(args)
