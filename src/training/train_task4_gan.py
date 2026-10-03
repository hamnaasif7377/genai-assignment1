"""
Task 4: Train the conditional GAN for face-to-sketch generation.
Includes Optuna search and a final training run, following the same
pattern as Tasks 1-2. GAN trials use fewer epochs per the assignment's
allowance for computationally demanding search spaces.
"""
import argparse
import json
import os

import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.fs2k_dataset import FS2KPairedDataset
from src.models.task4_gan import UNetGenerator, PatchGANDiscriminator


def load_splits(data_root, seed=42):
    from sklearn.model_selection import train_test_split
    with open(os.path.join(data_root, "anno_train.json")) as f:
        train_anno = json.load(f)
    styles = [e["style"] for e in train_anno]
    train_split, val_split = train_test_split(
        train_anno, test_size=0.15, random_state=seed, stratify=styles
    )
    return train_split, val_split


def train_one_epoch(gen, disc, loader, opt_g, opt_d, criterion_gan, criterion_l1, lambda_l1, device):
    gen.train()
    disc.train()
    g_loss_sum, d_loss_sum, l1_loss_sum = 0.0, 0.0, 0.0
    n = 0

    for photo, sketch, style in loader:
        photo, sketch, style = photo.to(device), sketch.to(device), style.to(device)
        bs = photo.size(0)

        # --- Train Discriminator ---
        opt_d.zero_grad()
        fake_sketch = gen(photo, style)

        real_logits = disc(photo, sketch, style)
        fake_logits = disc(photo, fake_sketch.detach(), style)

        real_labels = torch.ones_like(real_logits)
        fake_labels = torch.zeros_like(fake_logits)

        d_loss_real = criterion_gan(real_logits, real_labels)
        d_loss_fake = criterion_gan(fake_logits, fake_labels)
        d_loss = (d_loss_real + d_loss_fake) * 0.5
        d_loss.backward()
        opt_d.step()

        # --- Train Generator ---
        opt_g.zero_grad()
        fake_logits_for_g = disc(photo, fake_sketch, style)
        g_adv_loss = criterion_gan(fake_logits_for_g, torch.ones_like(fake_logits_for_g))
        g_l1_loss = criterion_l1(fake_sketch, sketch)
        g_loss = g_adv_loss + lambda_l1 * g_l1_loss
        g_loss.backward()
        opt_g.step()

        g_loss_sum += g_adv_loss.item() * bs
        d_loss_sum += d_loss.item() * bs
        l1_loss_sum += g_l1_loss.item() * bs
        n += bs

    return g_loss_sum / n, d_loss_sum / n, l1_loss_sum / n


@torch.no_grad()
def evaluate(gen, loader, criterion_l1, device):
    gen.eval()
    l1_sum, n = 0.0, 0
    for photo, sketch, style in loader:
        photo, sketch, style = photo.to(device), sketch.to(device), style.to(device)
        fake_sketch = gen(photo, style)
        l1_sum += criterion_l1(fake_sketch, sketch).item() * photo.size(0)
        n += photo.size(0)
    return l1_sum / n


def run_search(args):
    train_split, val_split = load_splits(args.data_root)
    train_dataset = FS2KPairedDataset(train_split, args.data_root, augment=True)
    val_dataset = FS2KPairedDataset(val_split, args.data_root, augment=False)

    os.makedirs(args.optuna_root, exist_ok=True)
    optuna_db = f"sqlite:///{args.optuna_root}/task4_gan.db"

    def objective(trial):
        lr_g = trial.suggest_float("lr_g", 1e-5, 1e-3, log=True)
        lr_d = trial.suggest_float("lr_d", 1e-5, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
        base_ch = trial.suggest_categorical("base_ch", [32, 64])
        dropout = trial.suggest_categorical("dropout", [True, False])
        style_dim = trial.suggest_categorical("style_dim", [8, 16, 32])
        lambda_l1 = trial.suggest_float("lambda_l1", 10, 200, log=True)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        gen = UNetGenerator(base_ch=base_ch, style_dim=style_dim, dropout=dropout).to(args.device)
        disc = PatchGANDiscriminator(base_ch=base_ch, style_dim=style_dim).to(args.device)

        opt_g = torch.optim.Adam(gen.parameters(), lr=lr_g, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(disc.parameters(), lr=lr_d, betas=(0.5, 0.999))

        criterion_gan = nn.BCEWithLogitsLoss()
        criterion_l1 = nn.L1Loss()

        val_l1 = None
        for epoch in range(args.trial_epochs):
            train_one_epoch(gen, disc, train_loader, opt_g, opt_d, criterion_gan, criterion_l1, lambda_l1, args.device)
            val_l1 = evaluate(gen, val_loader, criterion_l1, args.device)

            trial.report(val_l1, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return val_l1

    study = optuna.create_study(
        study_name="task4_gan", storage=optuna_db,
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

    train_split, val_split = load_splits(args.data_root)
    train_dataset = FS2KPairedDataset(train_split, args.data_root, augment=True)
    val_dataset = FS2KPairedDataset(val_split, args.data_root, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=best_params["batch_size"], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=best_params["batch_size"], shuffle=False, num_workers=2)

    gen = UNetGenerator(base_ch=best_params["base_ch"], style_dim=best_params["style_dim"],
                         dropout=best_params["dropout"]).to(args.device)
    disc = PatchGANDiscriminator(base_ch=best_params["base_ch"], style_dim=best_params["style_dim"]).to(args.device)

    opt_g = torch.optim.Adam(gen.parameters(), lr=best_params["lr_g"], betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=best_params["lr_d"], betas=(0.5, 0.999))

    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_l1 = nn.L1Loss()

    os.makedirs(args.checkpoint_root, exist_ok=True)
    best_gen_ckpt = f"{args.checkpoint_root}/task4_generator_best.pt"
    latest_gen_ckpt = f"{args.checkpoint_root}/task4_generator_latest.pt"
    latest_disc_ckpt = f"{args.checkpoint_root}/task4_discriminator_latest.pt"

    start_epoch = 0
    best_val_l1 = float("inf")
    if args.resume and os.path.exists(latest_gen_ckpt):
        gen.load_state_dict(torch.load(latest_gen_ckpt)["model_state"])
        disc.load_state_dict(torch.load(latest_disc_ckpt)["model_state"])
        ckpt = torch.load(latest_gen_ckpt)
        opt_g.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val_l1 = ckpt["best_val_l1"]
        print(f"Resumed from epoch {start_epoch}, best_val_l1 so far: {best_val_l1:.4f}")

    for epoch in range(start_epoch, args.epochs):
        g_adv, d_loss, l1_loss = train_one_epoch(
            gen, disc, train_loader, opt_g, opt_d, criterion_gan, criterion_l1,
            best_params["lambda_l1"], args.device
        )
        val_l1 = evaluate(gen, val_loader, criterion_l1, args.device)

        print(f"Epoch {epoch+1}/{args.epochs}  "
              f"g_adv={g_adv:.4f}  d_loss={d_loss:.4f}  train_l1={l1_loss:.4f}  val_l1={val_l1:.4f}")

        torch.save({
            "epoch": epoch, "model_state": gen.state_dict(),
            "optimizer_state": opt_g.state_dict(), "best_val_l1": best_val_l1, "val_l1": val_l1,
        }, latest_gen_ckpt)
        torch.save({
            "epoch": epoch, "model_state": disc.state_dict(), "optimizer_state": opt_d.state_dict(),
        }, latest_disc_ckpt)

        if val_l1 < best_val_l1:
            best_val_l1 = val_l1
            torch.save(gen.state_dict(), best_gen_ckpt)
            print(f"  -> New best! Saved to {best_gen_ckpt}")

        if (epoch + 1) % 10 == 0:
            torch.save(gen.state_dict(), f"{args.checkpoint_root}/task4_generator_epoch{epoch+1}.pt")

    print(f"\nTraining complete. Best val_l1: {best_val_l1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["search", "final"], required=True)
    parser.add_argument("--data_root", default="/content/data/FS2K")
    parser.add_argument("--optuna_root", default="/content/drive/MyDrive/GenAI_A1_Task4/optuna")
    parser.add_argument("--checkpoint_root", default="/content/drive/MyDrive/GenAI_A1_Task4/checkpoints")
    parser.add_argument("--best_params_out", default="configs/task4_best_params.json")
    parser.add_argument("--n_trials", type=int, default=15)
    parser.add_argument("--trial_epochs", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.mode == "search":
        run_search(args)
    else:
        run_final(args)
