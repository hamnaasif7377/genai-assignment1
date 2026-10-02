"""
Task 2, Part A: train the corruption classifier.
Includes Optuna search (learning rate, batch size, conv channels, dropout,
weight decay) and a final training run, following the same pattern as Task 1.
"""
import argparse
import json
import os
import random

import numpy as np
import optuna
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                              confusion_matrix)
from torch.utils.data import DataLoader
from torchvision.datasets import OxfordIIITPet

from src.data.balanced_sampler import BalancedCorruptionDataset
from src.data.corruptions import CORRUPTION_TYPES
from src.models.task2_classifier import CorruptionClassifier

SEED = 42


def get_splits(data_root):
    trainval = OxfordIIITPet(root=data_root, split="trainval", download=True)
    n_total = len(trainval)
    indices = list(range(n_total))
    rng = random.Random(SEED)
    rng.shuffle(indices)
    n_train = int(0.8 * n_total)
    return trainval, indices[:n_train], indices[n_train:]


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    return acc, precision, recall, f1, all_labels, all_preds


def run_search(args):
    trainval, train_indices, val_indices = get_splits(args.data_root)
    train_dataset = BalancedCorruptionDataset(trainval, train_indices)
    val_dataset = BalancedCorruptionDataset(trainval, val_indices)

    os.makedirs(args.optuna_root, exist_ok=True)
    optuna_db = f"sqlite:///{args.optuna_root}/task2_classifier.db"

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        base_ch = trial.suggest_categorical("base_ch", [16, 32, 64])
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        model = CorruptionClassifier(base_ch=base_ch, dropout=dropout).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(args.trial_epochs):
            model.train()
            for x, y in train_loader:
                x, y = x.to(args.device), y.to(args.device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()

            acc, _, _, f1, _, _ = evaluate(model, val_loader, args.device)
            trial.report(1 - f1, epoch)  # minimize (1 - macro F1)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return 1 - f1

    study = optuna.create_study(
        study_name="task2_classifier", storage=optuna_db,
        direction="minimize", load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(objective, n_trials=args.n_trials)

    print("Best trial:", study.best_trial.params)
    print("Best (1 - macro F1):", study.best_value)

    os.makedirs(os.path.dirname(args.best_params_out), exist_ok=True)
    with open(args.best_params_out, "w") as f:
        json.dump({**study.best_trial.params, "optuna_best_value": study.best_value}, f, indent=2)


def run_final(args):
    with open(args.best_params_out, encoding="utf-8-sig") as f:
        best_params = json.load(f)

    trainval, train_indices, val_indices = get_splits(args.data_root)
    train_dataset = BalancedCorruptionDataset(trainval, train_indices)
    val_dataset = BalancedCorruptionDataset(trainval, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=best_params["batch_size"], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=best_params["batch_size"], shuffle=False, num_workers=2)

    model = CorruptionClassifier(base_ch=best_params["base_ch"], dropout=best_params["dropout"]).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    os.makedirs(args.checkpoint_root, exist_ok=True)
    best_ckpt = f"{args.checkpoint_root}/task2_classifier_best.pt"
    best_f1 = 0.0

    for epoch in range(args.epochs):
        model.train()
        train_loss_sum = 0
        for x, y in train_loader:
            x, y = x.to(args.device), y.to(args.device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * x.size(0)
        train_loss = train_loss_sum / len(train_dataset)

        acc, precision, recall, f1, labels, preds = evaluate(model, val_loader, args.device)
        print(f"Epoch {epoch+1}/{args.epochs}  train_loss={train_loss:.4f}  "
              f"val_acc={acc:.4f}  val_f1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), best_ckpt)
            print(f"  -> New best! Saved to {best_ckpt}")

    # Final confusion matrix + per-class metrics, using the BEST checkpoint
    model.load_state_dict(torch.load(best_ckpt))
    acc, precision, recall, f1, labels, preds = evaluate(model, val_loader, args.device)
    cm = confusion_matrix(labels, preds, normalize="true")
    per_class_p, per_class_r, per_class_f1, support = precision_recall_fscore_support(
        labels, preds, average=None, zero_division=0
    )

    report = {
        "overall_accuracy": float(acc),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "per_class": {
            CORRUPTION_TYPES[i]: {
                "precision": float(per_class_p[i]),
                "recall": float(per_class_r[i]),
                "f1": float(per_class_f1[i]),
                "support": int(support[i]),
            } for i in range(len(CORRUPTION_TYPES))
        },
        "confusion_matrix_normalized": cm.tolist(),
        "confusion_matrix_labels": CORRUPTION_TYPES,
    }
    os.makedirs(os.path.dirname(args.report_out), exist_ok=True)
    with open(args.report_out, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nFinal best val_f1: {best_f1:.4f}")
    print(f"Saved classification report to {args.report_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["search", "final"], required=True)
    parser.add_argument("--data_root", default="/content/data")
    parser.add_argument("--optuna_root", default="/content/drive/MyDrive/GenAI_A1/optuna")
    parser.add_argument("--checkpoint_root", default="/content/drive/MyDrive/GenAI_A1/checkpoints")
    parser.add_argument("--best_params_out", default="configs/task2_classifier_best_params.json")
    parser.add_argument("--report_out", default="/content/drive/MyDrive/GenAI_A1/reports/task2_classifier_report.json")
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--trial_epochs", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.mode == "search":
        run_search(args)
    else:
        run_final(args)
