"""Train an ADHD EEG classifier with subject-wise stratified k-fold CV.

Usage:
    python -m src.train --config configs/config.yaml
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import EEGDataset, load_dataset, compute_class_weights
from src.models import build_model
from src.utils import load_config, set_seed, get_device, get_logger, EarlyStopping, save_checkpoint


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, all_preds, all_labels, all_probs = 0.0, [], [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            if train:
                optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits, y)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * x.size(0)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")  # only one class present in this batch/fold

    return avg_loss, acc, f1, auc


def main(config_path: str):
    config = load_config(config_path)
    set_seed(config["train"]["seed"])
    device = get_device(config["train"]["device"])
    logger = get_logger(log_dir=config["train"]["log_dir"])
    logger.info(f"Using device: {device}")

    X, y, groups = load_dataset(config)
    logger.info(f"Loaded dataset: X={X.shape}, y distribution={np.bincount(y)}")

    n_channels, n_timesteps = X.shape[1], X.shape[2]
    k_folds = config["train"]["k_folds"]
    sgkf = StratifiedGroupKFold(n_splits=k_folds, shuffle=True, random_state=config["train"]["seed"])

    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        logger.info(f"\n===== Fold {fold + 1}/{k_folds} =====")
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        train_ds = EEGDataset(X_train, y_train)
        val_ds = EEGDataset(X_val, y_val)
        train_loader = DataLoader(train_ds, batch_size=config["train"]["batch_size"], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config["train"]["batch_size"], shuffle=False)

        model = build_model(config, n_channels, n_timesteps).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config["train"]["lr"], weight_decay=config["train"]["weight_decay"]
        )

        if config["train"]["class_weighted_loss"]:
            weights = compute_class_weights(y_train).to(device)
            criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            criterion = nn.CrossEntropyLoss()

        early_stopper = EarlyStopping(patience=config["train"]["early_stopping_patience"], mode="max")
        best_ckpt_path = os.path.join(config["train"]["checkpoint_dir"], f"fold{fold + 1}_best.pt")

        pbar = tqdm(range(config["train"]["epochs"]), desc=f"Fold {fold + 1}")
        for epoch in pbar:
            train_loss, train_acc, train_f1, _ = run_epoch(
                model, train_loader, criterion, optimizer, device, train=True
            )
            val_loss, val_acc, val_f1, val_auc = run_epoch(
                model, val_loader, criterion, optimizer, device, train=False
            )

            pbar.set_postfix(
                {"tr_loss": f"{train_loss:.3f}", "val_acc": f"{val_acc:.3f}", "val_f1": f"{val_f1:.3f}"}
            )

            is_best = early_stopper.step(val_f1)
            if is_best:
                save_checkpoint(model, best_ckpt_path, val_acc=val_acc, val_f1=val_f1, val_auc=val_auc, epoch=epoch)

            if early_stopper.should_stop:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        fold_metrics.append(
            {"fold": fold + 1, "val_acc": val_acc, "val_f1": early_stopper.best_score, "val_auc": val_auc}
        )
        logger.info(f"Fold {fold + 1} best val F1: {early_stopper.best_score:.4f}")

    accs = [m["val_acc"] for m in fold_metrics]
    f1s = [m["val_f1"] for m in fold_metrics]
    aucs = [m["val_auc"] for m in fold_metrics]

    logger.info("\n===== Cross-validation summary =====")
    logger.info(f"Accuracy: {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    logger.info(f"F1:       {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
    logger.info(f"ROC-AUC:  {np.nanmean(aucs):.4f} +/- {np.nanstd(aucs):.4f}")

    # Save the single best fold checkpoint as the overall "best_model.pt"
    best_fold = max(fold_metrics, key=lambda m: m["val_f1"])
    src_path = os.path.join(config["train"]["checkpoint_dir"], f"fold{best_fold['fold']}_best.pt")
    dst_path = os.path.join(config["train"]["checkpoint_dir"], "best_model.pt")
    if os.path.exists(src_path):
        import shutil
        shutil.copy(src_path, dst_path)
        logger.info(f"Best overall model (fold {best_fold['fold']}) saved to {dst_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
