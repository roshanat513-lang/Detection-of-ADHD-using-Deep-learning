"""Standalone evaluation of a trained checkpoint on a held-out dataset.

Usage:
    python -m src.evaluate --config configs/config.yaml --checkpoint checkpoints/best_model.pt
"""
import argparse

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from torch.utils.data import DataLoader

from src.data.dataset import EEGDataset, load_dataset
from src.models import build_model
from src.utils import load_config, get_device, load_checkpoint, get_logger


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def main(config_path: str, checkpoint_path: str):
    config = load_config(config_path)
    device = get_device(config["train"]["device"])
    logger = get_logger(log_dir=config["train"]["log_dir"])

    X, y, _ = load_dataset(config)
    n_channels, n_timesteps = X.shape[1], X.shape[2]

    dataset = EEGDataset(X, y)
    loader = DataLoader(dataset, batch_size=config["evaluate"]["batch_size"], shuffle=False)

    model = build_model(config, n_channels, n_timesteps).to(device)
    load_checkpoint(model, checkpoint_path, map_location=device)

    labels, preds, probs = evaluate(model, loader, device)

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float("nan")

    logger.info("===== Evaluation Results =====")
    logger.info(f"Accuracy:  {acc:.4f}")
    logger.info(f"Precision: {prec:.4f}")
    logger.info(f"Recall:    {rec:.4f}")
    logger.info(f"F1-score:  {f1:.4f}")
    logger.info(f"ROC-AUC:   {auc:.4f}")
    logger.info(f"\nConfusion matrix:\n{confusion_matrix(labels, preds)}")
    logger.info(f"\n{classification_report(labels, preds, target_names=['control', 'adhd'], zero_division=0)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()
    main(args.config, args.checkpoint)
