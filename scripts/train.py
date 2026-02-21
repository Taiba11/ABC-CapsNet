"""
Training Script for ABC-CapsNet.

Usage:
    python scripts/train.py --config configs/asvspoof2019.yaml --data_dir data/spectrograms/asvspoof2019

    python scripts/train.py \
        --config configs/default.yaml \
        --data_dir data/spectrograms/asvspoof2019 \
        --output_dir experiments/asvspoof2019 \
        --epochs 100 --batch_size 32 --lr 0.0001
"""

import os
import sys
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import ABCCapsNet
from models.losses import MarginLoss, CombinedLoss
from datasets import ASVspoof2019Dataset, FoRDataset, AudioDeepfakeDataset
from utils.metrics import compute_eer, compute_accuracy, compute_metrics
from utils.logger import TrainingLogger


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str):
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def build_model(config):
    """Build ABC-CapsNet model from configuration."""
    model_cfg = config.get("model", {})
    caps1 = model_cfg.get("capsule_net_1", {})
    caps2 = model_cfg.get("capsule_net_2", {})

    model = ABCCapsNet(
        num_classes=2,
        pretrained_backbone=model_cfg.get("backbone", {}).get("pretrained", True),
        freeze_backbone=model_cfg.get("backbone", {}).get("freeze_layers", 0),
        attention_hidden_dim=model_cfg.get("attention", {}).get("hidden_dim", 256),
        attention_dropout=model_cfg.get("attention", {}).get("dropout", 0.1),
        cn1_primary_num_caps=caps1.get("primary_caps", {}).get("num_capsules", 8),
        cn1_primary_cap_dim=caps1.get("primary_caps", {}).get("capsule_dim", 32),
        cn1_primary_kernel=caps1.get("primary_caps", {}).get("kernel_size", 9),
        cn1_primary_stride=caps1.get("primary_caps", {}).get("stride", 2),
        cn1_digit_cap_dim=caps1.get("digit_caps", {}).get("capsule_dim", 16),
        cn2_secondary_num_caps=caps2.get("primary_caps", {}).get("num_capsules", 4),
        cn2_secondary_cap_dim=caps2.get("primary_caps", {}).get("capsule_dim", 16),
        cn2_digit_cap_dim=caps2.get("digit_caps", {}).get("capsule_dim", 16),
        routing_iterations=caps1.get("digit_caps", {}).get("routing_iterations", 3),
    )

    return model


def build_loss(config):
    """Build loss function from configuration."""
    loss_cfg = config.get("training", {}).get("loss", {})
    loss_name = loss_cfg.get("name", "margin")

    if loss_name == "margin":
        return MarginLoss(
            m_plus=loss_cfg.get("m_plus", 0.9),
            m_minus=loss_cfg.get("m_minus", 0.1),
            lambda_val=loss_cfg.get("lambda_val", 0.5),
        )
    elif loss_name == "combined":
        return CombinedLoss(
            m_plus=loss_cfg.get("m_plus", 0.9),
            m_minus=loss_cfg.get("m_minus", 0.1),
            lambda_val=loss_cfg.get("lambda_val", 0.5),
        )
    else:
        return MarginLoss()


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        v_j = model(images)
        loss = criterion(v_j, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        running_loss += loss.item() * images.size(0)

        # Predictions from capsule norms
        capsule_norms = torch.sqrt((v_j ** 2).sum(dim=-1) + 1e-8)
        preds = capsule_norms.argmax(dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().detach().numpy())

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = compute_accuracy(all_labels, all_preds)

    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluate the model."""
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_scores = []

    for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        v_j = model(images)
        loss = criterion(v_j, labels)

        running_loss += loss.item() * images.size(0)

        capsule_norms = torch.sqrt((v_j ** 2).sum(dim=-1) + 1e-8)
        preds = capsule_norms.argmax(dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_scores.extend(capsule_norms[:, 1].cpu().numpy())  # Score for "fake" class

    epoch_loss = running_loss / len(dataloader.dataset)
    metrics = compute_metrics(all_labels, all_preds, all_scores)
    metrics["loss"] = epoch_loss

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train ABC-CapsNet")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config file")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to spectrogram data directory")
    parser.add_argument("--output_dir", type=str, default="experiments/default",
                        help="Output directory for checkpoints and logs")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=str, default="asvspoof2019",
                        choices=["asvspoof2019", "for"])
    parser.add_argument("--mode", type=str, default="spectrogram",
                        choices=["spectrogram", "audio"])
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override config with CLI args
    training_cfg = config.get("training", {})
    epochs = args.epochs or training_cfg.get("epochs", 100)
    batch_size = args.batch_size or training_cfg.get("batch_size", 32)
    lr = args.lr or training_cfg.get("optimizer", {}).get("lr", 0.0001)

    # Set seed
    set_seed(args.seed)

    # Device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  ABC-CapsNet Training")
    print(f"  Device: {device}")
    print(f"  Epochs: {epochs} | Batch Size: {batch_size} | LR: {lr}")
    print(f"{'='*60}\n")

    # Build model
    model = build_model(config).to(device)
    total_params, trainable_params = model.get_num_params()
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable\n")

    # Build loss and optimizer
    criterion = build_loss(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Build datasets
    if args.dataset == "asvspoof2019":
        dataset_handler = ASVspoof2019Dataset(args.data_dir, mode=args.mode)
        train_dataset = dataset_handler.get_train_dataset()
        val_dataset = dataset_handler.get_dev_dataset()
    elif args.dataset == "for":
        dataset_handler = FoRDataset(args.data_dir, mode=args.mode)
        datasets = dataset_handler.get_datasets()
        train_dataset = datasets.get("train")
        val_dataset = datasets.get("val")

    # Dataloaders
    num_workers = config.get("data", {}).get("num_workers", 4)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    # Logger
    logger = TrainingLogger(args.output_dir, use_tensorboard=True)
    logger.log(f"Training samples: {len(train_dataset)}")
    logger.log(f"Validation samples: {len(val_dataset)}")

    # Training loop
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Evaluate
        val_metrics = evaluate(model, val_loader, criterion, device)
        val_loss = val_metrics["loss"]
        val_acc = val_metrics["accuracy"]
        val_eer = val_metrics.get("eer", None)

        # Step scheduler
        scheduler.step()

        # Log
        logger.log_epoch(epoch, train_loss, val_loss, val_acc, val_eer)

        # Save checkpoint
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
        logger.save_checkpoint(model, optimizer, epoch, val_acc, is_best)

    # Final log
    logger.log(f"\nTraining complete. Best validation accuracy: {best_acc:.2f}%")
    logger.close()


if __name__ == "__main__":
    main()
