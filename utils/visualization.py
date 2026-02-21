"""
Visualization Utilities for ABC-CapsNet.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_spectrogram(spectrogram, title="Mel Spectrogram", save_path=None):
    """Plot a Mel spectrogram."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", cmap="viridis")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Time Frames")
    ax.set_ylabel("Mel Frequency Bins")
    plt.colorbar(im, ax=ax, label="dB")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_training_curves(history, save_path=None):
    """Plot training loss and validation accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss curves
    ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=2)
    ax1.plot(epochs, history["val_loss"], "r-", label="Val Loss", linewidth=2)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Accuracy curve
    ax2.plot(epochs, history["val_acc"], "g-", label="Val Accuracy", linewidth=2)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy (%)", fontsize=12)
    ax2.set_title("Validation Accuracy", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(cm, class_names=None, title="Confusion Matrix", save_path=None):
    """Plot confusion matrix heatmap."""
    if class_names is None:
        class_names = ["Real", "Fake"]

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        annot_kws={"size": 14},
    )
    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("Actual", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_eer_comparison(results_dict, save_path=None):
    """
    Plot EER comparison across attacks/datasets.

    Args:
        results_dict: dict mapping attack/dataset names to EER values.
    """
    names = list(results_dict.keys())
    eers = list(results_dict.values())

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    bars = ax.bar(names, eers, color=sns.color_palette("viridis", len(names)), edgecolor="black")

    for bar, eer in zip(bars, eers):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{eer:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xlabel("Attack / Dataset", fontsize=12)
    ax.set_ylabel("EER (%)", fontsize=12)
    ax.set_title("Equal Error Rate Comparison", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
