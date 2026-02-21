"""
Training Logger for ABC-CapsNet.

Handles logging to console, file, and TensorBoard.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path

import torch


class TrainingLogger:
    """
    Unified logger for training experiments.

    Args:
        output_dir (str): Directory to save logs and checkpoints.
        experiment_name (str): Name of the experiment.
        use_tensorboard (bool): Whether to log to TensorBoard.
    """

    def __init__(
        self,
        output_dir: str,
        experiment_name: str = "abc_capsnet",
        use_tensorboard: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = f"{experiment_name}_{timestamp}"

        # File logger
        log_file = self.output_dir / f"{self.experiment_name}.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(str(log_file)),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(self.experiment_name)

        # TensorBoard
        self.writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = self.output_dir / "tensorboard" / self.experiment_name
                self.writer = SummaryWriter(str(tb_dir))
            except ImportError:
                self.logger.warning("TensorBoard not available. Skipping.")

        # Training history
        self.history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_eer": []}

        self.logger.info(f"Experiment: {self.experiment_name}")
        self.logger.info(f"Output directory: {self.output_dir}")

    def log(self, message: str):
        """Log a message."""
        self.logger.info(message)

    def log_epoch(self, epoch: int, train_loss: float, val_loss: float,
                  val_acc: float, val_eer: float = None):
        """Log epoch metrics."""
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)
        self.history["val_acc"].append(val_acc)
        if val_eer is not None:
            self.history["val_eer"].append(val_eer)

        msg = (
            f"Epoch [{epoch}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}%"
        )
        if val_eer is not None:
            msg += f" | Val EER: {val_eer:.4f}%"
        self.logger.info(msg)

        # TensorBoard
        if self.writer:
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Loss/val", val_loss, epoch)
            self.writer.add_scalar("Accuracy/val", val_acc, epoch)
            if val_eer is not None:
                self.writer.add_scalar("EER/val", val_eer, epoch)

    def save_checkpoint(self, model, optimizer, epoch, val_acc, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_acc,
        }

        # Save latest
        path = self.output_dir / "latest_model.pth"
        torch.save(checkpoint, str(path))

        # Save best
        if is_best:
            best_path = self.output_dir / "best_model.pth"
            torch.save(checkpoint, str(best_path))
            self.logger.info(f"Saved best model (Acc: {val_acc:.2f}%)")

    def save_history(self):
        """Save training history to JSON."""
        path = self.output_dir / "training_history.json"
        with open(str(path), "w") as f:
            json.dump(self.history, f, indent=2)

    def close(self):
        """Close logger and TensorBoard writer."""
        if self.writer:
            self.writer.close()
        self.save_history()
