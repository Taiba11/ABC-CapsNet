"""
Evaluation Script for ABC-CapsNet.

Usage:
    python scripts/evaluate.py \
        --config configs/asvspoof2019.yaml \
        --checkpoint experiments/asvspoof2019/best_model.pth \
        --data_dir data/ASVspoof2019/LA \
        --split eval

    # Per-attack evaluation on ASVspoof2019
    python scripts/evaluate.py \
        --config configs/asvspoof2019.yaml \
        --checkpoint experiments/asvspoof2019/best_model.pth \
        --data_dir data/ASVspoof2019/LA \
        --split eval --per_attack
"""

import os
import sys
import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import ABCCapsNet
from datasets import ASVspoof2019Dataset, FoRDataset
from utils.metrics import compute_metrics
from utils.visualization import plot_confusion_matrix, plot_eer_comparison


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def evaluate_dataset(model, dataloader, device):
    model.eval()
    all_labels, all_preds, all_scores = [], [], []

    for images, labels in tqdm(dataloader, desc="Evaluating"):
        images = images.to(device)
        v_j = model(images)
        capsule_norms = torch.sqrt((v_j ** 2).sum(dim=-1) + 1e-8)
        preds = capsule_norms.argmax(dim=1)

        all_labels.extend(labels.numpy())
        all_preds.extend(preds.cpu().numpy())
        all_scores.extend(capsule_norms[:, 1].cpu().numpy())

    return compute_metrics(all_labels, all_preds, all_scores)


def main():
    parser = argparse.ArgumentParser(description="Evaluate ABC-CapsNet")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/")
    parser.add_argument("--split", type=str, default="eval")
    parser.add_argument("--dataset", type=str, default="asvspoof2019",
                        choices=["asvspoof2019", "for"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--per_attack", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    config = load_config(args.config)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Load model
    from scripts.train import build_model
    model = build_model(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    print(f"\n{'='*60}")
    print(f"  ABC-CapsNet Evaluation | {args.dataset} | {args.split}")
    print(f"{'='*60}\n")

    results = {}

    if args.dataset == "asvspoof2019":
        handler = ASVspoof2019Dataset(args.data_dir, mode="audio")

        if args.per_attack:
            attacks = [f"A{i:02d}" for i in range(7, 20)]
            eer_dict = {}
            for attack in attacks:
                print(f"\n--- Attack: {attack} ---")
                ds = handler.get_eval_dataset(attack_type=attack)
                loader = DataLoader(ds, batch_size=args.batch_size, num_workers=4)
                metrics = evaluate_dataset(model, loader, device)
                eer_dict[attack] = metrics.get("eer", 0)
                results[attack] = metrics
                print(f"  Accuracy: {metrics['accuracy']:.2f}% | EER: {metrics.get('eer', 0):.4f}%")

            plot_eer_comparison(eer_dict, os.path.join(args.output_dir, "eer_per_attack.png"))

        # Full evaluation
        if args.split == "eval":
            ds = handler.get_eval_dataset()
        elif args.split == "dev":
            ds = handler.get_dev_dataset()
        else:
            ds = handler.get_train_dataset()

        loader = DataLoader(ds, batch_size=args.batch_size, num_workers=4)
        metrics = evaluate_dataset(model, loader, device)
        results["full"] = metrics

    elif args.dataset == "for":
        handler = FoRDataset(args.data_dir, mode="audio")
        datasets = handler.get_datasets()
        ds = datasets.get("test") or datasets.get("val")
        loader = DataLoader(ds, batch_size=args.batch_size, num_workers=4)
        metrics = evaluate_dataset(model, loader, device)
        results["full"] = metrics

    # Print results
    full = results.get("full", {})
    print(f"\n{'='*60}")
    print(f"  Final Results:")
    print(f"    Accuracy:  {full.get('accuracy', 0):.2f}%")
    print(f"    EER:       {full.get('eer', 0):.4f}%")
    print(f"    Precision: {full.get('precision', 0):.2f}%")
    print(f"    Recall:    {full.get('recall', 0):.2f}%")
    print(f"    F1-Score:  {full.get('f1', 0):.2f}%")
    print(f"{'='*60}\n")

    # Save results
    save_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {save_path}")

    # Plot confusion matrix
    if "confusion_matrix" in full:
        cm = np.array(full["confusion_matrix"])
        plot_confusion_matrix(cm, save_path=os.path.join(args.output_dir, "confusion_matrix.png"))


if __name__ == "__main__":
    main()
