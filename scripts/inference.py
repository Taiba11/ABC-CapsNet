"""
Single-File Inference Script for ABC-CapsNet.

Usage:
    python scripts/inference.py \
        --checkpoint experiments/asvspoof2019/best_model.pth \
        --audio_path path/to/audio.wav
"""

import os
import sys
import argparse

import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import ABCCapsNet
from datasets.preprocessing import AudioPreprocessor, MelSpectrogramGenerator
from torchvision import transforms


def main():
    parser = argparse.ArgumentParser(description="ABC-CapsNet Inference")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--audio_path", type=str, required=True,
                        help="Path to audio file (.wav, .flac, etc.)")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config file")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Load config
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    # Build and load model
    model = ABCCapsNet(num_classes=2).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Preprocess audio
    preprocessor = AudioPreprocessor(sample_rate=16000, duration=4.0)
    mel_generator = MelSpectrogramGenerator(sample_rate=16000)

    print(f"\nProcessing: {args.audio_path}")

    waveform = preprocessor.load_audio(args.audio_path)
    mel_image = mel_generator.generate(waveform)

    # Normalize
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    mel_image = normalize(mel_image).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        predictions, confidences = model.predict(mel_image)

    pred_label = predictions.item()
    conf = confidences.squeeze()
    real_conf = conf[0].item()
    fake_conf = conf[1].item()

    label_str = "REAL (Bonafide)" if pred_label == 0 else "FAKE (Spoofed)"

    print(f"\n{'='*50}")
    print(f"  Prediction:  {label_str}")
    print(f"  Confidence:  Real={real_conf:.4f} | Fake={fake_conf:.4f}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
