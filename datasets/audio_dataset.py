"""
Base Audio Deepfake Dataset.

Generic dataset class that loads precomputed Mel spectrogram images
or generates them on-the-fly from audio files.
"""

import os
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from .preprocessing import AudioPreprocessor, MelSpectrogramGenerator


class AudioDeepfakeDataset(Dataset):
    """
    Base dataset for audio deepfake detection.

    Supports two modes:
        1. Precomputed spectrograms: loads PNG/JPG images from disk.
        2. On-the-fly: generates Mel spectrograms from raw audio files.

    Args:
        file_list (list): List of (filepath, label) tuples.
            - filepath: path to audio file or spectrogram image.
            - label: 0 = bonafide (real), 1 = spoof (fake).
        mode (str): "spectrogram" for precomputed images, "audio" for on-the-fly.
        transform: Optional torchvision transforms for data augmentation.
        sample_rate (int): Audio sample rate (for on-the-fly mode).
        duration (float): Audio duration in seconds (for on-the-fly mode).
    """

    def __init__(
        self,
        file_list: list,
        mode: str = "spectrogram",
        transform=None,
        sample_rate: int = 16000,
        duration: float = 4.0,
    ):
        self.file_list = file_list
        self.mode = mode
        self.transform = transform or self._default_transform()

        if mode == "audio":
            self.preprocessor = AudioPreprocessor(
                sample_rate=sample_rate, duration=duration
            )
            self.mel_generator = MelSpectrogramGenerator(
                sample_rate=sample_rate
            )

    def _default_transform(self):
        """Default transforms: normalize with ImageNet stats."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filepath, label = self.file_list[idx]

        if self.mode == "spectrogram":
            # Load precomputed spectrogram image
            image = Image.open(filepath).convert("RGB")
            image = self.transform(image)
        elif self.mode == "audio":
            # Generate spectrogram on-the-fly
            waveform = self.preprocessor.load_audio(filepath)
            image = self.mel_generator.generate(waveform)
            # Apply normalization
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
            image = normalize(image)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        label = torch.tensor(label, dtype=torch.long)

        return image, label
