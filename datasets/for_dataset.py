"""
Fake or Real (FoR) Dataset Loader.

Supports all four versions:
    - for-original: Raw TTS + real speech
    - for-norm: Volume-normalized
    - for-2seconds: 2-second clips
    - for-rerecorded: Re-recorded in real environments

Labels:
    - real → 0
    - fake → 1
"""

import os
from pathlib import Path
from typing import List, Optional

from sklearn.model_selection import train_test_split

from .audio_dataset import AudioDeepfakeDataset


class FoRDataset:
    """
    Fake or Real (FoR) dataset handler.

    Args:
        data_dir (str): Root directory of FoR dataset.
        spectrogram_dir (str): Directory with precomputed spectrograms (optional).
        mode (str): "audio" for raw files, "spectrogram" for precomputed images.
        versions (list): Which FoR versions to include.
        use_combined (bool): Whether to combine all versions.
    """

    VERSIONS = ["for-original", "for-norm", "for-2seconds", "for-rerecorded"]
    AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg"}

    def __init__(
        self,
        data_dir: str,
        spectrogram_dir: Optional[str] = None,
        mode: str = "audio",
        versions: Optional[List[str]] = None,
        use_combined: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.spectrogram_dir = Path(spectrogram_dir) if spectrogram_dir else None
        self.mode = mode
        self.versions = versions or self.VERSIONS
        self.use_combined = use_combined

    def _scan_directory(self, directory: Path):
        """
        Scan a directory for audio files organized as:
            directory/
                training/
                    real/
                    fake/
                validation/
                    real/
                    fake/
                testing/
                    real/
                    fake/

        Returns:
            dict with keys 'train', 'val', 'test', each containing
            list of (filepath, label) tuples.
        """
        splits = {
            "train": [],
            "val": [],
            "test": [],
        }

        split_dirs = {
            "train": ["training", "train"],
            "val": ["validation", "val"],
            "test": ["testing", "test", "evaluation", "eval"],
        }

        for split_name, possible_dirs in split_dirs.items():
            for dir_name in possible_dirs:
                split_path = directory / dir_name
                if split_path.exists():
                    for label_name, label in [("real", 0), ("fake", 1)]:
                        label_path = split_path / label_name
                        if label_path.exists():
                            for audio_file in label_path.iterdir():
                                if audio_file.suffix.lower() in self.AUDIO_EXTENSIONS:
                                    splits[split_name].append(
                                        (str(audio_file), label)
                                    )

        # Fallback: scan top-level real/fake directories
        if all(len(v) == 0 for v in splits.values()):
            all_files = []
            for label_name, label in [("real", 0), ("fake", 1)]:
                label_path = directory / label_name
                if label_path.exists():
                    for audio_file in label_path.rglob("*"):
                        if audio_file.suffix.lower() in self.AUDIO_EXTENSIONS:
                            all_files.append((str(audio_file), label))

            if all_files:
                filepaths, labels = zip(*all_files)
                train_fp, test_fp, train_lb, test_lb = train_test_split(
                    filepaths, labels, test_size=0.2, random_state=42, stratify=labels
                )
                train_fp, val_fp, train_lb, val_lb = train_test_split(
                    train_fp, train_lb, test_size=0.125, random_state=42, stratify=train_lb
                )
                splits["train"] = list(zip(train_fp, train_lb))
                splits["val"] = list(zip(val_fp, val_lb))
                splits["test"] = list(zip(test_fp, test_lb))

        return splits

    def get_datasets(
        self,
        version: Optional[str] = None,
        transform=None,
    ):
        """
        Get train/val/test datasets for a specific FoR version or combined.

        Args:
            version: Specific version (e.g., "for-original"). If None, uses combined.
            transform: Optional data transforms.

        Returns:
            dict with 'train', 'val', 'test' AudioDeepfakeDataset instances.
        """
        if version:
            splits = self._scan_directory(self.data_dir / version)
        elif self.use_combined:
            splits = {"train": [], "val": [], "test": []}
            for v in self.versions:
                v_splits = self._scan_directory(self.data_dir / v)
                for key in splits:
                    splits[key].extend(v_splits[key])
        else:
            raise ValueError("Specify a version or set use_combined=True")

        datasets = {}
        for split_name, file_list in splits.items():
            if file_list:
                datasets[split_name] = AudioDeepfakeDataset(
                    file_list, mode=self.mode, transform=transform
                )

        return datasets
