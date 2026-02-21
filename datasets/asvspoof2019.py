"""
ASVspoof 2019 (LA) Dataset Loader.

Parses the official ASVspoof 2019 protocol files and creates
training, development, and evaluation splits.

LA attacks:
    - Training: A01-A06
    - Evaluation: A07-A19

Labels:
    - bonafide → 0 (real)
    - spoof → 1 (fake)
"""

import os
from pathlib import Path
from typing import Optional

from .audio_dataset import AudioDeepfakeDataset


class ASVspoof2019Dataset:
    """
    ASVspoof 2019 Logical Access (LA) dataset handler.

    Args:
        data_dir (str): Root directory of ASVspoof2019 LA data.
        spectrogram_dir (str): Directory with precomputed spectrograms (optional).
        mode (str): "audio" for raw files, "spectrogram" for precomputed images.
    """

    LABEL_MAP = {"bonafide": 0, "spoof": 1}

    def __init__(
        self,
        data_dir: str,
        spectrogram_dir: Optional[str] = None,
        mode: str = "audio",
    ):
        self.data_dir = Path(data_dir)
        self.spectrogram_dir = Path(spectrogram_dir) if spectrogram_dir else None
        self.mode = mode

    def _parse_protocol(self, protocol_path: str, audio_dir: str):
        """
        Parse an ASVspoof protocol file.

        Protocol format:
            SPEAKER_ID AUDIO_FILENAME - ATTACK_TYPE LABEL

        Returns:
            list of (filepath, label) tuples.
        """
        file_list = []
        with open(protocol_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    speaker_id = parts[0]
                    audio_name = parts[1]
                    attack_type = parts[3]
                    label_str = parts[4]

                    label = self.LABEL_MAP.get(label_str, 1)

                    if self.mode == "spectrogram" and self.spectrogram_dir:
                        filepath = self.spectrogram_dir / f"{audio_name}.png"
                    else:
                        filepath = Path(audio_dir) / f"{audio_name}.flac"

                    if filepath.exists():
                        file_list.append((str(filepath), label))

        return file_list

    def get_train_dataset(self, transform=None):
        """Get training dataset."""
        protocol = self.data_dir / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.train.trn.txt"
        audio_dir = self.data_dir / "ASVspoof2019_LA_train" / "flac"
        file_list = self._parse_protocol(str(protocol), str(audio_dir))
        return AudioDeepfakeDataset(file_list, mode=self.mode, transform=transform)

    def get_dev_dataset(self, transform=None):
        """Get development dataset."""
        protocol = self.data_dir / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.dev.trl.txt"
        audio_dir = self.data_dir / "ASVspoof2019_LA_dev" / "flac"
        file_list = self._parse_protocol(str(protocol), str(audio_dir))
        return AudioDeepfakeDataset(file_list, mode=self.mode, transform=transform)

    def get_eval_dataset(self, transform=None, attack_type: Optional[str] = None):
        """
        Get evaluation dataset, optionally filtered by attack type.

        Args:
            transform: Optional data transforms.
            attack_type: If specified, only include samples from this attack
                        (e.g., "A07", "A08", ..., "A19").
        """
        protocol = self.data_dir / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.eval.trl.txt"
        audio_dir = self.data_dir / "ASVspoof2019_LA_eval" / "flac"
        file_list = self._parse_protocol(str(protocol), str(audio_dir))

        if attack_type:
            # Filter by attack type using protocol file
            filtered = []
            with open(str(protocol), "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        audio_name = parts[1]
                        atk = parts[3]
                        label_str = parts[4]
                        if atk == attack_type or label_str == "bonafide":
                            label = self.LABEL_MAP.get(label_str, 1)
                            if self.mode == "spectrogram" and self.spectrogram_dir:
                                fp = self.spectrogram_dir / f"{audio_name}.png"
                            else:
                                fp = Path(str(audio_dir)) / f"{audio_name}.flac"
                            if fp.exists():
                                filtered.append((str(fp), label))
            file_list = filtered

        return AudioDeepfakeDataset(file_list, mode=self.mode, transform=transform)
