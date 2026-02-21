"""
Audio Preprocessing & Mel Spectrogram Generation for ABC-CapsNet (Section 3.1).

Pipeline:
    1. Resample to 16 kHz
    2. Noise reduction
    3. Amplitude normalization to [-1, 1]
    4. Silence removal
    5. Mel spectrogram generation (224 × 224 × 3)
"""

import numpy as np
import torch
import torchaudio
import librosa
import warnings

from PIL import Image

warnings.filterwarnings("ignore")


class AudioPreprocessor:
    """
    Audio preprocessing pipeline as described in Section 3.1.

    Args:
        sample_rate (int): Target sample rate (default: 16000).
        duration (float): Target duration in seconds for padding/truncation.
        noise_reduction (bool): Whether to apply noise reduction.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        duration: float = 4.0,
        noise_reduction: bool = True,
    ):
        self.sample_rate = sample_rate
        self.duration = duration
        self.noise_reduction = noise_reduction
        self.target_length = int(sample_rate * duration)

    def load_audio(self, filepath: str):
        """
        Load and preprocess an audio file.

        Args:
            filepath: Path to the audio file.

        Returns:
            waveform: Preprocessed 1D numpy array.
        """
        # Load audio
        waveform, sr = torchaudio.load(filepath)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        waveform = waveform.squeeze(0).numpy()

        # Step 1: Resample to target sample rate
        if sr != self.sample_rate:
            waveform = librosa.resample(
                waveform, orig_sr=sr, target_sr=self.sample_rate
            )

        # Step 2: Noise reduction
        if self.noise_reduction:
            waveform = self._reduce_noise(waveform)

        # Step 3: Amplitude normalization to [-1, 1]
        waveform = self._normalize(waveform)

        # Step 4: Silence removal
        waveform = self._remove_silence(waveform)

        # Step 5: Pad or truncate to target length
        waveform = self._pad_or_truncate(waveform)

        return waveform

    def _reduce_noise(self, waveform):
        """Simple spectral gating noise reduction."""
        try:
            import noisereduce as nr
            return nr.reduce_noise(y=waveform, sr=self.sample_rate)
        except ImportError:
            # Fallback: basic high-pass filter
            from scipy.signal import butter, filtfilt
            b, a = butter(4, 80 / (self.sample_rate / 2), btype="high")
            return filtfilt(b, a, waveform).astype(np.float32)

    def _normalize(self, waveform):
        """Normalize amplitude to [-1, 1]."""
        max_val = np.abs(waveform).max()
        if max_val > 0:
            waveform = waveform / max_val
        return waveform

    def _remove_silence(self, waveform, threshold_db: float = 30.0):
        """Remove leading/trailing silence."""
        intervals = librosa.effects.split(
            waveform, top_db=threshold_db
        )
        if len(intervals) > 0:
            waveform = np.concatenate(
                [waveform[start:end] for start, end in intervals]
            )
        return waveform

    def _pad_or_truncate(self, waveform):
        """Pad with zeros or truncate to target length."""
        if len(waveform) > self.target_length:
            waveform = waveform[: self.target_length]
        elif len(waveform) < self.target_length:
            padding = self.target_length - len(waveform)
            waveform = np.pad(waveform, (0, padding), mode="constant")
        return waveform


class MelSpectrogramGenerator:
    """
    Mel Spectrogram Generator (Section 3.1).

    Generates 224×224×3 Mel spectrogram images using the Hanning window.

    The Mel scale transformation:
        m = 2595 * log10(1 + f/700)

    Args:
        sample_rate (int): Audio sample rate.
        n_fft (int): FFT window size (default: 2048).
        hop_length (int): Hop length (default: 512).
        n_mels (int): Number of Mel filter banks (default: 224).
        image_size (tuple): Output image size (default: (224, 224)).
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 224,
        image_size: tuple = (224, 224),
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.image_size = image_size

    def generate(self, waveform):
        """
        Generate a Mel spectrogram image from a waveform.

        Args:
            waveform: 1D numpy array of audio samples.

        Returns:
            mel_image: (3, H, W) torch tensor — RGB Mel spectrogram image.
        """
        # Compute Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            window="hann",
        )

        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize to [0, 255] for image representation
        mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (
            mel_spec_db.max() - mel_spec_db.min() + 1e-8
        )
        mel_spec_uint8 = (mel_spec_norm * 255).astype(np.uint8)

        # Resize to target image size
        img = Image.fromarray(mel_spec_uint8)
        img = img.resize(self.image_size, Image.BILINEAR)

        # Convert to 3-channel RGB (replicate across channels)
        img_array = np.array(img)
        if img_array.ndim == 2:
            img_array = np.stack([img_array] * 3, axis=-1)

        # Convert to torch tensor: (H, W, 3) → (3, H, W), normalized to [0, 1]
        mel_image = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0

        return mel_image

    def save_spectrogram(self, waveform, save_path: str):
        """Generate and save Mel spectrogram as an image file."""
        mel_spec = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            window="hann",
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (
            mel_spec_db.max() - mel_spec_db.min() + 1e-8
        )
        mel_spec_uint8 = (mel_spec_norm * 255).astype(np.uint8)

        img = Image.fromarray(mel_spec_uint8)
        img = img.resize(self.image_size, Image.BILINEAR)

        # Save as RGB
        img_rgb = img.convert("RGB")
        img_rgb.save(save_path)
