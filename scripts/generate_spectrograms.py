"""
Batch Mel Spectrogram Generation Script.

Generates and saves Mel spectrogram images from audio files
for faster training and evaluation.

Usage:
    python scripts/generate_spectrograms.py \
        --dataset asvspoof2019 \
        --data_dir data/ASVspoof2019/LA \
        --output_dir data/spectrograms/asvspoof2019

    python scripts/generate_spectrograms.py \
        --dataset for \
        --data_dir data/FoR \
        --output_dir data/spectrograms/for
"""

import os
import sys
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datasets.preprocessing import AudioPreprocessor, MelSpectrogramGenerator


def process_single_file(args_tuple):
    """Process a single audio file and save its spectrogram."""
    audio_path, output_path, sample_rate, duration = args_tuple

    try:
        preprocessor = AudioPreprocessor(sample_rate=sample_rate, duration=duration)
        mel_gen = MelSpectrogramGenerator(sample_rate=sample_rate)

        waveform = preprocessor.load_audio(audio_path)
        mel_gen.save_spectrogram(waveform, output_path)
        return True, audio_path
    except Exception as e:
        return False, f"{audio_path}: {e}"


def main():
    parser = argparse.ArgumentParser(description="Generate Mel Spectrograms")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["asvspoof2019", "for", "custom"])
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect audio files
    tasks = []

    if args.dataset == "asvspoof2019":
        data_dir = Path(args.data_dir)
        for split in ["ASVspoof2019_LA_train", "ASVspoof2019_LA_dev", "ASVspoof2019_LA_eval"]:
            flac_dir = data_dir / split / "flac"
            if flac_dir.exists():
                split_out = output_dir / split
                split_out.mkdir(parents=True, exist_ok=True)
                for f in flac_dir.glob("*.flac"):
                    out_path = str(split_out / f"{f.stem}.png")
                    tasks.append((str(f), out_path, args.sample_rate, args.duration))

    elif args.dataset == "for":
        data_dir = Path(args.data_dir)
        for version_dir in data_dir.iterdir():
            if version_dir.is_dir():
                for audio_file in version_dir.rglob("*"):
                    if audio_file.suffix.lower() in {".wav", ".flac", ".mp3", ".ogg"}:
                        rel_path = audio_file.relative_to(data_dir)
                        out_path = str(output_dir / rel_path.with_suffix(".png"))
                        os.makedirs(os.path.dirname(out_path), exist_ok=True)
                        tasks.append((str(audio_file), out_path, args.sample_rate, args.duration))

    elif args.dataset == "custom":
        data_dir = Path(args.data_dir)
        for audio_file in data_dir.rglob("*"):
            if audio_file.suffix.lower() in {".wav", ".flac", ".mp3", ".ogg"}:
                rel_path = audio_file.relative_to(data_dir)
                out_path = str(output_dir / rel_path.with_suffix(".png"))
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                tasks.append((str(audio_file), out_path, args.sample_rate, args.duration))

    print(f"Found {len(tasks)} audio files to process")
    print(f"Output directory: {output_dir}")

    # Process with multiprocessing
    success_count = 0
    fail_count = 0

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_single_file, t): t for t in tasks}

        for future in tqdm(as_completed(futures), total=len(tasks), desc="Generating spectrograms"):
            success, info = future.result()
            if success:
                success_count += 1
            else:
                fail_count += 1
                if fail_count <= 10:
                    print(f"  Failed: {info}")

    print(f"\nDone! Success: {success_count}, Failed: {fail_count}")


if __name__ == "__main__":
    main()
