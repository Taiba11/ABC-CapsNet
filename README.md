<div align="center">

# ABC-CapsNet: Attention-Based Cascaded Capsule Network for Audio Deepfake Detection

[![Paper](https://img.shields.io/badge/Paper-CVPR%20Workshop%202024-blue.svg)](https://openaccess.thecvf.com/content/CVPR2024W/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-yellow.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-red.svg)](https://pytorch.org/)
[![Conference](https://img.shields.io/badge/CVPRW-2024-purple.svg)](#)

**Official implementation of the paper accepted at CVPR Workshop 2024**

[Taiba Majid Wani](mailto:majid@diag.uniroma1.it)&nbsp;&nbsp;&nbsp;
[Reeva Gulzar](mailto:gulzar.1958711@studenti.uniroma1.it)&nbsp;&nbsp;&nbsp;
[Irene Amerini](mailto:amerini@diag.uniroma1.it)

**Sapienza University of Rome, Italy**

<br>

<img src="assets/architecture.png" alt="ABC-CapsNet Architecture" width="850"/>

</div>

---

## 📋 Abstract

In response to the escalating challenge of audio deepfake detection, this study introduces **ABC-CapsNet** (Attention-Based Cascaded Capsule Network), a novel architecture that merges the perceptual strengths of Mel spectrograms with the robust feature extraction capabilities of VGG18, enhanced by a strategically placed attention mechanism. This architecture pioneers the use of **cascaded capsule networks** to delve deeper into complex audio data patterns, setting a new standard in the precision of identifying manipulated audio content.

### 🏆 Key Results

| Dataset | EER (%) | Accuracy (%) |
|---------|---------|-------------|
| **ASVspoof 2019 (LA)** | **0.06** | **98.1** |
| **FoR (Combined)** | **0.04** | **99.0** |

---

## 🔥 Highlights

- **Advanced Feature Extraction** — Integration of Mel spectrograms with VGG18 for detailed audio feature extraction
- **Attention Mechanism** — Post-feature extraction attention to highlight and prioritize key discriminative features
- **Cascaded Capsule Networks** — Two-stage capsule networks (CN1 → CN2) for deep structural analysis of audio data
- **State-of-the-Art Performance** — Achieves EER of 0.06% on ASVspoof2019 and 0.04% on FoR dataset

---

## 🏗️ Architecture

```
Audio Input
    │
    ▼
┌──────────────────┐
│  Preprocessing    │  Resample (16kHz) → Noise Reduction → Normalization → Segmentation
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Mel Spectrogram   │  224 × 224 × 3, n_fft=2048, hop=512, n_mels=224
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  VGG18 Backbone   │  16 Conv layers + 3 FC layers (pretrained ImageNet)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Attention Module  │  Softmax-weighted feature refinement
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Capsule Net 1     │  Conv → PrimaryCaps → DigitCaps (Dynamic Routing)
│     (CN1)         │
└────────┬─────────┘
         │  vⱼ
         ▼
┌──────────────────┐
│ Capsule Net 2     │  SecondaryCaps → DigitCaps (Dynamic Routing)
│     (CN2)         │
└────────┬─────────┘
         │  v'ⱼ
         ▼
┌──────────────────┐
│  Classification   │  Margin Loss → Real / Fake
└──────────────────┘
```

---

## 📁 Project Structure

```
ABC-CapsNet/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── setup.py
├── .gitignore
│
├── configs/
│   ├── default.yaml              # Default training configuration
│   ├── asvspoof2019.yaml         # ASVspoof2019 specific config
│   └── for_dataset.yaml          # FoR dataset specific config
│
├── datasets/
│   ├── __init__.py
│   ├── audio_dataset.py          # Base audio dataset class
│   ├── asvspoof2019.py           # ASVspoof2019 dataset loader
│   ├── for_dataset.py            # FoR dataset loader
│   └── preprocessing.py          # Audio preprocessing & Mel spectrogram generation
│
├── models/
│   ├── __init__.py
│   ├── abc_capsnet.py            # Full ABC-CapsNet architecture
│   ├── vgg18.py                  # VGG18 feature extractor
│   ├── attention.py              # Attention mechanism module
│   ├── capsule_network.py        # Capsule Network (CN1 & CN2)
│   ├── capsule_layers.py         # Primary & Digit capsule layers
│   └── losses.py                 # Margin loss implementation
│
├── utils/
│   ├── __init__.py
│   ├── metrics.py                # EER, accuracy, and other metrics
│   ├── logger.py                 # Training logger
│   └── visualization.py          # Spectrogram & results visualization
│
├── scripts/
│   ├── train.py                  # Training script
│   ├── evaluate.py               # Evaluation script
│   ├── inference.py              # Single-file inference
│   └── generate_spectrograms.py  # Batch spectrogram generation
│
├── assets/
│   └── architecture.png          # Architecture diagram
│
└── docs/
    └── RESULTS.md                # Detailed experimental results
```

---

## ⚙️ Installation

### Prerequisites
- Python ≥ 3.8
- PyTorch ≥ 1.12
- CUDA 11.3+ (for GPU training)

### Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/ABC-CapsNet.git
cd ABC-CapsNet

# Create virtual environment
conda create -n abccapsnet python=3.9 -y
conda activate abccapsnet

# Install dependencies
pip install -r requirements.txt

# (Optional) Install as package
pip install -e .
```

---

## 📊 Dataset Preparation

### ASVspoof 2019 (LA)

1. Download from the [official ASVspoof website](https://www.asvspoof.org/index2019.html)
2. Organize the data as follows:

```
data/
└── ASVspoof2019/
    └── LA/
        ├── ASVspoof2019_LA_train/
        ├── ASVspoof2019_LA_dev/
        ├── ASVspoof2019_LA_eval/
        └── ASVspoof2019_LA_cm_protocols/
```

### FoR Dataset

1. Download from the [FoR dataset page](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset)
2. Organize the data:

```
data/
└── FoR/
    ├── for-original/
    ├── for-norm/
    ├── for-2seconds/
    └── for-rerecorded/
```

### Generate Mel Spectrograms

```bash
# ASVspoof 2019
python scripts/generate_spectrograms.py \
    --dataset asvspoof2019 \
    --data_dir data/ASVspoof2019/LA \
    --output_dir data/spectrograms/asvspoof2019

# FoR dataset
python scripts/generate_spectrograms.py \
    --dataset for \
    --data_dir data/FoR \
    --output_dir data/spectrograms/for
```

---

## 🚀 Training

### Train on ASVspoof 2019

```bash
python scripts/train.py \
    --config configs/asvspoof2019.yaml \
    --data_dir data/spectrograms/asvspoof2019 \
    --output_dir experiments/asvspoof2019 \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.0001
```

### Train on FoR Dataset

```bash
python scripts/train.py \
    --config configs/for_dataset.yaml \
    --data_dir data/spectrograms/for \
    --output_dir experiments/for \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.0001
```

### Training with Custom Settings

```bash
python scripts/train.py \
    --config configs/default.yaml \
    --data_dir <path_to_spectrograms> \
    --output_dir <experiment_dir> \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.0001 \
    --routing_iterations 3 \
    --num_capsules 8 \
    --capsule_dim 16 \
    --gpu 0
```

---

## 📈 Evaluation

```bash
# Evaluate on ASVspoof 2019
python scripts/evaluate.py \
    --config configs/asvspoof2019.yaml \
    --checkpoint experiments/asvspoof2019/best_model.pth \
    --data_dir data/spectrograms/asvspoof2019 \
    --split eval

# Evaluate on FoR dataset
python scripts/evaluate.py \
    --config configs/for_dataset.yaml \
    --checkpoint experiments/for/best_model.pth \
    --data_dir data/spectrograms/for \
    --split test
```

### Single File Inference

```bash
python scripts/inference.py \
    --checkpoint experiments/asvspoof2019/best_model.pth \
    --audio_path path/to/audio.wav \
    --output_dir results/
```

---

## 📊 Results

### ASVspoof 2019 (LA) — Per-Attack Results

| Attack | Accuracy (%) | EER (%) |
|--------|-------------|---------|
| A07 | 97.0 | 0.14 |
| A08 | 97.2 | 0.11 |
| A09 | 97.5 | 0.10 |
| A10 | 96.8 | 0.20 |
| A11 | 97.1 | 0.15 |
| A12 | 97.3 | 0.12 |
| A13 | 96.5 | 0.30 |
| A14 | 97.0 | 0.18 |
| A15 | 96.7 | 0.25 |
| A16 | 97.4 | 0.41 |
| A17 | 95.5 | 0.38 |
| A18 | 96.0 | 1.36 |
| A19 | 96.2 | 1.20 |
| **Full LA** | **98.1** | **0.06** |

### FoR Dataset Results

| Version | Accuracy (%) | EER (%) |
|---------|-------------|---------|
| for-original | 97.3 | 0.09 |
| for-norm | 98.8 | 0.07 |
| for-2seconds | 98.0 | 0.13 |
| for-rerecorded | 96.3 | 2.20 |
| **Combined** | **99.0** | **0.04** |

### Comparison with State-of-the-Art

| Method | Dataset | Architecture | EER (%) |
|--------|---------|-------------|---------|
| Luo et al. | ASVspoof2019 | Capsule Network | 1.07 |
| Mao et al. | ASVspoof2019 | MFCC Capsule | 9.21 |
| Mao et al. | ASVspoof2019 | CQCC Capsule | 5.09 |
| **ABC-CapsNet (Ours)** | **FoR** | **ABC-CapsNet** | **0.04** |
| **ABC-CapsNet (Ours)** | **ASVspoof2019** | **ABC-CapsNet** | **0.06** |

---

## 📜 Citation

If you find this work useful in your research, please consider citing:

```bibtex
@inproceedings{wani2024abccapsnet,
    title     = {ABC-CapsNet: Attention based Cascaded Capsule Network for Audio Deepfake Detection},
    author    = {Wani, Taiba Majid and Gulzar, Reeva and Amerini, Irene},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
    year      = {2024},
    pages     = {2464--2472}
}
```

---

## 🙏 Acknowledgments

This study has been partially supported by:
- **SERICS** (PE00000014) under the MUR National Recovery and Resilience Plan funded by the European Union – NextGenerationEU
- **Sapienza University of Rome** project 2022–2024 "EV2" (003 009 22)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**⭐ If you find this repository helpful, please consider giving it a star! ⭐**

</div>
