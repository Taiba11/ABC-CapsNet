# Detailed Experimental Results

## ASVspoof 2019 (LA) — Per-Attack Analysis

### Accuracy (%)

| Attack | Type | Accuracy (%) |
|--------|------|-------------|
| A07 | TTS (vocoder) | 97.0 |
| A08 | TTS (neural) | 97.2 |
| A09 | TTS (waveform) | 97.5 |
| A10 | VC (neural) | 96.8 |
| A11 | VC (spectral) | 97.1 |
| A12 | VC (vocoder) | 97.3 |
| A13 | TTS/VC hybrid | 96.5 |
| A14 | TTS (GAN-based) | 97.0 |
| A15 | VC (neural source-filter) | 96.7 |
| A16 | TTS (WaveRNN) | 97.4 |
| A17 | VC (autoencoder) | 95.5 |
| A18 | TTS (advanced neural) | 96.0 |
| A19 | VC (advanced neural) | 96.2 |
| **Full LA** | **All attacks** | **98.1** |

### EER (%)

| Attack | EER (%) |
|--------|---------|
| A07 | 0.14 |
| A08 | 0.11 |
| A09 | 0.10 |
| A10 | 0.20 |
| A11 | 0.15 |
| A12 | 0.12 |
| A13 | 0.30 |
| A14 | 0.18 |
| A15 | 0.25 |
| A16 | 0.41 |
| A17 | 0.38 |
| A18 | 1.36 |
| A19 | 1.20 |
| **Full LA** | **0.06** |

**Note:** Attacks A18 and A19 present the greatest challenge, likely due to their use of advanced neural synthesis methods that produce more realistic audio.

---

## FoR Dataset — Per-Version Results

| Version | Description | Accuracy (%) | EER (%) |
|---------|------------|-------------|---------|
| for-original | Raw TTS + real speech | 97.3 | 0.09 |
| for-norm | Volume-normalized | 98.8 | 0.07 |
| for-2seconds | 2-second clips | 98.0 | 0.13 |
| for-rerecorded | Re-recorded in real environment | 96.3 | 2.20 |
| **Combined** | **All versions** | **99.0** | **0.04** |

**Note:** The for-rerecorded version shows the highest EER due to the added complexity of real-world acoustic conditions.

---

## Comparison with State-of-the-Art (Capsule-based Methods)

| Study | Dataset | Architecture | EER (%) |
|-------|---------|-------------|---------|
| Luo et al. (ICASSP 2021) | ASVspoof2019 | Capsule Network | 1.07 |
| Mao et al. (FCS 2021) | ASVspoof2019 | MFCC Capsule | 9.21 |
| Mao et al. (FCS 2021) | ASVspoof2019 | CQCC Capsule | 5.09 |
| **ABC-CapsNet (Ours)** | **FoR** | **ABC-CapsNet** | **0.04** |
| **ABC-CapsNet (Ours)** | **ASVspoof2019** | **ABC-CapsNet** | **0.06** |

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch Size | 32 |
| Epochs | 100 |
| Optimizer | Adam |
| Learning Rate | 0.0001 |
| Loss Function | Margin Loss + Cross-Entropy |
| Mel Spectrogram Size | 224 × 224 × 3 |
| n_fft | 2048 |
| Hop Length | 512 |
| Number of Mel Bands | 224 |
| Routing Iterations | 3 |
