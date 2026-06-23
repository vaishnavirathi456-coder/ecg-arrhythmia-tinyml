# ECG Arrhythmia Classification using TinyML

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.12-orange.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-yellow.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-brightgreen.svg)

End-to-end deep learning pipeline for cardiac arrhythmia classification from ECG signals — trained in PyTorch, deployed as a quantized TFLite model targeting embedded hardware (Raspberry Pi).

**Built by:** Vaishnavi Rathi | MS ECE @ Northeastern University

---

## Results

| Model | Accuracy | F1 Score | Size |
|---|---|---|---|
| Random Forest | 100% | 100% | — |
| PyTorch 1D CNN | 97.5% | 97.5% | 2.95 MB |
| TensorFlow 1D CNN | 99.5% | 99.5% | 2.96 MB |
| TFLite quantized | 89.5% | 89.2% | 784 KB |

**Inference latency (simulated RPi 4):** ~0.75ms per beat

---

## Dataset

- MIT-BIH Arrhythmia Database (PhysioNet)
- 10 records — 21,550 heartbeats extracted
- 4 beat classes: Normal, Atrial, Ventricular, Left Bundle Branch Block
- Sampling frequency: 360 Hz
- Beat window: 180 samples (500ms)

---

## Model Architecture

Input (1, 180) — 180 ECG samples per beat
Conv1D(32, k=5) + BatchNorm + ReLU + MaxPool
Conv1D(64, k=5) + BatchNorm + ReLU + MaxPool
Conv1D(128, k=3) + BatchNorm + ReLU + MaxPool
Flatten → Dense(256) → Dropout(0.3)
Dense(64) → Dropout(0.3)
Dense(4) — 4 class output

Parameters: 773,508
Quantized size: 784 KB (74% reduction)

---

## Project Structure

ml-journey/
├── week1/  — NumPy, Pandas, Matplotlib basics
├── week2/  — Heart disease EDA and ML models
├── week3/  — PyTorch neural network
├── week4/  — ECG data loading and preprocessing
├── week5/  — 1D CNN training 97.5% accuracy
├── week6/  — TFLite conversion and quantization
├── week7/  — Raspberry Pi deployment simulation
└── week8/  — Documentation and paper outline

---

## Setup

pip install torch torchvision tensorflow wfdb numpy pandas matplotlib scikit-learn

---

## Pipeline

Raw ECG Signal (MIT-BIH)
→ Beat Extraction (180 samples, 500ms window)
→ Normalization (min-max per beat)
→ 1D CNN Training (PyTorch)
→ TensorFlow Equivalent Model
→ TFLite Conversion and int8 Quantization
→ Embedded Deployment (Raspberry Pi 4)

---

## Related Work

toggletest-hw — Hardware testing library
PyPI: pypi.org/project/toggletest-hw
GitHub: github.com/vaishnavirathi456-coder/toggletest

---

## License

MIT