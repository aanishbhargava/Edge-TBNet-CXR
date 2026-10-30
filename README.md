# Edge-TBNet: Lightweight Attention-Augmented Network for Explainable TB CXR Screening

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Official implementation of **Edge-TBNet**, a sub-1M parameter attention-augmented deep convolutional neural network designed for rapid, explainable screening of tuberculosis-consistent radiographic manifestations on frontal chest X-rays.

---

## Key Performance Benchmarks (Locked Test Set N=120)

| Model | Parameters | Size (MB) | CPU Latency (ms) | ROC-AUC | Sensitivity | Specificity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Edge-TBNet (Proposed)** | **0.97M** | **11.37 MB** | **19.7 +/- 3.3 ms** | **0.9341** | **71.19%** | **91.80%** |
| Vanilla MobileNetV3 | 0.93M | 3.67 MB | 17.7 +/- 3.9 ms | 0.9366 | 69.49% | 91.80% |
| EfficientNet-B0 | 4.01M | 15.58 MB | 57.5 +/- 10.8 ms | 0.9719 | 89.83% | 88.52% |
| ResNet-18 | 11.18M | 42.71 MB | 56.5 +/- 4.0 ms | 0.9444 | 71.19% | 95.08% |
| DenseNet-121 | 6.95M | 27.11 MB | 105.3 +/- 11.0 ms | 0.9775 | 93.22% | 93.44% |

---

## Repository Structure

```text
Edge-TBNet-CXR/
├── configs/
│   └── default.yaml
├── src/
│   ├── model.py
│   └── dataset.py
├── results/
│   ├── figures/
│   ├── tables/
│   └── checkpoints/
├── requirements.txt
└── README.md
```

## Setup & Quickstart

```bash
git clone [https://github.com/anish-bhargava/Edge-TBNet-CXR.git](https://github.com/anish-bhargava/Edge-TBNet-CXR.git)
cd Edge-TBNet-CXR
pip install -r requirements.txt
```

## Research Disclaimer
This software repository represents an academic research prototype evaluated retrospectively on public CXR cohorts (Shenzhen and Montgomery). It is strictly intended for scientific benchmarking and triage research, and is not certified as an autonomous clinical diagnostic system.
