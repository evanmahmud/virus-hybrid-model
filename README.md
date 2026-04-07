# 🦠 Virus Particle Detection & Classification
### Hybrid Multi-Detector + Ensemble Classification System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green)
![Platform](https://img.shields.io/badge/Platform-Kaggle%20P100-orange?logo=kaggle)

A deep learning pipeline for **22-class virus particle classification** from electron microscopy images, combining 4 CNN classifiers, 4 Transformer classifiers, and 4 object detection models into a unified hybrid inference system.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Models](#models)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Key Design Decisions](#key-design-decisions)

---

## Overview

This project tackles the problem of identifying 22 different virus species from raw `.tif` electron microscopy images. The pipeline has three stages:

1. **Detection** — Four detectors localize virus particles via bounding boxes
2. **Classification** — Eight classifiers (4 CNN + 4 Transformer) form a weighted ensemble
3. **Hybrid fusion** — Detection confidence gates how much spatial (crop-level) signal blends with the full-image ensemble

The key insight is that classifiers must always run on **full images** (their trained domain), while detectors provide supplementary spatial signal — not a replacement signal.

---

## Architecture

```
Input Image (variable resolution .tif)
│
├── ── DETECTION BRANCH ──────────────────────────────────────────┐
│    ├── Faster R-CNN (ResNet50-FPN)                              │
│    ├── RetinaNet   (ResNet50-FPN)                               │
│    ├── FCOS        (ResNet50-FPN, anchor-free)                  │
│    └── DETR        (ResNet50 + Transformer, fp32-forced)        │
│         └── NMS Box Fusion → fused_boxes, det_confidence        │
│                                                                 │
├── ── CLASSIFICATION BRANCH ───────────────────────────────────┐ │
│    CNN Models:                                                │ │
│    ├── ResNet50                                               │ │
│    ├── EfficientNetV2-S                                       │ │
│    ├── ConvNeXt-Small                                         │ │
│    └── DenseNet121                                            │ │
│    Transformer Models:                                        │ │
│    ├── ViT-Base/16                                            │ │
│    ├── Swin-Small                                             │ │
│    ├── DeiT-Base/16                                           │ │
│    └── MaxViT-Tiny                                            │ │
│         └── Weighted Soft Voting → full_image_probs           │ │
│                                                               │ │
└── ── HYBRID FUSION ───────────────────────────────────────────┘─┘
          alpha  = 1.0 - 0.30 × det_confidence
          final  = alpha × full_image_probs
                 + (1 - alpha) × crop_probs
          pred   = argmax(final)
```

---

## Dataset

**Context Virus RAW** dataset from Kaggle:
- **22 virus classes** (e.g., TBE, HIV, Adenovirus, Influenza, …)
- Images: variable-resolution `.tif` electron microscopy scans
- Particle positions provided as `_particlepositions.txt` files (x;y center coordinates)
- Bounding boxes generated from centers with a fixed `box_size=64` px

```
/kaggle/input/virus-images/context_virus_RAW/
├── train/
│   └── <ClassName>/
│       ├── <image>.tif
│       └── particle_positions/
│           └── <image>_particlepositions.txt
└── validation/
    └── <ClassName>/
        └── ...
```

---

## Models

### Classifiers (trained with AMP + cosine LR + class-weighted CE loss)

| Model | Type | Val Accuracy | Val F1 |
|---|---|---|---|
| Swin-Small | Transformer | **89.9%** | 0.851 |
| MaxViT-Tiny | Transformer | 88.7% | 0.845 |
| DeiT-Base | Transformer | 86.7% | 0.831 |
| ConvNeXt-Small | CNN | 85.5% | 0.796 |
| ViT-Base/16 | Transformer | 84.3% | 0.761 |
| DenseNet121 | CNN | 80.2% | 0.763 |
| EfficientNetV2-S | CNN | 79.8% | 0.737 |
| ResNet50 | CNN | 66.1% | 0.609 |
| **Full Ensemble (Equal)** | Ensemble | 89.5% | 0.846 |
| **Full Ensemble (Weighted)** | Ensemble | 89.5% | 0.846 |

### Detectors (trained with AdamW + StepLR, mAP evaluated)

| Model | Type | Notes |
|---|---|---|
| Faster R-CNN | Two-stage | ResNet50-FPN backbone |
| RetinaNet | Single-stage | Focal loss, ResNet50-FPN |
| FCOS | Anchor-free | ResNet50-FPN |
| DETR | Transformer | HuggingFace `facebook/detr-resnet-50`, forced fp32 |

---

## Results

### Individual & Ensemble Classifiers

| Model | Accuracy | Macro F1 |
|---|---|---|
| Swin (best single) | 89.9% | 0.851 |
| Full Weighted Ensemble | 89.5% | 0.846 |

### Hybrid Multi-Detector + Classification

| Metric | Score |
|---|---|
| Accuracy | **~90%** (target) |
| Macro Precision | — |
| Macro Recall | — |
| Macro F1 | — |
| Macro AUC | — |

> Results update after full hybrid training run.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/virus-hybrid-detection.git
cd virus-hybrid-detection

# Install dependencies
pip install torch torchvision timm transformers torchmetrics \
            scikit-learn tifffile seaborn matplotlib tqdm pandas
```

**Requirements:**
- Python ≥ 3.10
- PyTorch ≥ 2.0
- CUDA GPU recommended (tested on Kaggle P100 16 GB)

---

## Usage

All code runs top-to-bottom in a single notebook. Open `hybrid_multi_detection.ipynb` on Kaggle or locally.

### Step-by-step

**1. Setup**
```python
# Config block sets paths and model weights
TRAIN_ROOT = "/kaggle/input/virus-images/context_virus_RAW/train"
VAL_ROOT   = "/kaggle/input/virus-images/context_virus_RAW/validation"
```

**2. Train classifiers** *(skip if `.pth` files already exist)*
```python
# Trains one model at a time, saves best checkpoint, frees GPU
for name in CLS_MODELS:
    train_classifier(name, epochs=15)
```

**3. Train detectors** *(skip if `.pth` files already exist)*
```python
# DETR automatically disables AMP to avoid fp16 NaN in Hungarian matcher
for det_name, factory in DET_FACTORIES.items():
    train_detector(factory(), det_name, epochs=10)
```

**4. Load all models**
```python
# Classifiers loaded in fp16 to save VRAM
# Detectors loaded in fp32
classifiers = { name: load_cls(name) for name in CLS_MODELS }
detectors   = { name: load_det(name) for name in DET_FACTORIES }
```

**5. Run hybrid evaluation**
```python
y_true, y_pred, y_probs = evaluate_hybrid(
    detectors, classifiers, MODEL_WEIGHTS, val_cls_loader, device)
```

**6. Visualizations are saved automatically to `results/hybrid_multi/`**

---

## Project Structure

```
virus-hybrid-detection/
│
├── hybrid_multi_detection.ipynb   # Main notebook (self-contained, run top-to-bottom)
├── README.md
│
├── results/
│   └── hybrid_multi/
│       ├── metrics_summary.csv
│       ├── metrics_dashboard.png
│       ├── confusion_matrix.png
│       ├── confusion_matrix.csv
│       ├── classification_report.txt
│       ├── roc_curves.png
│       ├── pr_curves.png
│       ├── per_class_auc.png
│       └── per_class_f1.png
│
├── <ModelName>_best.pth           # Saved classifier checkpoints
└── <DetectorName>_det_best.pth   # Saved detector checkpoints
```

---

## Key Design Decisions

### 1. One model at a time training
All 8 classifiers and 4 detectors are trained sequentially, with `del model` + `torch.cuda.empty_cache()` between each. This is the only way to fit the full suite inside 16 GB VRAM without out-of-memory errors.

### 2. FP16 classifiers at inference
Classifiers are loaded as `.half()` (fp16), halving their VRAM footprint (~50% reduction) with negligible accuracy loss (<0.1%). Detectors stay in fp32.

### 3. DETR forced to fp32
DETR's Hungarian matcher computes generalised IoU inside a scipy-style optimal assignment. Under AMP fp16, intermediate attention values underflow to `NaN` before the matcher runs, crashing training. `autocast(enabled=False)` forces fp32 for DETR while all other detectors retain AMP.

### 4. Full-image classification (not crops)
Classifiers were trained on full 224×224 microscopy images. Running them on individual particle crops (~64×64, out-of-domain) collapses accuracy from 90% → 24%. The hybrid design always uses the full image as the primary signal; detection boxes only contribute a secondary blended signal (max 30% weight, gated by detection confidence).

### 5. NMS box fusion across detectors
Boxes from all 4 detectors are pooled and deduplicated with `torchvision.ops.nms(iou=0.5)` before any crop processing, eliminating redundant crops from overlapping predictions.

### 6. Class imbalance handling
Inverse-frequency class weights are passed to `CrossEntropyLoss` during classifier training, preventing majority-class bias across the 22 virus classes.

---

## License

MIT License — free to use, modify, and distribute with attribution.

---

## Acknowledgements

- Dataset: [Context Virus RAW — Kaggle](https://www.kaggle.com/datasets)
- Pretrained models: [timm](https://github.com/huggingface/pytorch-image-models) · [HuggingFace Transformers](https://github.com/huggingface/transformers) · [TorchVision](https://github.com/pytorch/vision)
- DETR: [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872) — Carion et al., 2020
