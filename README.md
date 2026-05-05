# 🕐 Analog Clock Reader

A deep learning system that reads analog clock faces from photographs and predicts the displayed time as `hh:mm`. Built with **MATLAB R2026a** using a **two-head ResNet-18** architecture trained on the [TickTockVQA](https://huggingface.co/datasets/jaeha-choi/TickTockVQA) dataset.

---

## ✨ Features

- **Two-head CNN architecture** — separate classification heads for hour (12 classes) and minute (60 classes)
- **Transfer learning** from ImageNet-pretrained ResNet-18
- **Custom training loop** with Adam optimizer, learning rate decay, and early stopping
- **Works on any clock style** — wall clocks, wristwatches, tower clocks, desk clocks
- **Circular time error metric** — handles 12-hour wrap-around correctly (11:55 → 12:05 = 10 min, not 710)
- **Auto GPU/CPU detection** — runs on NVIDIA CUDA GPUs or falls back to CPU gracefully

---

## 🏗️ Architecture

```
Input Image [224×224×3]
        │
┌───────▼────────────────────┐
│   ResNet-18 Backbone       │ ← Frozen (ImageNet weights)
│   conv1 → res2 → ... → res5│
│   Global Average Pooling    │
│         [512-dim]           │
└───────┬──────────┬─────────┘
        │          │
   ┌────▼────┐ ┌──▼──────┐
   │  Hour   │ │ Minute  │
   │  Head   │ │  Head   │
   │ FC(256) │ │ FC(256) │
   │ BN+ReLU │ │ BN+ReLU │
   │Drop(0.4)│ │Drop(0.4)│
   │ FC(12)  │ │ FC(60)  │
   │ Softmax │ │ Softmax │
   └────┬────┘ └──┬──────┘
        │          │
   Hour 1-12  Minute 0-59
        │          │
        └────┬─────┘
         hh:mm
```

**Why two heads?** Hour and minute hands provide semi-independent visual cues. Separate heads give cleaner gradients and avoid the sparse supervision problem of a single 720-class output.

---

## 📁 Project Structure

```
analog-clock-reader/
├── main.m                     # 🎯 Main pipeline — run this (7 sections, Ctrl+Enter)
├── download_dataset.m         # Dataset download via huggingface-cli
├── load_annotations.m         # JSON → MATLAB table parser
├── create_datastores.m        # Train/val/test split + augmentation
├── build_clock_model.m        # Two-head ResNet-18 construction
├── train_clock_model.m        # Custom training loop (Adam + early stopping)
├── evaluate_model.m           # MATE, confusion matrices, histograms
├── predict_time.m             # Single-image inference + visualization
│
├── utils/
│   ├── circular_time_error.m  # Circular 12-hour time distance
│   ├── freeze_layers.m        # Backbone layer freezing
│   └── preprocess_clock_image.m
│
├── download_dataset.py        # Python download (handles Windows filename issues)
├── fix_missing_images.py      # Recovery for files with illegal characters
├── audit_dataset.py           # Dataset integrity checker
│
├── Documentation/
│   ├── Documentation.md       # Full project documentation
│   └── Diagrams.md            # 13 Mermaid diagrams (architecture, flows, etc.)
│
├── data/dataset/              # ⬇️ Downloaded separately (not in repo)
│   ├── annotations.json
│   └── images/{train,test}/
│
└── results/                   # 📊 Generated during training (not in repo)
    ├── trained_model.mat
    ├── training_log.mat
    └── figures/*.png
```

---

## 📋 Requirements

### MATLAB (R2022b or newer)

| Toolbox | Required |
|---------|----------|
| Deep Learning Toolbox | ✅ |
| Computer Vision Toolbox | ✅ |
| Image Processing Toolbox | ✅ |
| Statistics and Machine Learning Toolbox | ✅ |

### Python (for dataset download only)

```
Python 3.8+
huggingface_hub
```

### Hardware

| | CPU | NVIDIA GPU |
|---|---|---|
| Training (30 epochs) | 6–12 hours | 30–60 minutes |
| Inference (1 image) | < 1 second | < 0.1 second |

> ⚠️ **GPU Note:** MATLAB requires NVIDIA GPUs with CUDA. AMD GPUs are not supported. The code auto-detects via `canUseGPU()` and falls back to CPU.

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/analog-clock-reader.git
cd analog-clock-reader
```

### 2. Download the dataset (3.7 GB)

```bash
pip install -U huggingface_hub
python download_dataset.py
```

If you encounter Windows filename errors:
```bash
python fix_missing_images.py
```

Verify the download:
```bash
python audit_dataset.py
# Expected: 12,483/12,483 images (100%)
```

### 3. Run in MATLAB

```matlab
cd path/to/analog-clock-reader
edit main.m
% Run each section with Ctrl+Enter
```

| Section | What it does | Time |
|---------|-------------|------|
| 1. Setup | Configure paths, check GPU | Instant |
| 2. Download | Verify dataset exists | ~2 sec |
| 3. Annotations | Parse JSON → MATLAB table | ~10 sec |
| 4. Datastores | Build data pipelines + augmentation | ~20 sec |
| 5. Build Model | Construct two-head ResNet-18 | ~15 sec |
| 6. **Train** | **Custom training loop** | **6–12h CPU / 30–60min GPU** |
| 7. Evaluate | Test metrics + visualizations | ~10 min |

### 4. Predict on your own image

```matlab
addpath('utils');
loaded = load('results/trained_model.mat');
[timeStr, conf] = predict_time(loaded.net, 'your_clock_photo.jpg');
% Displays: predicted time, confidence bars, top-3 predictions
```

---

## 📊 Dataset

**[TickTockVQA](https://huggingface.co/datasets/jaeha-choi/TickTockVQA)** — 12,483 labeled analog clock images from the paper *"It's Time to Get It Right"* (CVPR 2026 Findings).

| Split | Images | Purpose |
|-------|--------|---------|
| Train | 6,150 | Model training (with augmentation) |
| Validation | 1,086 | Early stopping & checkpointing |
| Test | 5,247 | Final evaluation |

**Sources:** OpenImages, COCO, ClockMovies, VisualGenome, CC12M, ImageNet, SBU

**Clock types:** Wall clocks, Wristwatches, Tower clocks, Alarm/Desk clocks, Post clocks

---

## 🎯 Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Hour Accuracy | % correct hour predictions | > 85% |
| Minute Accuracy | % exact minute match | > 60% |
| **MATE** | **Mean Absolute Time Error (circular, minutes)** | **< 10 min** |
| Within 5 min | % predictions within 5 min of truth | > 70% |

### Circular Time Error

Standard absolute difference fails at clock boundaries. Our MATE metric uses modular distance on a 720-minute circle:

```
error = min(|diff|, 720 - |diff|)
```

| Predicted | True | Naive Error | MATE |
|-----------|------|-------------|------|
| 12:05 | 11:55 | 710 min ❌ | **10 min** ✅ |
| 03:42 | 03:45 | 3 min | **3 min** |

---

## 🔧 Training Details

| Hyperparameter | Value |
|----------------|-------|
| Backbone | ResNet-18 (ImageNet pretrained, frozen) |
| Optimizer | Adam |
| Initial Learning Rate | 1e-4 |
| LR Schedule | Step decay ×0.5 every 10 epochs |
| Batch Size | 32 |
| Max Epochs | 30 |
| Early Stopping | Patience = 5 (on validation loss) |
| Dropout | 0.4 |
| Loss | CrossEntropy(hour) + CrossEntropy(minute) |

### Data Augmentation (training only)

| Transform | Range | Note |
|-----------|-------|------|
| Rotation | [-15°, +15°] | Slight tilt variation |
| Brightness | [-20, +20] | Lighting conditions |
| Contrast | [0.8×, 1.2×] | Indoor/outdoor |
| **Flip** | **Disabled** | **Flipping mirrors the clock → changes time** |

---

## 📖 Documentation

Detailed documentation is available in the [`Documentation/`](Documentation/) folder:

- **[Documentation.md](Documentation/Documentation.md)** — Full technical documentation: architecture, algorithms, code walkthrough, installation, troubleshooting
- **[Diagrams.md](Documentation/Diagrams.md)** — 13 Mermaid diagrams: system architecture, pipeline flow, ResNet-18 layers, data flow, training sequence, loss computation, evaluation metrics, file dependencies, and more

---

## 📚 References

- **Dataset:** Choi et al., *"It's Time to Get It Right: Improving Analog Clock Reading and Clock-Hand Spatial Reasoning in Vision-Language Models,"* CVPR 2026 Findings. [arXiv:2603.08011](https://arxiv.org/abs/2603.08011)
- **ResNet-18:** He et al., *"Deep Residual Learning for Image Recognition,"* CVPR 2016
- **Adam:** Kingma & Ba, *"Adam: A Method for Stochastic Optimization,"* ICLR 2015

---

## 📄 License

This project is for **academic/educational use**. The TickTockVQA dataset has mixed licensing — see the [dataset page](https://huggingface.co/datasets/jaeha-choi/TickTockVQA) for details.
