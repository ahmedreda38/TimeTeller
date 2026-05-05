# TimeTeller: Data-Driven Analog Clock Reading via Transfer Learning

A deep learning system for reading analog clock faces from photographs, predicting the displayed time as `hh:mm`. Built with **MATLAB R2026a** using a **two-head ResNet-18** architecture trained on the [TickTockVQA](https://huggingface.co/datasets/jaeha-choi/TickTockVQA) dataset (12,483 labeled images).

---

## Overview

This project addresses the problem of automated analog clock reading — a visual recognition task that requires identifying clock hand positions across diverse clock styles, lighting conditions, and viewing angles. We employ a convolutional neural network (CNN) with transfer learning from ImageNet, using a dual-head classification approach that independently predicts the hour (12 classes) and minute (60 classes) from a single input image.

### Key Contributions

- **Dual-head classification architecture** separating hour and minute prediction for cleaner gradient flow
- **Circular time error metric (MATE)** that correctly handles 12-hour wrap-around
- **Custom training loop** with Adam optimization, step-decay learning rate scheduling, and early stopping
- **Comprehensive evaluation pipeline** with confusion matrices, error distribution analysis, and per-hour breakdowns

---

## Model Architecture

```
Input Image [224 x 224 x 3]
        |
+-------v----------------------------+
|   ResNet-18 Backbone               |  <-- Frozen (ImageNet weights)
|   conv1 -> res2 -> ... -> res5     |
|   Global Average Pooling           |
|         [512-dim]                  |
+-------+--------------+------------+
        |              |
   +----v----+    +----v------+
   |  Hour   |    |  Minute   |
   |  Head   |    |   Head    |
   | FC(256) |    |  FC(256)  |
   | BN+ReLU |    |  BN+ReLU  |
   |Drop(0.4)|    | Drop(0.4) |
   | FC(12)  |    |  FC(60)   |
   | Softmax |    |  Softmax  |
   +----+----+    +----+------+
        |              |
   Hour 1-12     Minute 0-59
        |              |
        +------+-------+
            hh:mm
```

**Design rationale:** Separate classification heads allow the hour and minute branches to learn from semi-independent visual cues (short vs. long hand), avoiding the sparse supervision problem inherent in a single 720-class output. Classification is preferred over regression due to the cyclic nature of clock angles, where 0 degrees and 360 degrees represent the same position.

---

## Project Structure

```
TimeTeller/
|-- main.m                        Main pipeline (7 sections, cell-by-cell execution)
|-- download_dataset.m            Dataset acquisition via HuggingFace CLI
|-- load_annotations.m            JSON annotation parser
|-- create_datastores.m           Train/val/test split with augmentation
|-- build_clock_model.m           Two-head ResNet-18 construction
|-- train_clock_model.m           Custom training loop implementation
|-- evaluate_model.m              Test set evaluation and visualization
|-- predict_time.m                Single-image inference
|
|-- utils/
|   |-- circular_time_error.m     Circular time distance computation
|   |-- freeze_layers.m           Backbone layer freezing utility
|   +-- preprocess_clock_image.m  Image preprocessing pipeline
|
|-- download_dataset.py           Python download with filename sanitization
|-- fix_missing_images.py         Recovery script for platform-specific issues
|-- audit_dataset.py              Dataset integrity verification
|
|-- Documentation/
|   |-- Documentation.md          Full technical documentation
|   +-- Diagrams.md               Software architecture diagrams (Mermaid)
|
|-- data/dataset/                 Dataset directory (downloaded separately)
|   |-- annotations.json
|   +-- images/{train,test}/
|
+-- results/                      Training outputs (generated at runtime)
    |-- trained_model.mat
    |-- training_log.mat
    +-- figures/
```

---

## Requirements

### MATLAB (R2022b or newer)

| Toolbox | Status |
|---------|--------|
| Deep Learning Toolbox | Required |
| Computer Vision Toolbox | Required |
| Image Processing Toolbox | Required |
| Statistics and Machine Learning Toolbox | Required |

### Python (dataset download only)

```
Python 3.8+
huggingface_hub
```

### Hardware

| Configuration | Training Time (30 epochs) | Inference (per image) |
|---------------|--------------------------|----------------------|
| CPU only | 6 -- 12 hours | < 1 second |
| NVIDIA GPU (CUDA) | 30 -- 60 minutes | < 0.1 second |

> **Note:** MATLAB GPU acceleration requires NVIDIA GPUs with CUDA support. AMD GPUs are not supported by the MATLAB Parallel Computing Toolbox. The code automatically detects GPU availability via `canUseGPU()` and falls back to CPU execution.

---

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/ahmedreda38/TimeTeller.git
cd TimeTeller
```

### 2. Download the dataset (3.7 GB)

```bash
pip install -U huggingface_hub
python download_dataset.py
```

On Windows, some filenames in the dataset contain characters that are invalid on NTFS. If errors occur during download:

```bash
python fix_missing_images.py
```

To verify dataset integrity:

```bash
python audit_dataset.py
```

Expected output: `12,483/12,483 images (100.0%)`

### 3. Run the pipeline in MATLAB

```matlab
cd path/to/TimeTeller
edit main.m
```

Execute each section sequentially using **Ctrl+Enter**:

| Section | Description | Estimated Time |
|---------|-------------|----------------|
| 1. Setup | Path configuration, GPU detection | Instant |
| 2. Download | Dataset existence verification | ~2 seconds |
| 3. Annotations | JSON parsing into MATLAB table | ~10 seconds |
| 4. Datastores | Data pipeline construction with augmentation | ~20 seconds |
| 5. Model | Two-head ResNet-18 assembly | ~15 seconds |
| 6. Training | Custom training loop execution | 6--12h (CPU) / 30--60min (GPU) |
| 7. Evaluation | Test set metrics and visualizations | ~10 minutes |

### 4. Inference on new images

```matlab
addpath('utils');
loaded = load('results/trained_model.mat');
[timeStr, confidence] = predict_time(loaded.net, 'path/to/clock_image.jpg');
```

---

## Dataset

This project uses the **TickTockVQA** dataset, introduced in *"It's Time to Get It Right"* (CVPR 2026 Findings).

| Property | Value |
|----------|-------|
| Total images | 12,483 |
| Sources | OpenImages, COCO, ClockMovies, VisualGenome, CC12M, ImageNet, SBU |
| Clock types | Wall clocks, wristwatches, tower clocks, alarm/desk clocks |
| Annotation fields | Hour (1--12), minute (0--59), clock type, source, design |

**Data splits used in this project:**

| Split | Samples | Purpose |
|-------|---------|---------|
| Train | 6,150 | Model training with augmentation |
| Validation | 1,086 | Checkpoint selection and early stopping |
| Test | 5,247 | Final performance evaluation |

The validation set is constructed via stratified holdout (15%) from the original training split, preserving the hour distribution.

---

## Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Backbone | ResNet-18 (ImageNet pretrained, frozen) |
| Optimizer | Adam |
| Initial learning rate | 1 x 10^-4 |
| LR schedule | Step decay: multiply by 0.5 every 10 epochs |
| Batch size | 32 |
| Maximum epochs | 30 |
| Early stopping patience | 5 epochs (monitoring validation loss) |
| Dropout rate | 0.4 |
| Loss function | Sum of cross-entropy losses (hour + minute heads) |

### Data Augmentation

Applied exclusively during training:

| Transform | Range | Rationale |
|-----------|-------|-----------|
| Random rotation | [-15, +15] degrees | Account for tilted clock orientations |
| Brightness jitter | [-20, +20] intensity | Varying illumination conditions |
| Contrast adjustment | [0.8, 1.2] multiplier | Indoor vs. outdoor lighting |
| Horizontal/vertical flip | **Disabled** | Mirroring a clock face alters the displayed time |

---

## Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Hour Accuracy | Percentage of correct hour predictions | > 85% |
| Minute Accuracy | Percentage of exact minute matches | > 60% |
| MATE | Mean Absolute Time Error (circular, in minutes) | < 10 minutes |
| Within 5 min | Percentage of predictions within 5 minutes of ground truth | > 70% |

### Circular Time Error

Standard absolute difference produces incorrect results at clock boundaries. The MATE metric computes the shortest arc on a 720-minute (12-hour) circle:

```
error = min(|diff|, 720 - |diff|)
```

| Predicted | Ground Truth | Naive Error | Circular Error (MATE) |
|-----------|-------------|-------------|----------------------|
| 12:05 | 11:55 | 710 min | 10 min |
| 03:42 | 03:45 | 3 min | 3 min |

---

## Documentation

Comprehensive documentation is available in the [`Documentation/`](Documentation/) directory:

- [**Documentation.md**](Documentation/Documentation.md) — Technical reference covering architecture details, algorithm descriptions, code walkthrough, and troubleshooting guide
- [**Diagrams.md**](Documentation/Diagrams.md) — Software engineering diagrams (13 Mermaid diagrams) including system architecture, data flow, training pipeline, and evaluation flows

---

## References

1. Choi, J., Lee, J.W., You, S., & Lee, J. (2026). *It's Time to Get It Right: Improving Analog Clock Reading and Clock-Hand Spatial Reasoning in Vision-Language Models.* CVPR 2026 Findings. [arXiv:2603.08011](https://arxiv.org/abs/2603.08011)

2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition.* IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

3. Kingma, D.P., & Ba, J. (2015). *Adam: A Method for Stochastic Optimization.* International Conference on Learning Representations (ICLR).

---

## License

This project is intended for academic and educational purposes. The TickTockVQA dataset carries mixed licensing terms from its constituent sources — refer to the [dataset repository](https://huggingface.co/datasets/jaeha-choi/TickTockVQA) for detailed attribution and redistribution conditions.
