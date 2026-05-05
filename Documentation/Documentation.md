# Analog Clock Reader — Project Documentation

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [System Requirements](#2-system-requirements)
3. [Project Structure](#3-project-structure)
4. [Architecture & Algorithms](#4-architecture--algorithms)
5. [Dataset](#5-dataset)
6. [Installation & Setup](#6-installation--setup)
7. [Running the Pipeline](#7-running-the-pipeline)
8. [Code Walkthrough](#8-code-walkthrough)
9. [Evaluation Metrics](#9-evaluation-metrics)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Project Overview

This project builds a **data-driven analog clock reader** using deep learning. Given a photograph of any analog clock (wall clock, wristwatch, tower clock, desk clock), the model predicts the displayed time as `hh:mm`.

**Approach:** Fine-tune a pretrained **ResNet-18** CNN with **two classification heads** — one for the hour (12 classes) and one for the minute (60 classes) — trained on the **TickTockVQA** dataset of ~12,500 labeled real-world clock images.

### Why Deep Learning (Not Traditional CV)?

Traditional computer vision approaches (Hough transforms, contour detection) fail on real-world clocks due to:
- Varying clock face designs (Roman numerals, Arabic, no numerals)
- Occlusion, reflections, shadows
- Diverse viewing angles and lighting conditions
- Wristwatch complications, ornate hands

A CNN learns robust visual features directly from labeled examples, generalizing across styles.

---

## 2. System Requirements

### Software

| Component | Requirement |
|-----------|-------------|
| MATLAB | R2022b or newer (developed on R2026a) |
| Deep Learning Toolbox | Required |
| Computer Vision Toolbox | Required |
| Image Processing Toolbox | Required |
| Statistics and Machine Learning Toolbox | Required |
| Python | 3.8+ (for dataset download only) |

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8 GB | 16 GB |
| Disk | 6 GB free | 10 GB free |
| GPU | None (CPU works) | NVIDIA with CUDA |

> **Note:** MATLAB GPU computing requires **NVIDIA GPUs with CUDA**. AMD GPUs (e.g., RX 580) are not supported. The code auto-detects GPU via `canUseGPU()` and falls back to CPU.

| Hardware | Training Time (30 epochs) |
|----------|--------------------------|
| CPU only | 6–12 hours |
| NVIDIA GTX 1060+ | 30–60 minutes |

---

## 3. Project Structure

```
d:\vision-projects\
│
├── main.m                        # Orchestrator — run this (7 sections)
│
├── download_dataset.m            # Downloads TickTockVQA from HuggingFace
├── load_annotations.m            # Parses annotations.json → MATLAB table
├── create_datastores.m           # Builds train/val/test data pipelines
├── build_clock_model.m           # Constructs two-head ResNet-18
├── train_clock_model.m           # Custom training loop (Adam + early stop)
├── evaluate_model.m              # Test set metrics + visualizations
├── predict_time.m                # Single-image inference + display
│
├── utils/
│   ├── circular_time_error.m     # Circular time distance (wrap-around)
│   ├── freeze_layers.m           # Freeze backbone layer weights
│   └── preprocess_clock_image.m  # Image resize + gray2rgb + type cast
│
├── download_dataset.py           # Python download with filename sanitization
├── fix_missing_images.py         # Recovery script for Windows filename issues
├── audit_dataset.py              # Dataset integrity checker
│
├── data/
│   └── dataset/                  # TickTockVQA dataset (after download)
│       ├── annotations.json      # 12,483 labeled records
│       └── images/
│           ├── train/            # 7,236 training images
│           └── test/             # 5,247 test images
│
└── results/                      # Generated during training
    ├── trained_model.mat         # Best model checkpoint
    ├── training_log.mat          # Per-epoch metrics
    ├── test_metrics.mat          # Evaluation results
    └── figures/                  # Saved plots
        ├── training_curves.png
        ├── confusion_hour.png
        ├── confusion_minute_binned.png
        ├── error_distribution.png
        └── mate_per_hour.png
```

---

## 4. Architecture & Algorithms

### 4.1 Model Architecture: Two-Head ResNet-18

The model uses **transfer learning** from a ResNet-18 pretrained on ImageNet (1.2M natural images, 1000 classes). The original 1000-class head is removed and replaced with two parallel classification branches.

```
┌──────────────────────────────────────────────────────────┐
│                    INPUT IMAGE                           │
│                   [224 × 224 × 3]                        │
└──────────────────────┬───────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────┐
│              ResNet-18 BACKBONE (Frozen)                  │
│                                                          │
│  ┌──────────┐   ┌──────────┐         ┌──────────┐       │
│  │  conv1   │──▶│ res2a/b  │──▶ ... ─▶│ res5a/b  │      │
│  │ 7×7, 64  │   │ 64 ch    │         │ 512 ch   │       │
│  └──────────┘   └──────────┘         └──────────┘       │
│                                           │              │
│                                    ┌──────▼──────┐       │
│                                    │   pool5     │       │
│                                    │  GAP 1×1    │       │
│                                    │  512-dim    │       │
│                                    └──────┬──────┘       │
└───────────────────────────┬──────────────┬───────────────┘
                            │              │
              ┌─────────────▼──┐    ┌──────▼─────────────┐
              │   HOUR HEAD    │    │   MINUTE HEAD      │
              │                │    │                    │
              │  FC(512→256)   │    │  FC(512→256)       │
              │  BatchNorm     │    │  BatchNorm         │
              │  ReLU          │    │  ReLU              │
              │  Dropout(0.4)  │    │  Dropout(0.4)      │
              │  FC(256→12)    │    │  FC(256→60)        │
              │  Softmax       │    │  Softmax           │
              │                │    │                    │
              │  Output: P(h)  │    │  Output: P(m)      │
              │  [12 × 1]      │    │  [60 × 1]          │
              └────────────────┘    └────────────────────┘
```

### 4.2 Why Two Heads Instead of One?

| Approach | Classes | Problem |
|----------|---------|---------|
| Single head (hh:mm) | 720 | Extremely sparse labels, slow convergence |
| **Two heads (h + m)** | **12 + 60 = 72** | **Clean gradients, independent learning** |
| Regression (angle) | Continuous | Cyclic wrap-around (0°=360°) needs special loss |

The two-head approach lets each branch focus on its own visual cues:
- **Hour hand**: shorter, thicker — determines the hour
- **Minute hand**: longer, thinner — determines the minute
- Each head gets its own gradient signal, avoiding interference

### 4.3 Why Classification Instead of Regression?

Clock hand angles are **cyclic**: 0° and 360° are the same position. Standard regression loss (MSE) treats them as maximally different. Classification avoids this entirely — each possible position is a discrete class, and the softmax output naturally handles uncertainty.

### 4.4 Transfer Learning Strategy

**Phase 1 — Feature extraction (backbone frozen):**
- All ResNet-18 backbone layers have their learn rate set to 0
- Only the two new heads (12 layers total) are trained
- This leverages ImageNet features (edges, textures, shapes) without forgetting them

**Phase 2 — Fine-tuning (optional):**
- Unfreeze the last residual block (`res5b`)
- Train at a much lower learning rate (1e-5)
- Allows the backbone to adapt its high-level features to clock-specific patterns

### 4.5 Training Algorithm

The training uses a **custom training loop** with:

**Optimizer:** Adam (Adaptive Moment Estimation)
- Maintains per-parameter running averages of gradient (momentum) and squared gradient
- Automatically adapts learning rate for each weight
- Formula: `θ_new = θ - lr * m̂ / (√v̂ + ε)` where `m̂` and `v̂` are bias-corrected moment estimates

**Loss Function:**
```
L_total = w_hour × CrossEntropy(Y_hour, T_hour) + w_minute × CrossEntropy(Y_minute, T_minute)
```
Where:
- `CrossEntropy(Y, T) = -Σ T_i × log(Y_i)` (categorical cross-entropy)
- `w_hour = w_minute = 1.0` (equal weighting)
- `Y` = predicted softmax probabilities, `T` = one-hot encoded targets

**Learning Rate Schedule:** Step decay
```
lr(epoch) = initial_lr × drop_factor ^ floor((epoch - 1) / drop_period)
```
- `initial_lr = 1e-4`, `drop_factor = 0.5`, `drop_period = 10`
- LR halves every 10 epochs: 1e-4 → 5e-5 → 2.5e-5

**Early Stopping:**
- Monitors validation loss after each epoch
- If val loss doesn't improve for 5 consecutive epochs → stop training
- Always saves the best model checkpoint (lowest val loss)

**Gradient Computation:** MATLAB's automatic differentiation (`dlgradient`) computes gradients through the entire network graph. Only parameters with non-zero learn rate factors receive updates.

### 4.6 Data Augmentation

Applied **only during training** (not validation/test):

| Augmentation | Range | Rationale |
|-------------|-------|-----------|
| Random rotation | [-15°, +15°] | Clocks may be slightly tilted in photos |
| Brightness jitter | [-20, +20] intensity | Varying lighting conditions |
| Contrast adjustment | [0.8, 1.2] | Indoor vs outdoor, flash vs ambient |
| **NO horizontal flip** | — | **Flipping mirrors the clock → changes the time** |
| **NO vertical flip** | — | **Would invert the clock face** |

### 4.7 Circular Time Error (MATE)

Standard absolute difference doesn't work for clock times because of wrap-around:
- Naive: |11:55 - 12:05| = 710 minutes (WRONG)
- Circular: min(710, 720-710) = **10 minutes** (CORRECT)

```matlab
predTotal = mod(predH - 1, 12) * 60 + predM;   % convert to 0..719 minutes
trueTotal = mod(trueH - 1, 12) * 60 + trueM;
diff = abs(predTotal - trueTotal);
err = min(diff, 720 - diff);                     % shorter arc on 12-hour circle
```

---

## 5. Dataset

### 5.1 TickTockVQA

- **Source:** [HuggingFace — jaeha-choi/TickTockVQA](https://huggingface.co/datasets/jaeha-choi/TickTockVQA)
- **Paper:** "It's Time to Get It Right" (CVPR 2026 Findings)
- **Size:** 12,483 images, 3.7 GB total
- **License:** CC BY 4.0 (OpenImages), mixed for other sources

### 5.2 Annotation Format

Each record in `annotations.json`:
```json
{
  "image_name": "000578c4cbd9c9f2.jpg",
  "image_path": "test/000578c4cbd9c9f2.jpg",
  "hour": 12,
  "minute": 26,
  "ampm": "PM",
  "time_string": "12:26 PM",
  "source": "OpenImages",
  "clock_type": "Wall clocks",
  "environment": "Outdoor",
  "transformation": "Normal",
  "design": ["Roman"]
}
```

### 5.3 Data Splits

| Split | Images | Usage |
|-------|--------|-------|
| Train | 6,150 (85% of original train) | Model training with augmentation |
| Val | 1,086 (15% of original train) | Early stopping & checkpoint selection |
| Test | 5,247 | Final evaluation (never seen during training) |

The validation split is created using **stratified sampling on hour**, ensuring each hour is proportionally represented in both train and val.

### 5.4 Data Sources

| Source | Count | License |
|--------|-------|---------|
| CC12M | 3,677 | CC BY 4.0 |
| COCO 2017 | 2,063 | CC BY 4.0 |
| OpenImages | 1,940 | CC BY 4.0 |
| SBU | 1,533 | Custom |
| VisualGenome | 1,246 | CC BY 4.0 |
| ClockMovies | 1,244 | Custom |
| ImageNet | 780 | Research use |

### 5.5 Label Encoding

MATLAB uses 1-based indexing, which creates an offset for minutes:

| Value | Storage Index | Categorical Label |
|-------|--------------|-------------------|
| Hour 1 | 1 | `"h01"` |
| Hour 12 | 12 | `"h12"` |
| Minute 0 | **1** | `"m00"` |
| Minute 59 | **60** | `"m59"` |

String-prefixed labels (`"h01"` not `"1"`) ensure correct alphabetical ordering in MATLAB categorical arrays.

---

## 6. Installation & Setup

### 6.1 Prerequisites

1. **MATLAB R2022b+** with the four toolboxes listed in Section 2
2. **Python 3.8+** with pip (for dataset download)
3. **~6 GB free disk space** on the target drive

### 6.2 Step-by-Step Installation

#### Step 1: Clone or copy the project

Place all `.m` files into `d:\vision-projects\` as shown in Section 3.

#### Step 2: Install Python download tool

Open **PowerShell** or **Command Prompt**:

```powershell
pip install -U huggingface_hub
```

#### Step 3: (Optional) Set up HuggingFace token for faster downloads

```powershell
huggingface-cli login
```
Paste a token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

#### Step 4: Download the dataset (3.7 GB)

```powershell
cd d:\vision-projects
python download_dataset.py
```

This script handles Windows filename issues automatically (some ClockMovies filenames contain quotes, illegal on Windows).

**Expected time:** 10–30 minutes depending on internet speed.

#### Step 5: Fix any missing images (if needed)

```powershell
$env:PYTHONIOENCODING='utf-8'; python fix_missing_images.py
```

This recovers files with illegal characters from the HuggingFace cache and patches `annotations.json`.

#### Step 6: Verify the download

```powershell
python audit_dataset.py
```

Expected output:
```
Usable images:  12483/12483 (100.0%)
Missing images: 0/12483 (0.0%)
```

#### Step 7: Verify MATLAB toolboxes

Open MATLAB and run:
```matlab
ver('deeplearning')
ver('vision')
ver('images')
ver('stats')
```

All four should return version information without errors.

---

## 7. Running the Pipeline

### 7.1 Quick Start

```matlab
cd d:\vision-projects
edit main.m
% Then run each section with Ctrl+Enter
```

### 7.2 Section-by-Section Guide

`main.m` is divided into 7 executable sections (separated by `%%`). Click inside a section and press **Ctrl+Enter** to run it.

#### Section 1: SETUP (~1 second)
- Adds `utils/` to MATLAB path
- Sets configuration (batch size, learning rate, epochs)
- Checks GPU availability

#### Section 2: DOWNLOAD DATASET (~2 seconds if already downloaded)
- Calls `download_dataset.m`
- Checks if data exists, skips download if so
- Verifies image counts

#### Section 3: LOAD ANNOTATIONS (~5–15 seconds)
- Reads `annotations.json` → MATLAB table
- Displays clock type and source distributions
- Filters out records with missing image files

#### Section 4: CREATE DATASTORES (~10–30 seconds)
- Stratified 85/15 train/val split
- Creates `minibatchqueue` pipelines with augmentation
- Verifies a sample batch and displays 5 training images

#### Section 5: BUILD MODEL (~10–30 seconds)
- Loads pretrained ResNet-18 (may prompt to install support package)
- Removes original classification head
- Adds hour and minute branches
- Freezes backbone, verifies forward pass

#### Section 6: TRAIN MODEL (6–12 hours CPU / 30–60 min GPU)
- Custom training loop with Adam optimizer
- Live updating loss/accuracy plot
- Auto-saves best model to `results/trained_model.mat`
- Early stopping on validation loss (patience = 5 epochs)
- If a saved model exists, asks whether to retrain or load

#### Section 7: EVALUATE & DEMO (~5–15 minutes)
- Runs all test images through the model
- Computes MATE, hour/minute accuracy, within-N-min accuracy
- Generates confusion matrices, error histogram, per-hour breakdown
- Runs inference on 3 random test images with visual display

### 7.3 Smoke Test (Quick Validation)

To test the entire pipeline quickly before a full training run, change the epoch count in Section 1:

```matlab
config.maxEpochs = 2;   % Quick test: 2 epochs (~30 min CPU)
```

### 7.4 Loading a Saved Model Later

```matlab
cd d:\vision-projects
addpath('utils');
loaded = load('results/trained_model.mat');
net = loaded.net;

% Predict on any clock image:
[timeStr, conf] = predict_time(net, 'path/to/clock.jpg');
```

---

## 8. Code Walkthrough

### 8.1 `load_annotations.m`

**Purpose:** Parse the JSON annotation file into a structured MATLAB table.

**Key logic:**
1. `fileread` + `jsondecode` reads the JSON into a struct array
2. Each record is validated: hour must be 1–12, minute must be 0–59
3. Image paths are constructed: `fullfile(baseDir, 'images', record.image_path)`
4. Missing files are filtered out with a validity mask
5. Labels are encoded as categoricals with fixed orderings: `"h01"..."h12"`, `"m00"..."m59"`
6. The `split` field is parsed from the image path prefix (`train/` or `test/`)

**Output table columns:**

| Column | Type | Example |
|--------|------|---------|
| `imagePath` | string | `"d:\...\images\train\abc.jpg"` |
| `hour` | double | `3` |
| `minute` | double | `45` |
| `hourLabel` | categorical | `"h03"` |
| `minuteLabel` | categorical | `"m45"` |
| `split` | string | `"train"` |
| `clockType` | string | `"Wall clocks"` |
| `source` | string | `"OpenImages"` |

---

### 8.2 `create_datastores.m`

**Purpose:** Build GPU-ready data pipelines with augmentation.

**Data flow:**

```
imageDatastore ──┐
arrayDatastore ──┤──▶ combine ──▶ transform ──▶ minibatchqueue
arrayDatastore ──┘    (3 cols)    (preprocess)   (batching)
```

**Key design decisions:**

1. **Stratified split:** Uses `cvpartition` with `'HoldOut'` stratified on the hour column — ensures each hour has equal representation in train and val
2. **Minute offset:** Minutes 0–59 are stored as indices 1–60 (`+1`) for MATLAB's 1-based indexing. Reversed at prediction time (`-1`)
3. **Transform function:** `preprocessSample` handles grayscale→RGB, RGBA→RGB, augmentation (train only), resize, and type conversion
4. **`minibatchqueue` output:** Images as `dlarray` with `'SSCB'` format (Spatial-Spatial-Channel-Batch); labels as plain doubles (one-hot encoded in the training loop)

---

### 8.3 `build_clock_model.m`

**Purpose:** Construct the two-head network architecture.

**Steps:**
1. Load `resnet18` → convert to `layerGraph`
2. Remove layers: `fc1000`, `prob`, `ClassificationLayer_predictions`
3. Add hour branch (6 layers) connected from `pool5`
4. Add minute branch (6 layers) connected from `pool5`
5. Convert to `dlnetwork`
6. Freeze all backbone layers via `setLearnRateFactor(..., 0)`
7. Verify with a random input forward pass

**Layer naming:**

| Branch | Layer Names |
|--------|-------------|
| Hour | `hour_fc1`, `hour_bn`, `hour_relu`, `hour_drop`, `hour_fc2`, `hour_softmax` |
| Minute | `min_fc1`, `min_bn`, `min_relu`, `min_drop`, `min_fc2`, `min_softmax` |

---

### 8.4 `train_clock_model.m`

**Purpose:** Train the network using a custom training loop.

**Training loop pseudocode:**
```
for epoch = 1 to maxEpochs:
    lr = initialLR × dropFactor^(floor((epoch-1) / dropPeriod))
    
    for each mini-batch in training set:
        [X, hourIdx, minIdx] = next(dsTrain)
        THour = one_hot_encode(hourIdx, 12)
        TMin  = one_hot_encode(minIdx,  60)
        
        [loss, gradients] = dlfeval(@modelLoss, net, X, THour, TMin)
        [net, state] = adamupdate(net, gradients, state, iteration, lr)
    end
    
    [valLoss, valHourAcc, valMinAcc] = evaluate_on_validation_set()
    
    if valLoss < bestValLoss:
        save best model
        reset patience counter
    else:
        increment patience counter
        if patience exceeded: STOP
end
```

**Helper functions inside this file:**
- `modelLoss`: Forward pass → cross-entropy on both heads → `dlgradient`
- `oneHotEncode`: Converts index vectors to one-hot `dlarray` matrices
- `predictBatch`: Forward pass → argmax on both heads
- `evaluateOnSet`: Full validation/test evaluation (no gradients)
- `updatePlots`: Refreshes the live training figure

---

### 8.5 `evaluate_model.m`

**Purpose:** Compute test metrics and generate visualizations.

**Metrics computed:**

| Metric | Formula |
|--------|---------|
| Hour accuracy | `mean(predH == trueH) × 100` |
| Minute accuracy | `mean(predM == trueM) × 100` |
| MATE | `mean(circular_time_error(...))` |
| Median ATE | `median(circular_time_error(...))` |
| Within 5 min | `mean(error ≤ 5) × 100` |
| Within 15 min | `mean(error ≤ 15) × 100` |

**Plots generated:**
1. **Hour confusion matrix** — 12×12, row and column normalized
2. **Minute confusion (5-min bins)** — 12×12, groups minutes into 0–4, 5–9, etc.
3. **Error distribution histogram** — with MATE and median lines
4. **Per-hour MATE bar chart** — shows which hours are hardest to predict

---

### 8.6 `predict_time.m`

**Purpose:** Run inference on a single new clock image.

**Flow:**
1. Read image → preprocess (resize 224×224, gray→RGB)
2. Wrap as `dlarray('SSCB')`, move to GPU if available
3. Forward pass → get probability vectors for hour and minute
4. `argmax` on each → decode: hour index → hour, minute index − 1 → minute
5. Display: original image + hour bar chart + minute bar chart

---

### 8.7 Utility Functions

#### `circular_time_error.m`
Computes the shortest distance between two times on a 12-hour, 720-minute circle. Handles wrap-around correctly.

#### `freeze_layers.m`
Iterates through `net.Learnables` and calls `setLearnRateFactor(net, layerName, paramName, 0)` for all parameters in the specified layers.

#### `preprocess_clock_image.m`
Reads an image file, handles grayscale/RGBA conversion, resizes to target dimensions, converts to `single` precision.

---

## 9. Evaluation Metrics

### Target Performance

| Metric | Target | Description |
|--------|--------|-------------|
| Hour accuracy | > 85% | Correct hour prediction |
| Minute accuracy | > 60% | Exact minute prediction |
| MATE | < 10 min | Mean circular time error |
| Within 5 min | > 70% | Predictions within 5 minutes |

### Understanding MATE

MATE (Mean Absolute Time Error) is the primary metric. It measures, on average, how many minutes off the prediction is from the true time, using circular distance on a 12-hour clock.

**Examples:**
| Predicted | True | Naive Error | Circular Error |
|-----------|------|-------------|----------------|
| 03:45 | 03:42 | 3 min | **3 min** |
| 12:05 | 11:55 | 710 min | **10 min** |
| 06:00 | 06:00 | 0 min | **0 min** |
| 01:00 | 11:00 | 120 min | **120 min** |

---

## 10. Troubleshooting

| Problem | Solution |
|---------|----------|
| `resnet18` not found | MATLAB will prompt to install the support package. Click Install. |
| Out of memory | Reduce batch size: `config.batchSize = 16;` in `main.m` Section 1 |
| Training too slow on CPU | Set `config.maxEpochs = 5;` for a shorter run |
| `canUseGPU()` returns 0 on NVIDIA | Ensure CUDA toolkit is installed; run `gpuDevice()` for diagnostics |
| Missing images after download | Run `python fix_missing_images.py` from PowerShell |
| JSON parse error | Ensure `annotations.json` is valid; check `annotations_original.json` backup |
| `download_dataset.m` fails | Download manually via Python: `python download_dataset.py` |
| Want to resume interrupted training | Re-run Section 6; type `n` to load the last saved checkpoint |

---

## References

- **Dataset:** Choi et al., "It's Time to Get It Right: Improving Analog Clock Reading and Clock-Hand Spatial Reasoning in Vision-Language Models," CVPR 2026 Findings. [arXiv:2603.08011](https://arxiv.org/abs/2603.08011)
- **ResNet-18:** He et al., "Deep Residual Learning for Image Recognition," CVPR 2016.
- **Adam Optimizer:** Kingma & Ba, "Adam: A Method for Stochastic Optimization," ICLR 2015.
- **MATLAB Documentation:** [Train Network with Multiple Outputs](https://www.mathworks.com/help/deeplearning/ug/train-network-with-multiple-outputs.html)

---

*Documentation generated for the Analog Clock Reader project. Last updated: May 2026.*
