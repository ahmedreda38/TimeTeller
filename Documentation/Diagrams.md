# Analog Clock Reader — Software Diagrams

---

## 1. System Architecture (High-Level)

```mermaid
graph TB
    subgraph Input
        A[/"📷 Clock Image<br/>(any style)"/]
    end

    subgraph Preprocessing
        B["Image Reader<br/>imread()"]
        C["Resize to 224×224"]
        D["Grayscale → RGB"]
        E["Cast to single<br/>(0-255 range)"]
    end

    subgraph Model["Two-Head ResNet-18"]
        F["ResNet-18 Backbone<br/>(ImageNet pretrained, frozen)"]
        G["Global Average Pooling<br/>(pool5: 512-dim)"]
        
        subgraph HourHead["Hour Head"]
            H1["FC(512→256)"]
            H2["BatchNorm + ReLU"]
            H3["Dropout(0.4)"]
            H4["FC(256→12)"]
            H5["Softmax"]
        end
        
        subgraph MinuteHead["Minute Head"]
            M1["FC(512→256)"]
            M2["BatchNorm + ReLU"]
            M3["Dropout(0.4)"]
            M4["FC(256→60)"]
            M5["Softmax"]
        end
    end

    subgraph Output
        O1["Hour Prediction<br/>argmax → 1..12"]
        O2["Minute Prediction<br/>argmax−1 → 0..59"]
        O3[/"⏰ Predicted Time<br/>hh:mm"/]
    end

    A --> B --> C --> D --> E --> F --> G
    G --> H1 --> H2 --> H3 --> H4 --> H5 --> O1
    G --> M1 --> M2 --> M3 --> M4 --> M5 --> O2
    O1 --> O3
    O2 --> O3

    style Input fill:#e1f5fe,stroke:#0288d1
    style Model fill:#fff3e0,stroke:#f57c00
    style HourHead fill:#e8f5e9,stroke:#388e3c
    style MinuteHead fill:#e3f2fd,stroke:#1976d2
    style Output fill:#fce4ec,stroke:#c62828
```

---

## 2. Complete Pipeline Flowchart

```mermaid
flowchart TD
    START([Start]) --> S1

    subgraph S1["Step 1: Dataset Download"]
        S1A{"annotations.json<br/>exists?"}
        S1B["Run huggingface-cli<br/>download 3.7GB"]
        S1C["Validate: count<br/>train + test images"]
        S1A -->|No| S1B --> S1C
        S1A -->|Yes| S1C
    end

    S1 --> S2

    subgraph S2["Step 2: Load Annotations"]
        S2A["Read annotations.json<br/>jsondecode()"]
        S2B["Build MATLAB table<br/>imagePath, hour, minute,<br/>hourLabel, minuteLabel, split"]
        S2C["Filter missing images"]
        S2A --> S2B --> S2C
    end

    S2 --> S3

    subgraph S3["Step 3: Create Datastores"]
        S3A["Split train → train 85% + val 15%<br/>(stratified by hour)"]
        S3B["Create imageDatastore<br/>+ arrayDatastore × 2"]
        S3C["Apply transform:<br/>resize, gray2rgb, augment"]
        S3D["Wrap in minibatchqueue<br/>(batched, GPU-ready)"]
        S3A --> S3B --> S3C --> S3D
    end

    S3 --> S4

    subgraph S4["Step 4: Build Model"]
        S4A["Load pretrained<br/>ResNet-18"]
        S4B["Remove fc1000 +<br/>prob + ClassificationLayer"]
        S4C["Add Hour branch<br/>(6 layers from pool5)"]
        S4D["Add Minute branch<br/>(6 layers from pool5)"]
        S4E["Convert to dlnetwork"]
        S4F["Freeze backbone<br/>setLearnRateFactor = 0"]
        S4A --> S4B --> S4C --> S4D --> S4E --> S4F
    end

    S4 --> S5

    subgraph S5["Step 5: Train"]
        S5A["For each epoch:"]
        S5B["Forward pass<br/>both heads"]
        S5C["Compute combined loss<br/>L = CE_hour + CE_minute"]
        S5D["Backprop: dlgradient"]
        S5E["Update: adamupdate"]
        S5F["Validate on val set"]
        S5G{"Val loss<br/>improved?"}
        S5H["Save best model"]
        S5I["Increment patience"]
        S5J{"Patience<br/>exceeded?"}
        S5A --> S5B --> S5C --> S5D --> S5E --> S5F --> S5G
        S5G -->|Yes| S5H --> S5A
        S5G -->|No| S5I --> S5J
        S5J -->|No| S5A
        S5J -->|Yes| S5K["Early Stop"]
    end

    S5 --> S6

    subgraph S6["Step 6: Evaluate"]
        S6A["Run all test images<br/>through model"]
        S6B["Compute: MATE, hour acc,<br/>minute acc, within-5-min"]
        S6C["Generate confusion<br/>matrices + histograms"]
        S6A --> S6B --> S6C
    end

    S6 --> S7

    subgraph S7["Step 7: Inference"]
        S7A["Load any clock image"]
        S7B["Predict time + confidence"]
        S7C["Display result with<br/>bar charts"]
        S7A --> S7B --> S7C
    end

    S7 --> DONE([Pipeline Complete])

    style S1 fill:#e3f2fd,stroke:#1565c0
    style S2 fill:#e8f5e9,stroke:#2e7d32
    style S3 fill:#fff3e0,stroke:#ef6c00
    style S4 fill:#f3e5f5,stroke:#7b1fa2
    style S5 fill:#fce4ec,stroke:#c62828
    style S6 fill:#e0f7fa,stroke:#00838f
    style S7 fill:#fff8e1,stroke:#f9a825
```

---

## 3. ResNet-18 Backbone Architecture (Layer Detail)

```mermaid
graph TD
    IN["Input<br/>224×224×3"] --> CONV1["conv1<br/>7×7, 64, stride 2<br/>→ 112×112×64"]
    CONV1 --> BN1["bn_conv1<br/>BatchNorm"]
    BN1 --> RELU1["relu_conv1<br/>ReLU"]
    RELU1 --> POOL1["pool1<br/>MaxPool 3×3, stride 2<br/>→ 56×56×64"]

    POOL1 --> RES2A["res2a<br/>3×3, 64 → 64<br/>+ skip connection"]
    RES2A --> RES2B["res2b<br/>3×3, 64 → 64<br/>+ skip connection<br/>→ 56×56×64"]

    RES2B --> RES3A["res3a<br/>3×3, 64 → 128, stride 2<br/>+ 1×1 projection"]
    RES3A --> RES3B["res3b<br/>3×3, 128 → 128<br/>→ 28×28×128"]

    RES3B --> RES4A["res4a<br/>3×3, 128 → 256, stride 2<br/>+ 1×1 projection"]
    RES4A --> RES4B["res4b<br/>3×3, 256 → 256<br/>→ 14×14×256"]

    RES4B --> RES5A["res5a<br/>3×3, 256 → 512, stride 2<br/>+ 1×1 projection"]
    RES5A --> RES5B["res5b<br/>3×3, 512 → 512<br/>→ 7×7×512"]

    RES5B --> POOL5["pool5<br/>Global Avg Pool<br/>→ 1×1×512"]

    POOL5 --> HOUR["🕐 Hour Head<br/>FC→BN→ReLU→Drop→FC→Softmax<br/>→ 12 classes"]
    POOL5 --> MIN["🕐 Minute Head<br/>FC→BN→ReLU→Drop→FC→Softmax<br/>→ 60 classes"]

    style IN fill:#e1f5fe
    style POOL5 fill:#fff9c4,stroke:#f9a825,stroke-width:3px
    style HOUR fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px
    style MIN fill:#bbdefb,stroke:#1565c0,stroke-width:2px
```

---

## 4. Data Flow Diagram

```mermaid
flowchart LR
    subgraph Storage["Disk Storage"]
        JSON[("annotations.json<br/>12,483 records")]
        IMGS[("images/<br/>train: 7,236<br/>test: 5,247")]
    end

    subgraph Parse["load_annotations.m"]
        P1["jsondecode()"]
        P2["Build MATLAB table<br/>with categoricals"]
        P3["Validate file existence"]
    end

    subgraph Split["create_datastores.m"]
        SP1["cvpartition<br/>stratified by hour"]
        SP2["Train: 6,150"]
        SP3["Val: 1,086"]
        SP4["Test: 5,247"]
    end

    subgraph DS["Datastore Pipeline"]
        DS1["imageDatastore"]
        DS2["arrayDatastore<br/>(hour indices)"]
        DS3["arrayDatastore<br/>(minute indices)"]
        DS4["combine()"]
        DS5["transform()<br/>resize + augment"]
        DS6["minibatchqueue<br/>batch=32"]
    end

    subgraph Batch["Mini-Batch Output"]
        B1["X: dlarray<br/>[224×224×3×32]<br/>format: SSCB"]
        B2["hourIdx: double<br/>[1×32]<br/>values: 1..12"]
        B3["minIdx: double<br/>[1×32]<br/>values: 1..60"]
    end

    JSON --> P1 --> P2 --> P3
    IMGS --> P3
    P3 --> SP1
    SP1 --> SP2 & SP3 & SP4

    SP2 --> DS1 & DS2 & DS3
    DS1 & DS2 & DS3 --> DS4 --> DS5 --> DS6
    DS6 --> B1 & B2 & B3

    style Storage fill:#e3f2fd
    style Parse fill:#e8f5e9
    style Split fill:#fff3e0
    style DS fill:#f3e5f5
    style Batch fill:#fce4ec
```

---

## 5. Training Loop Sequence Diagram

```mermaid
sequenceDiagram
    participant Main as main.m
    participant Train as train_clock_model.m
    participant MBQ as minibatchqueue
    participant Net as dlnetwork
    participant Adam as adamupdate
    participant Val as Validation
    participant Disk as results/

    Main->>Train: train_clock_model(net, dsTrain, dsVal, opts)
    
    loop Each Epoch (1..30)
        Train->>Train: Compute LR = initialLR × 0.5^floor(epoch/10)
        Train->>MBQ: reset(dsTrain)
        
        loop Each Mini-Batch
            Train->>MBQ: [X, hIdx, mIdx] = next(dsTrain)
            Train->>Train: THour = oneHotEncode(hIdx, 12)
            Train->>Train: TMin = oneHotEncode(mIdx, 60)
            Train->>Net: dlfeval(@modelLoss, net, X, THour, TMin)
            Net-->>Train: [loss, gradients]
            Train->>Adam: adamupdate(net, gradients, state, iter, lr)
            Adam-->>Train: updated net
        end

        Train->>Val: evaluateOnSet(net, dsVal)
        Val-->>Train: [valLoss, valHourAcc, valMinAcc]
        
        alt Val loss improved
            Train->>Disk: save('trained_model.mat', net)
            Train->>Train: Reset patience = 0
        else No improvement
            Train->>Train: patience++
            alt patience >= 5
                Train-->>Main: Early stop → return best net
            end
        end
        
        Train->>Train: Update live plot
    end
    
    Train-->>Main: return [bestNet, trainingLog]
```

---

## 6. Model Loss Computation (Forward + Backward)

```mermaid
flowchart TD
    X["Input Batch X<br/>[224×224×3×B]"] --> FWD["forward(net, X)"]
    
    FWD --> YH["Y_hour<br/>[12×B] probabilities"]
    FWD --> YM["Y_minute<br/>[60×B] probabilities"]
    
    TH["T_hour (one-hot)<br/>[12×B] targets"] --> CEH
    TM["T_minute (one-hot)<br/>[60×B] targets"] --> CEM
    
    YH --> CEH["CrossEntropy<br/>L_hour = -Σ T·log(Y)"]
    YM --> CEM["CrossEntropy<br/>L_minute = -Σ T·log(Y)"]
    
    CEH --> TOTAL["Total Loss<br/>L = w_h × L_hour + w_m × L_minute"]
    CEM --> TOTAL
    
    TOTAL --> GRAD["dlgradient(L, net.Learnables)"]
    GRAD --> UPDATE["adamupdate(net, gradients,<br/>avgGrad, avgSqGrad, iter, lr)"]
    UPDATE --> NNET["Updated Network"]

    style X fill:#e3f2fd
    style TOTAL fill:#fff9c4,stroke:#f57f17,stroke-width:2px
    style GRAD fill:#fce4ec
    style NNET fill:#c8e6c9
```

---

## 7. Evaluation Metrics Flow

```mermaid
flowchart TD
    TEST["Test Set<br/>5,247 images"] --> PRED["Forward pass<br/>(batch by batch)"]
    
    PRED --> PH["Predicted Hours<br/>argmax(Y_hour)"]
    PRED --> PM["Predicted Minutes<br/>argmax(Y_min) − 1"]
    
    GT["Ground Truth<br/>from annotations"] --> TH["True Hours"]
    GT --> TM["True Minutes"]
    
    PH --> HACC["Hour Accuracy<br/>mean(predH == trueH)"]
    TH --> HACC
    
    PM --> MACC["Minute Accuracy<br/>mean(predM == trueM)"]
    TM --> MACC
    
    PH --> CTE["circular_time_error()"]
    PM --> CTE
    TH --> CTE
    TM --> CTE
    
    CTE --> MATE["MATE<br/>mean(errors)"]
    CTE --> MED["Median ATE<br/>median(errors)"]
    CTE --> W5["Within 5 min<br/>mean(err ≤ 5)"]
    CTE --> W15["Within 15 min<br/>mean(err ≤ 15)"]
    
    HACC --> REPORT["Final Report<br/>+ Confusion Matrices<br/>+ Error Histogram"]
    MACC --> REPORT
    MATE --> REPORT
    MED --> REPORT
    W5 --> REPORT
    W15 --> REPORT

    style TEST fill:#e3f2fd
    style CTE fill:#fff9c4,stroke:#f57f17,stroke-width:2px
    style REPORT fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px
```

---

## 8. File Dependency Graph

```mermaid
graph TD
    MAIN["main.m<br/>(orchestrator)"]
    
    MAIN --> DL["download_dataset.m"]
    MAIN --> LA["load_annotations.m"]
    MAIN --> CD["create_datastores.m"]
    MAIN --> BC["build_clock_model.m"]
    MAIN --> TC["train_clock_model.m"]
    MAIN --> EM["evaluate_model.m"]
    MAIN --> PT["predict_time.m"]
    
    BC --> FL["utils/freeze_layers.m"]
    PT --> PP["utils/preprocess_clock_image.m"]
    EM --> CTE["utils/circular_time_error.m"]
    PT --> CTE
    MAIN --> CTE
    
    CD -.->|reads| DATA[("data/dataset/<br/>annotations.json<br/>+ images/")]
    LA -.->|reads| DATA
    DL -.->|creates| DATA
    
    TC -.->|saves| RES[("results/<br/>trained_model.mat<br/>training_log.mat")]
    EM -.->|saves| RES
    EM -.->|saves| FIG[("results/figures/<br/>*.png")]

    style MAIN fill:#fff9c4,stroke:#f57f17,stroke-width:3px
    style DATA fill:#e3f2fd
    style RES fill:#c8e6c9
    style FIG fill:#f3e5f5
```

---

## 9. Circular Time Error Visualization

```mermaid
graph LR
    subgraph Clock["12-Hour Clock Circle (720 minutes)"]
        direction TB
        T12["12:00<br/>(0 min)"]
        T3["3:00<br/>(180 min)"]
        T6["6:00<br/>(360 min)"]
        T9["9:00<br/>(540 min)"]
    end

    subgraph Example1["Example 1"]
        E1A["Predicted: 11:55<br/>(715 min)"]
        E1B["True: 12:05<br/>(5 min)"]
        E1C["Naive: |715 − 5| = 710"]
        E1D["Circular: min(710, 720−710) = 10 ✓"]
        E1A --> E1C
        E1B --> E1C
        E1C --> E1D
    end

    subgraph Example2["Example 2"]
        E2A["Predicted: 3:45<br/>(225 min)"]
        E2B["True: 3:42<br/>(222 min)"]
        E2C["Naive: |225 − 222| = 3"]
        E2D["Circular: min(3, 720−3) = 3 ✓"]
        E2A --> E2C
        E2B --> E2C
        E2C --> E2D
    end

    style Example1 fill:#e8f5e9
    style Example2 fill:#e3f2fd
    style E1D fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px
    style E2D fill:#bbdefb,stroke:#1565c0,stroke-width:2px
```

---

## 10. Transfer Learning Strategy

```mermaid
graph TD
    subgraph Phase1["Phase 1: Feature Extraction"]
        direction TB
        P1A["ResNet-18 Backbone<br/>🔒 FROZEN (LR = 0)<br/>ImageNet weights preserved"]
        P1B["Hour Head<br/>🔓 TRAINABLE (LR = 1e-4)<br/>Random init"]
        P1C["Minute Head<br/>🔓 TRAINABLE (LR = 1e-4)<br/>Random init"]
        P1D["Epochs: 1–30<br/>Only heads learn"]
    end

    subgraph Phase2["Phase 2: Fine-Tuning (Optional)"]
        direction TB
        P2A["ResNet-18 res1–res4<br/>🔒 FROZEN"]
        P2B["ResNet-18 res5b<br/>🔓 UNFROZEN (LR = 1e-5)<br/>Adapts to clocks"]
        P2C["Hour + Minute Heads<br/>🔓 TRAINABLE (LR = 1e-5)"]
        P2D["Epochs: 31–40<br/>Backbone adapts"]
    end

    Phase1 -->|"If targets not met"| Phase2

    style Phase1 fill:#e8f5e9,stroke:#2e7d32
    style Phase2 fill:#fff3e0,stroke:#ef6c00
    style P1A fill:#ffcdd2
    style P2A fill:#ffcdd2
    style P1B fill:#c8e6c9
    style P1C fill:#c8e6c9
    style P2B fill:#fff9c4
    style P2C fill:#c8e6c9
```

---

## 11. Augmentation Pipeline

```mermaid
flowchart LR
    RAW["Raw Image<br/>(variable size)"] --> CHK{"Channels?"}
    
    CHK -->|"1 (gray)"| G2R["Grayscale → RGB<br/>repmat(img, [1 1 3])"]
    CHK -->|"4 (RGBA)"| A2R["RGBA → RGB<br/>img(:,:,1:3)"]
    CHK -->|"3 (RGB)"| AUG

    G2R --> AUG
    A2R --> AUG

    subgraph AUG["Training Augmentation"]
        AUG1["Random Rotation<br/>[-15°, +15°]"]
        AUG2["Brightness Jitter<br/>[-20, +20] intensity"]
        AUG3["Contrast Adjust<br/>[0.8, 1.2]×"]
        AUG4["❌ NO Flip<br/>(changes time!)"]
        AUG1 --> AUG2 --> AUG3
    end

    AUG --> RESIZE["Resize<br/>224 × 224"]
    RESIZE --> CAST["single()<br/>0–255 range"]
    CAST --> OUT["Preprocessed Image<br/>[224×224×3] single"]

    style AUG fill:#fff3e0,stroke:#ef6c00
    style AUG4 fill:#ffcdd2,stroke:#c62828
    style OUT fill:#c8e6c9
```

---

## 12. Early Stopping State Machine

```mermaid
stateDiagram-v2
    [*] --> Training: Start epoch 1

    Training --> CheckValLoss: Epoch complete
    
    CheckValLoss --> Improved: valLoss < bestValLoss
    CheckValLoss --> NotImproved: valLoss >= bestValLoss

    Improved --> SaveModel: Save checkpoint
    SaveModel --> ResetPatience: patience = 0
    ResetPatience --> CheckEpochs

    NotImproved --> IncrementPatience: patience++
    IncrementPatience --> CheckPatience

    CheckPatience --> CheckEpochs: patience < 5
    CheckPatience --> EarlyStop: patience >= 5

    CheckEpochs --> Training: epoch < maxEpochs
    CheckEpochs --> NormalStop: epoch == maxEpochs

    EarlyStop --> [*]: Return best model
    NormalStop --> [*]: Return best model
```

---

## 13. Inference Pipeline

```mermaid
flowchart TD
    IMG[/"📷 New Clock Image"/] --> READ["imread(imagePath)"]
    READ --> PRE["preprocess_clock_image()<br/>resize, gray2rgb, single"]
    PRE --> DL["dlarray(img, 'SSCB')<br/>batch of 1"]
    
    DL --> GPU{"canUseGPU()?"}
    GPU -->|Yes| TOGPU["gpuArray(X)"]
    GPU -->|No| FWD
    TOGPU --> FWD

    FWD["predict(net, X, 'Outputs', ...)"]
    FWD --> YH["Y_hour: [12×1]<br/>probability vector"]
    FWD --> YM["Y_minute: [60×1]<br/>probability vector"]

    YH --> MAXH["argmax → hourIdx"]
    YM --> MAXM["argmax → minIdx"]

    MAXH --> DECODE_H["hour = hourIdx<br/>(1..12)"]
    MAXM --> DECODE_M["minute = minIdx − 1<br/>(0..59)"]

    DECODE_H --> FORMAT["sprintf('%02d:%02d')"]
    DECODE_M --> FORMAT

    FORMAT --> DISPLAY["Display:<br/>• Image with overlay<br/>• Hour confidence bars<br/>• Minute confidence bars<br/>• Top-3 predictions"]

    style IMG fill:#e3f2fd
    style FORMAT fill:#fff9c4,stroke:#f57f17,stroke-width:2px
    style DISPLAY fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px
```

