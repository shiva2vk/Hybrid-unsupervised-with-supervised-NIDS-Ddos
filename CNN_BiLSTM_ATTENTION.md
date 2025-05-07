
#  CNN-BiLSTM-Attention Model for Multiclass Network Intrusion Detection

## Abstract

This project introduces a hybrid deep learning architecture combining **1D Convolutional Neural Networks (CNNs)**, **Bidirectional LSTMs (BiLSTM)**, and an **attention mechanism** to detect and classify multiclass network intrusions. The model is designed to capture both **local feature patterns** and **temporal dependencies**, making it effective for sequential network traffic data. 

Class imbalance is addressed using **SMOTE**, and the model was trained using focal loss with class weights to further enhance minority class recognition. Evaluation shows a strong weighted F1-score of **0.68** with improved recall for difficult attack classes, making this model a strong baseline for robust network intrusion detection systems.

---

##  Model Overview

This model uses a combination of **CNN + BiLSTM + Attention** layers to classify HTTP/network traffic data into one of 11 classes (benign + 10 attack types). It is designed for high-recall detection of temporal patterns in traffic and leverages SMOTE oversampling during training.

- **Architecture**: CNN → BiLSTM → Attention → Dense
- **Loss Function**: Focal Loss with class weights
- **Optimizer**: Adam
- **Sampling Strategy**: SMOTE
- **Framework**: PyTorch

---
(image.png)

## Intended Use

| Aspect            | Description                                               |
|-------------------|-----------------------------------------------------------|
| **Task**          | Multiclass classification (NIDS / WAF log classification) |
| **Input Format**  | 1D vector (sequence of features)                          |
| **Deployment**    | Server-grade environments (due to BiLSTM memory usage)    |
| **Expected Users**| Security analysts, ML ops, researchers                    |

---

## Model Performance (on Validation/Test Set)

| Metric              | Value     |
|---------------------|-----------|
| **Accuracy**        | 63.0%     |
| **Weighted F1 Score** | 0.68      |
| **Macro F1 Score**  | 0.34      |
| **Recall (Class 6)**| 1.00      |
| **Recall (Class 10)**| ~0.32    |

> SMOTE helped improve detection for rare classes while maintaining stable performance on major ones.

---

## Per-Class Performance Snapshot

| Class ID | Description         | Precision | Recall | F1-Score |
|----------|---------------------|-----------|--------|----------|
| 0        | BENIGN              | 0.88      | 0.71   | 0.79     |
| 1        | Attack Type 1       | 0.01      | 0.32   | 0.02     |
| 6        | Attack Type 6       | 1.00      | 1.00   | 1.00     |
| 10       | Attack Type 10      | 0.01      | 0.32   | 0.03     |
| ...      | ...                 | ...       | ...    | ...      |

---

##  Model Artifacts

| Artifact                         | Format       | Notes                          |
|----------------------------------|--------------|--------------------------------|
| `cnn_bilstm_attn.pth`            | PyTorch      | Original state_dict            |
| `cnn_bilstm_attn_traced.pt`      | TorchScript  | Optimized for inference        |
| `cnn_bilstm_attn.onnx`           | ONNX         | Cross-platform deployment      |

---

##  Inference Example

```python
import torch
model = torch.jit.load("cnn_bilstm_attn_traced.pt")
model.eval()
with torch.no_grad():
    output = model(torch.tensor(input_array).unsqueeze(0).unsqueeze(0).float())
    prediction = torch.argmax(output, dim=1).item()
```

---

## Limitations

- **Larger model size and slower inference** due to BiLSTM layers
- Recall for some rare classes (e.g., 1, 2, 4) is still low despite SMOTE
- Not ideal for edge deployment without model compression

---

## Ethical Considerations

- While rare class detection has improved, **false negatives may still occur**.
- It is recommended to **combine this model with rule-based or anomaly detection systems** for stronger defense.
- Monitor performance over time with real attack logs.

---

##  Future Work

- Replace BiLSTM with transformer or Temporal Convolution Network (TCN)
- Apply quantization/pruning for lightweight deployment
- Add attention visualizations for better interpretability

---

## ✍️ Author & Contact

- **Author**: Vivek K. (AI/ML + Cybersecurity Engineer)


---

> ⚠️ Evaluate this model in your specific environment before full deployment.
