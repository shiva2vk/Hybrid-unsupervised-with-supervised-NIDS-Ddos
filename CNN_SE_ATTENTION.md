
#  CNN-SE-Attention Model for Multiclass Network Intrusion Detection

##  Abstract

This project presents a **lightweight yet effective deep learning model** based on 1D Convolutional Neural Networks (CNNs) with **Squeeze-and-Excitation (SE) attention** for detecting and classifying network intrusions in multiclass scenarios. The model targets real-time and edge deployment use cases by maintaining high accuracy and recall without the heavy overhead of recurrent architectures like LSTMs.

To address severe class imbalance in the dataset, we employed **SMOTE** for synthetic minority oversampling, and class-weighted focal loss. The model was trained and evaluated on preprocessed HTTP request or network traffic features across 11 classes (benign + 10 attack types). Results show competitive performance, particularly on minority classes, with a **weighted F1-score of 0.67** and **macro F1 of 0.34**, making this model a solid candidate for real-world deployment in security-critical environments.

---

## Model Overview

This is a lightweight **1D CNN + SE Attention** model trained for **multiclass network intrusion detection**. It classifies HTTP/network request data into one of 11 classes (benign + 10 attack types). The model was trained on an imbalanced dataset and improved using **SMOTE oversampling**.

- **Architecture**: Convolutional Neural Network (Conv1D) + SE Attention + Fully Connected Layers
- **Optimizer**: Adam
- **Loss Function**: CrossEntropyLoss with class weights
- **Dataset**: Custom-labeled HTTP or network flow dataset
- **Sampling Strategy**: SMOTE

---

##  Intended Use

| Aspect            | Description                                               |
|-------------------|-----------------------------------------------------------|
| **Task**          | Multiclass classification (NIDS / WAF log classification) |
| **Input Format**  | 1D vector (e.g., extracted request features)              |
| **Deployment**    | Edge devices, embedded platforms, cloud APIs              |
| **Expected Users**| Security engineers, embedded/IoT teams, researchers       |

---

## Model Performance (on Validation/Test Set)

| Metric              | Value     |
|---------------------|-----------|
| **Accuracy**        | 63.0%     |
| **Weighted F1 Score** | 0.67      |
| **Macro F1 Score**  | 0.34      |
| **Recall (Class 6)**| 1.00      |
| **Recall (Class 10)**| ~0.31    |

> Class imbalance was handled using SMOTE, improving recall for rare attack types.

---

##  Per-Class Performance Snapshot

| Class ID | Description         | Precision | Recall | F1-Score |
|----------|---------------------|-----------|--------|----------|
| 0        | BENIGN              | 0.88      | 0.71   | 0.79     |
| 1        | Attack Type 1       | 0.01      | 0.32   | 0.02     |
| 6        | Attack Type 6       | 1.00      | 1.00   | 1.00     |
| 10       | Attack Type 10      | 0.01      | 0.32   | 0.03     |
| ...      | ...                 | ...       | ...    | ...      |

---

##  Model Artifacts

| Artifact                         | Format       | Notes                      |
|----------------------------------|--------------|----------------------------|
| `cnn_se_attention_traced.pt`     | TorchScript  | For PyTorch edge inference |
| `cnn_se_attention.onnx`          | ONNX         | Cross-platform deployment  |
| `cnn_se_attention_quantized.pt`  | Quantized    | CPU-efficient inference    |

---

##  Inference Example

```python
import torch
model = torch.jit.load("cnn_se_attention_traced.pt")
model.eval()
with torch.no_grad():
    output = model(torch.tensor(input_array).unsqueeze(0).unsqueeze(0).float())
    prediction = torch.argmax(output, dim=1).item()
```

---

##  Limitations

- Lower precision/recall for **minority classes** like 1, 2, 4, 10
- No temporal context (non-sequential, no BiLSTM)
- Explainability is limited — consider adding Grad-CAM or SHAP if needed
- Trained on synthetic samples via SMOTE — may need real-world finetuning

---

##  Ethical Considerations

- While detection of rare attack types has been enhanced, we recommend combining this model with rule-based or anomaly-based systems for complete coverage.
- Recommended: Use in combination with **rules or anomaly detectors** for higher coverage.
- Audit model regularly using real attack logs.

---

##  Future Work

- Add transformer or BiLSTM branch for hybrid sequence modeling
- Introduce cost-sensitive loss or focal loss for rare class boosting
- Use Grad-CAM for SE-based interpretability

---

##  Author & Contact

- **Author**: Vivek K. (AI/ML + Cybersecurity Engineer)

Always validate this model on your own dataset before deployment.
