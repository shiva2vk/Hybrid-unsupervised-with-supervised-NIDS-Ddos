# AI-Powered Network Intrusion Detection System (NIDS)

This repository presents a comprehensive deep learning-based NIDS pipeline, combining both **unsupervised** and **supervised** learning strategies using a variety of AutoEncoder (AE) architectures enhanced with **attention mechanisms**, **LightGBM**, and **sequence modeling**.

---

## Project Structure

The project focuses on **detecting cyberattacks in real-time** using the CIC-IoT2023 dataset. It explores and compares multiple AE-based architectures for anomaly detection and attack classification.

dataset: "http://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/Dataset/CSV/CSV/"
---

## Models Implemented

### 1. **Unsupervised: AE with Attention + Thresholding**
- Train on benign traffic only
- Use **reconstruction loss** to detect anomalies
- **Attention layer** learns feature importance dynamically
- Suitable for edge deployment or zero-day attack detection

### 2.  **AE + LSTM + Attention**
- Combines AE’s compression with **temporal modeling** via LSTM
- Attention enhances interpretability and focus on key time steps
- Ideal for sequential attack patterns (e.g., slow port scans)

### 3.  **AE + SE Attention (Squeeze-and-Excitation)**
- Focuses on **channel-wise feature importance**
- Lightweight and edge-friendly
- Uses SE blocks to reweight features based on global context

---

## upervised Extensions

### 4. **AE + Attention + LightGBM**
- First extract AE latent features and reconstruction loss
- Feed combined features to **LightGBM** for **multi-class classification**
- Achieves strong per-class F1 for known attack types

### 5. **CNN + BiLSTM + Attention**
- Built for sequence-based classification
- CNN captures short-term features, BiLSTM models sequences, Attention focuses on critical timesteps
- Early stopping + per-epoch F1 monitoring
- Achieved macro F1 ~0.68 and good performance on major attack types

---

## Dataset Used

**CIC-IoT2023**
- 13 attack types including: DDoS, PortScan, SQLi, XSS, Backdoor,  etc.
- CSV format, flow-level features extracted
- Handled extreme imbalance using sampling + metric-weighting

---

##  Tools & Frameworks

- Python, PyTorch, Scikit-learn, LightGBM
- t-SNE, PR Curve, confusion matrix, and loss visualization included
- Early stopping, model checkpointing, and F1-tuned evaluation

---

## Results Summary

| Model                          | Macro F1 | Weighted F1 | Accuracy |
|-------------------------------|----------|-------------|----------|
| AE + Attention + Threshold     | ~0.51    | ~0.83       | ~84%     |
| AE + SE Attention              | ~0.55    | ~0.85       | ~86%     |
| AE + Attention + LightGBM      | ~0.68    | ~0.91       | ~89%     |
| CNN + BiLSTM + Attention       | ~0.67    | ~0.74       | ~64%     |

---

##  Future Enhancements


- Online learning adaptation (for live network monitoring)
- Model distillation for on-device deployment
- Visualization of attention maps for explainability

---

## Author

Developed by **Vivek K.**, AI/ML + Cybersecurity Researcher  
_“Security isn't optional — it's learnable.”_

---

## Citation

If you use this code or architecture, please cite the repo and acknowledge Vivek's work.
