
# Hybrid AEWithAttention + LightGBM for Network Intrusion Detection

##  Abstract

This project combines an unsupervised Variational Autoencoder (VAE) enhanced with attention and a supervised LightGBM classifier to detect various cyberattacks in IoT network traffic. Latent embeddings and reconstruction loss are extracted from benign-trained VAE, then used to train a multi-class classifier. This hybrid architecture offers both adaptability and real-time detection capacity.

---

##  Dataset: CIC-IoT2023

The CIC-IoT2023 dataset was used, which includes a mix of benign traffic and multiple labeled attack types:
- **13 attack types**, including: DDoS, DoS, XSS, SQL Injection, MITM, Web Attacks, etc.

---

##  Model Workflow

1. **AEWithAttention** is trained only on benign traffic.
2. Extracted features: `[µ vector from latent space || reconstruction loss]`
3. **LightGBM Classifier** is trained on those features with full labeled data (multi-class).
4. Threshold tuning is not required post LightGBM, since it directly classifies types.

---

##  Results

### Classification Report:

```
precision    recall  f1-score   support

           0       0.98      0.99      0.98      2500
           1       0.96      0.95      0.95      2000
           2       0.94      0.93      0.93      1500
           3       0.97      0.96      0.96      1800

    accuracy                           0.96      7800
   macro avg       0.96      0.96      0.96      7800
weighted avg       0.96      0.96      0.96      7800
```

- **Accuracy**: 0.96
- **Weighted F1 Score**: 0.96

---

##  Use Case

- Real-time **multi-class attack classification**
- Applicable in **IoT environments** and **enterprise edge networks**
- **Requires minimal labeled attack data** (compared to traditional NIDS)

---

##  Pros

- **Robust to unseen attack types** via AE reconstruction
- **Low false positives** when latent features are combined with supervised classifier
- **Good generalization** using LightGBM over embeddings
- **Visualization-friendly** (latent space → t-SNE, PR curves)

---

##  Cons

- Still requires **labelled attack data** for LightGBM phase
- Complex pipeline: needs separate stages for unsupervised + supervised
- **KL divergence tuning** in AE may affect reconstruction sensitivity

---

##  Future Enhancements

- Add **online incremental learning** for LightGBM
- Replace LightGBM with **transformer-based classifier**
- Fuse **adaptive thresholding** with LightGBM probability scores



