
# VAEWithAttention for Network Intrusion Detection

##  Abstract

This project explores an **unsupervised anomaly detection model** using a **Variational Autoencoder (VAE)** enhanced with an **attention mechanism**, tailored for detecting cyberattacks in network traffic data. It extracts compressed latent representations from benign traffic and detects intrusions by evaluating reconstruction loss, with a threshold-based detection strategy.

---

##  Use Case

- **Network Intrusion Detection (NIDS)**
- Particularly suitable when **benign traffic is abundant**, but **attack patterns are rare or unknown**
- Designed for real-time detection with minimal labeled data
- Datasets: CSIC, CIC-IoT2023, custom CSVs like `Benign.csv`, `XSS.csv`, etc.

---

## Why VAEWithAttention?

###  Pros

- **Unsupervised Learning**: Requires only benign data for training
- **Attention Mechanism**: Improves feature learning by weighting input features dynamically
- **Latent Representation + Loss**: Offers richer anomaly signals than raw reconstruction loss
- **Real-Time Capable**: Once trained, detection involves only forward pass + thresholding
- **Modular & Extendable**: Can be integrated with LightGBM or other downstream classifiers

###  Cons

- **Threshold Sensitivity**: Performance depends heavily on threshold tuning
- **False Positives**: Reconstruction-based models may misclassify unfamiliar benign traffic
- **Resource Intensive**: t-SNE and full-scale VAEs require GPU for faster training


---

##  Dataset: CIC-IoT2023

This project uses the **CIC-IoT-2023 dataset**, developed by the Canadian Institute for Cybersecurity, 
which provides labeled traffic for both benign and multiple attack categories in IoT network environments.

###  Supported Attack Types (13 total):

- DDoS, DoS, Backdoor, MITM, Infiltration, Reconnaissance, Web Attack, Malware, Botnet, Password Attack, XSS, SQL Injection, Command Injection

Each attack type is loaded from a separate CSV file and processed for reconstruction-based detection.


---

##  Architecture Overview

- **Encoder**: Attention layer → Linear → µ, logσ
- **Decoder**: Latent z → Linear layers → Reconstructed x
- **Loss**: MSE + KL divergence
- **Feature Output**: [Latent µ || reconstruction loss]

---

##  Evaluation Metrics

- Precision, Recall, F1-score (overall and per attack type)
- Threshold tuning via reconstruction loss
- Visualizations: t-SNE of latent features, loss distribution, PR curve

---

##  Future Work

- Add **LightGBM classifier** on top of extracted embeddings
- Replace attention with **Transformer encoders** for better context learning
- Explore **adaptive thresholding** instead of fixed cutoffs
- Use **contrastive learning** for better anomaly separation

---


