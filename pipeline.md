# GRU IDS - Current Pipeline

## 1. Overview

This project implements a GRU-based Intrusion Detection System (IDS)
using the ROAD dataset.

The current model learns the temporal patterns of CAN IDs using a
center-ID prediction approach.

The training pipeline is currently divided into:

1. Dataset loading
2. CAN ID encoding
3. Sliding-window generation
4. Embedding
5. GRU
6. Dropout
7. Softmax classification
8. Model training

The current implementation uses only benign ROAD data for training.

---

# 2. Project Structure

```text
Anomaly_detection_hybrid_IDS/
│
├── config/
│   ├── dataset.yaml
│   └── model_test.yaml
│
├── evaluation/
│
├── models/
│   └── gru.py
│
├── preprocessing/
│   ├── load_data.py
│   ├── sliding_window.py
│   └── __init__.py
│
├── training/
│   ├── train.py
│   └── __init__.py
│
├── tests/
│   ├── test_data_pipeline.py
│   ├── test_sliding_window.py
│   ├── test_model.py
│   └── __init__.py
│
├── .gitignore
├── main.py
├── requirements.txt
├── pipeline.md
└── README.md
