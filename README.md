# Hybrid IDS for Anomaly Detection in Vehicle Diagnostic Data

This repository contains the development of an Intrusion Detection System (IDS) for detecting anomalous behavior in CAN based data, especially vehicle diagnostic communication using UDS.

The system analyzes vehicle diagnostic data obtained from **Unified Diagnostic Services (UDS)** request and response commands. It combines sequence-based deep learning models with anomaly-detection techniques to identify potentially malicious, invalid, or unusual diagnostic activity.

## Objectives

- Model normal patterns in UDS request and response sequences.
- Detect anomalous diagnostic behavior.
- Evaluate recurrent neural-network architectures, including GRU models.
- Support experimentation with hybrid IDS approaches.
- Optimize model hyperparameters using Optuna.

## Project Structure

```text
.
├── models/
│   ├── gru.py
│   └── tune_gru.py
├── data/
│   └── ...
├── notebooks/
│   └── ...
├── results/
│   └── ...
├── requirements.txt
└── README.md

