# Ethereum Security Monitoring System

An end-to-end framework for detecting smart contract vulnerabilities and on-chain abuse patterns (phishing, flash-loan attacks) on Ethereum. This repository contains:

- **Smart Contract Vulnerability Detection** using CodeBERT fine-tuning  
- **On-chain Abuse Detection**  
  - **GCN** (supervised) for phishing/scam address classification  
  - **GAE + KMeans** (unsupervised) for flash-loan anomaly detection  
- **Transaction Flow Visualization** with NetworkX  

---

## 📂 Repository Layout

- `contracts_vuln_detection/`  
  Contains modules and scripts related to smart contract vulnerability detection tasks:
  - `data_processing.py`: Loads the SmartBugs dataset and performs token/line masking.
  - `main.py`: CodeBERT fine-tuning and inference script.
  - `preprocess_single_file.py`: Helper script to preprocess a single Solidity (`.sol`) file.

- `onchain_abuse_detection/`  
  A module focused on on-chain abuse detection, including phishing and flash loan attacks:
  - `data_processing.py`: Converts CSV data to PyTorch Geometric (PyG) graph format.
  - `fraud_address_detection_model.py`: Graph Convolutional Network (GCN) model for supervised phishing detection.
  - `fraud_address_detection_test.py`: Interactive phishing prediction script.
  - `unsupervised_flashloan_detection_model.py`: Graph AutoEncoder (GAE) + KMeans flash-loan detector.
  - `Build_the_capital_flow.py`: Visualizes transaction flows using NetworkX.

- `environment.yml`  
  Conda environment specification to replicate the project environment.

- `README.md`  
  Project overview, usage instructions, and repository structure (this file).
