# Ethereum Security Monitoring System

An end-to-end framework for detecting smart contract vulnerabilities and on-chain abuse patterns (phishing, flash-loan attacks) on Ethereum. This repository contains:

- **Smart Contract Vulnerability Detection** using CodeBERT fine-tuning  
- **On-chain Abuse Detection**  
  - **GCN** (supervised) for phishing/scam address classification  
  - **GAE + KMeans** (unsupervised) for flash-loan anomaly detection  
- **Transaction Flow Visualization** with NetworkX  

---

## 📂 Repository Layout

- `On-chain anomaly detection/`  
  On-chain anomaly detection module including phishing and flash loan attacks detection:
  - `fraud_address_detection_model/`: Folder containing the GCN model and its configurations.
  - `Build_the_capital_flow.py`: NetworkX transaction flow visualizer.
  - `Dataset.csv`: Example dataset for testing and validation.
  - `data_processing.py`: Converts CSV data to PyTorch Geometric (PyG) graph format.
  - `fraud_address_detection_model.py`: GCN supervised phishing detection model.
  - `fraud_address_detection_test.py`: Interactive phishing address prediction and testing script.
  - `unsupervised_flashloan_detection_model.py`: Graph AutoEncoder (GAE) + KMeans flash loan attack detector.

- `smartbugs-curated-main/`  
  Smart contract vulnerability detection module based on the SmartBugs dataset:
  - `dataset/`: Contains curated smart contract vulnerability data.
  - `scripts/`: Helper scripts for preprocessing and analysis.
  - `ICSE2020_curated_69.txt`: Curated dataset list used in ICSE2020.
  - `LICENSE`: Project license.
  - `README.md`: Module-specific readme and usage.
  - `data_processing.py`: Loads and processes the SmartBugs dataset, token/line masking.
  - `main.py`: CodeBERT fine-tuning and inference script.
  - `test.py`: Evaluation and testing script.
  - `versions.csv`: Tracks dataset versions and changes.

- `environment.yml`  
  Conda environment specification to replicate the project environment.

- `README.md`  
  Project overview, usage instructions, and repository structure (this file).
