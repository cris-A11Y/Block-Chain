# Ethereum Security Monitoring System

An end-to-end framework for detecting smart contract vulnerabilities and on-chain abuse patterns (phishing, flash-loan attacks) on Ethereum. This repository contains:

- **Smart Contract Vulnerability Detection** using CodeBERT fine-tuning  
- **On-chain Abuse Detection**  
  - **GCN** (supervised) for phishing/scam address classification  
  - **GAE + KMeans** (unsupervised) for flash-loan anomaly detection  
- **Transaction Flow Visualization** with NetworkX  

---

## 📂 Repository Layout
.
├── contracts_vuln_detection/
│ ├── data_processing.py # SmartBugs dataset loader & token/line masking
│ ├── main.py # CodeBERT fine-tuning & inference script
│ └── preprocess_single_file.py # Helper to preprocess a single .sol file
│
├── onchain_abuse_detection/
│ ├── data_processing.py # CSV → PyG Data conversion
│ ├── fraud_address_detection_model.py # GCN supervised phishing detector
│ ├── fraud_address_detection_test.py # Interactive phishing prediction script
│ ├── unsupervised_flashloan_detection_model.py # GAE + KMeans flash-loan detector
│ └── Build_the_capital_flow.py # NetworkX transaction-flow visualizer
│
├── environment.yml # Conda environment specification
└── README.md # You are here
