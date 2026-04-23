## Notebooks Overview (Reproducible Pipeline)

This folder contains the full end-to-end pipeline for financial fraud detection using both traditional machine learning models and graph neural networks (GNNs). Each notebook is designed to be **self-contained and reproducible**, following a logical progression from data exploration to final evaluation.

---

### 01_eda.ipynb — Exploratory Data Analysis

* Initial exploration of PaySim, Elliptic, and IEEE-CIS datasets
* Class imbalance analysis and distribution visualization
* Feature inspection and dataset-specific observations

---

### 02_preprocessing.ipynb — Data Preprocessing

* Data cleaning and handling missing values
* Feature scaling and encoding
* Graph construction (nodes + edges) for GNN models
* Train/validation/test splits (chronological where applicable)

---

### 03_baseline_models.ipynb — Baseline Models

* Logistic Regression, MLP, and Random Forest implementations
* GPU-accelerated training where applicable
* Threshold tuning for imbalanced classification
* Baseline performance benchmarks

---

### 04_GraphSAGE.ipynb — GraphSAGE Model

* Full-batch and mini-batch GraphSAGE implementations
* Training on graph-structured data
* Evaluation with fraud-focused metrics (AUC-PR, F1, recall, precision)

---

### 05_GCN.ipynb — Graph Convolutional Network (GCN)

* GCN architecture for node classification
* Comparison of full-batch vs mini-batch training
* Performance analysis across datasets

---

### 06_GAT.ipynb — Graph Attention Network (GAT)

* Attention-based graph learning implementation
* Multi-head attention setup
* Evaluation and comparison with other GNNs

---

### 07_Results.ipynb — Final Results & Visualization

* Aggregated model comparisons across all datasets
* ROC curves and confusion matrices
* Final performance tables (AUC-PR, AUC-ROC, F1, Precision, Recall)
* Key insights and conclusions

---

## Reproducibility Notes

* Notebooks are intended to be run **in order (01 → 07)**
* Each notebook includes all necessary steps for its stage of the pipeline
* Ensure datasets are placed in the expected `/data/processed/` directory
* GPU is recommended for GNN training, but CPU execution is supported (slower)

---

## Summary

This notebook suite provides a complete, reproducible workflow for:

* Fraud detection on tabular and graph data
* Benchmarking baseline vs GNN models
* Evaluating performance under extreme class imbalance
