## Datasets

This project uses three public datasets for financial fraud detection:

1. **PaySim Mobile Money Dataset**
   https://www.kaggle.com/datasets/ealaxi/paysim1

2. **IEEE-CIS Fraud Detection Dataset**
   https://www.kaggle.com/datasets/lnasiri007/ieeecis-fraud-detection

3. **Elliptic Bitcoin Dataset**
   https://www.kaggle.com/datasets/ellipticco/elliptic-data-set

---

## Data Access (Kaggle API)

Raw datasets are **not manually downloaded or stored in this repository**.
Instead, the pipeline uses a **Kaggle API key (`kaggle.json`)** to programmatically download all datasets during preprocessing.

### Setup

1. Upload your `kaggle.json` file in the notebook environment
2. The preprocessing notebook will:

   * Authenticate with Kaggle
   * Download all required datasets
   * Extract them automatically

---

## Processing Pipeline

All preprocessing is handled in:

```
notebooks/02_preprocessing.ipynb
```

### What the pipeline does:

* Downloads raw datasets via Kaggle API
* Merges and cleans raw files (especially IEEE-CIS)
* Handles missing values and feature scaling
* Creates graph structures (nodes + edges)
* Assigns labels and unique IDs (`tx_id`)
* Applies dataset-specific fixes (e.g., Elliptic label remapping)

---

## Output Data (Google Drive)

After preprocessing, all processed data is packaged and stored as:

```
/content/drive/MyDrive/GNN_fraud_Project/processed_data.zip
```

When extracted, this contains:

```
data/processed/paysim_nodes.csv
data/processed/paysim_edges.csv

data/processed/ieee_train_nodes.csv
data/processed/ieee_train_edges.csv
data/processed/ieee_test_nodes.csv
data/processed/ieee_test_edges.csv

data/processed/elliptic_nodes.csv
data/processed/elliptic_edges.csv
```

---

## Use for Baseline Models

The `*_nodes.csv` files are used directly for tabular models:

* Logistic Regression
* MLP
* Random Forest

Each row represents a transaction with:

* Scaled numerical features
* `label` (fraud vs non-fraud)

---

## Use for Graph Neural Networks

The combination of node and edge files enables GNN training:

* GCN
* GAT
* GraphSAGE

### Structure

* **Nodes:** transaction features
* **Edges:** relationships between transactions

### Training Setup

* Full-batch and mini-batch training
* Mask-based splits (`train/val/test`)
* Imbalance-aware loss functions

---

## Notes

* No raw data is stored in this repository
* All data is **reproducibly generated via the preprocessing notebook**
* Google Drive is used to persist processed datasets across sessions
* This setup ensures consistency across experiments and environments

---

## Summary

* Fully automated data pipeline via Kaggle API
* Centralized processed dataset (`processed_data.zip`)
* Supports both tabular ML and graph-based learning
* Designed for reproducibility in Colab environments
