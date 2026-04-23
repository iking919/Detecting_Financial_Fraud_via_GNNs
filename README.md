# Detecting Financial Fraud via Graph Neural Networks

A multi-dataset study exploring financial fraud detection using **Graph Neural Networks (GNNs)** compared against traditional machine learning baselines. This project focuses on leveraging **graph structure and relational patterns** to improve fraud detection in highly imbalanced datasets.

---

## Overview

Fraud detection is inherently challenging due to:

* Extreme class imbalance
* Evolving attack patterns
* Hidden relationships between entities

This project addresses these challenges by combining:

* **Tabular models** (Logistic Regression, MLP, Random Forest)
* **Graph-based models** (GCN, GAT, GraphSAGE)

across three real-world datasets.

---

## Datasets

* **PaySim** – synthetic mobile money transactions
* **IEEE-CIS** – large-scale real-world fraud dataset
* **Elliptic** – Bitcoin transaction network

All datasets are **automatically downloaded via Kaggle API** and processed into graph-ready formats.

---

## Models

### Baselines

* Logistic Regression
* MLP
* Random Forest

### Graph Neural Networks

* **GCN (Graph Convolutional Network)**
* **GAT (Graph Attention Network)**
* **GraphSAGE**

Each GNN is evaluated under:

* Full-batch training
* Mini-batch (sampling-based) training

---

## Pipeline

The project follows a fully reproducible pipeline:

1. **EDA** – dataset exploration and imbalance analysis
2. **Preprocessing** – Kaggle download, cleaning, graph construction
3. **Baselines** – tabular model benchmarking
4. **GNN Training** – GraphSAGE, GCN, GAT
5. **Evaluation** – ROC curves, confusion matrices, metric comparison

All steps are implemented as notebooks in `/notebooks`.

---

## Project Structure

```
├── notebooks/        # Full reproducible pipeline (EDA → Results)
├── models/           # Trained GNN model checkpoints (.pth)
├── results/          # ROC curves, confusion matrices, summary CSV
├── data/             # (Generated via preprocessing, not stored)
```

---

## Key Results

* **Graph-based models outperform baselines** on relational datasets
* **GraphSAGE (full-batch)** achieves the best overall performance
* Mini-batch GNNs tend to favor **high recall but low precision**
* Random Forest remains competitive for purely tabular data (PaySim)

---

## Reproducibility

* Requires a valid **Kaggle API key (`kaggle.json`)**
* Data is automatically downloaded and processed
* Processed data is saved to Google Drive as:

  ```
  GNN_fraud_Project/processed_data.zip
  ```

Run notebooks in order:

```
01 → 02 → 03 → 04 → 05 → 06 → 07
```

---

## Takeaways

* Incorporating **graph structure significantly improves fraud detection**
* **Precision-recall tradeoffs** are critical in imbalanced settings
* Model choice depends on:

  * Data structure (tabular vs graph)
  * Deployment constraints (precision vs recall)

---

## Future Work

* Temporal GNNs for dynamic fraud detection
* Advanced sampling strategies for mini-batch training
* Feature engineering using domain-specific signals

---

Developed by Izaak King, Haoyuan Chen, Bryan Gelnett
>>>>>>> 8d0764ccca52ad3e81ce3953f6693ae186fa8a5a
