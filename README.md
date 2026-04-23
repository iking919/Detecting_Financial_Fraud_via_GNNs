<<<<<<< HEAD
# Detecting_Financial_Fraud_via_GNNs

Detecting Financial Fraud via Graph Neural Networks: a multi-dataset, graph-based learning project.

## Project Idea

This project explores how graph neural networks can help detect fraudulent financial activity across multiple public datasets. The core idea is to compare classic tabular baselines with graph-based models and see whether graph structure improves fraud detection.

The project is organized around three main goals:

1. Explore each dataset and understand its fraud patterns.
2. Preprocess raw transaction data into node and edge tables suitable for machine learning and graph learning.
3. Train and compare baseline models and GNN models such as GraphSAGE, GCN, and GAT.

## Datasets

The project uses three public fraud datasets:

1. PaySim Mobile Money Dataset
2. IEEE-CIS Fraud Detection Dataset
3. Elliptic Bitcoin Transaction Dataset

The dataset-specific file layout is documented in [data/README.md](data/README.md).

## Repository Layout

The notebook flow is intended to be run in order:

1. [01_eda.ipynb](notebooks/01_eda.ipynb) - exploratory data analysis
2. [02_preprocessing.ipynb](notebooks/02_preprocessing.ipynb) - data cleaning and graph/table preparation
3. [03_baseline_models.ipynb](notebooks/03_baseline_models.ipynb) - classical ML baselines
4. [04_GraphSAGE.ipynb](notebooks/04_GraphSAGE.ipynb) - GraphSAGE experiments
5. [05_GCN.ipynb](notebooks/05_GCN.ipynb) - GCN experiments
6. [06_GAT.ipynb](notebooks/06_GAT.ipynb) - GAT experiments
7. [07_Results.ipynb](notebooks/07_Results.ipynb) - final comparison and results

## How to Store the Data

The raw datasets are not committed to the repository. The easiest way to manage them is to keep the data in Google Drive and point the notebooks to that location.

Recommended setup:

1. Create a Google Drive folder for this project, for example `MyDrive/Detecting_Financial_Fraud_via_GNNs/`.
2. Store the raw datasets inside a `data/raw/` folder in Drive.
3. Save any generated outputs in `data/processed/` so preprocessing results are easy to reuse.
4. Update notebook paths if needed so they read from your Drive location instead of a local path.

If you use Google Colab, mount Drive at the start of the notebook and set the base path to your project folder:

```python
from google.colab import drive
drive.mount('/content/drive')

BASE_DIR = '/content/drive/MyDrive/Detecting_Financial_Fraud_via_GNNs'
RAW_DIR = f'{BASE_DIR}/data/raw'
PROCESSED_DIR = f'{BASE_DIR}/data/processed'
```

If you run locally, keep the same folder structure in a synced Drive folder or change the path constants in the notebooks to match your machine.

## How to Run the Project

The project is notebook-driven. A typical run looks like this:

1. Open the repository in VS Code or Jupyter.
2. Set up a Python environment with the libraries used in the notebooks.
3. Place the datasets in the correct `data/raw/` locations, or point the notebooks to your Google Drive folder.
4. Run the notebooks in sequence from `01_eda.ipynb` through `07_Results.ipynb`.

Suggested execution order:

1. Start with [01_eda.ipynb](notebooks/01_eda.ipynb) to inspect the datasets.
2. Run [02_preprocessing.ipynb](notebooks/02_preprocessing.ipynb) to create cleaned and graph-ready files.
3. Run [03_baseline_models.ipynb](notebooks/03_baseline_models.ipynb) to build baseline classifiers.
4. Run the GNN notebooks: [04_GraphSAGE.ipynb](notebooks/04_GraphSAGE.ipynb), [05_GCN.ipynb](notebooks/05_GCN.ipynb), and [06_GAT.ipynb](notebooks/06_GAT.ipynb).
5. Finish with [07_Results.ipynb](notebooks/07_Results.ipynb) to compare model performance.

## Practical Notes

The notebooks assume that the input data is available before preprocessing begins. If you change the Drive folder name or move the project, update the file paths in the notebooks accordingly.

The processed files in `data/processed/` are meant to be reused across notebooks, so you usually only need to regenerate them when the raw inputs or preprocessing logic changes.

## Expected Outcome

By the end of the workflow, you should have:

1. Exploratory analysis of all three datasets.
2. Processed node and edge files for classical ML and GNN training.
3. Baseline model results for comparison.
4. GNN results for GraphSAGE, GCN, and GAT.
5. A final results summary in the last notebook.
=======
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
