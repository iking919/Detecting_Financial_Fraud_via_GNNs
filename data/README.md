## Datasets

This project uses three public datasets for financial fraud detection:

1. **PaySim Mobile Money Dataset**  
   [Kaggle link](https://www.kaggle.com/datasets/ealaxi/paysim1)

2. **IEEE-CIS Fraud Detection Dataset**  
   [Kaggle link](https://www.kaggle.com/datasets/lnasiri007/ieeecis-fraud-detection)  
   **Note:** This dataset is already split into training and test sets.  
   We will use `train_transaction.csv` + `train_identity.csv` for training,  
   and `test_transaction.csv` + `test_identity.csv` for evaluation.

3. **Elliptic Data Set**  
   [Kaggle link](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)

### Raw Data File Structure

Place the downloaded raw datasets in the following directories:

data/raw/paysim/PS_20174392719_1491204439457_log.csv

data/raw/ieee_fraud/train_identity.csv
data/raw/ieee_fraud/train_transaction.csv
data/raw/ieee_fraud/test_identity.csv
data/raw/ieee_fraud/test_transaction.csv

data/raw/elliptic/elliptic_txs_classes.csv
data/raw/elliptic/elliptic_txs_edgelist.csv
data/raw/elliptic/elliptic_txs_features.csv


**Note:** Raw data is not stored in this repository.  

---

## Processed Data

After running the preprocessing scripts, the following structured CSVs are generated:

data/processed/paysim_nodes.csv
data/processed/paysim_edges.csv

data/processed/ieee_train_nodes.csv
data/processed/ieee_train_edges.csv
data/processed/ieee_test_nodes.csv
data/processed/ieee_test_edges.csv

data/processed/elliptic_nodes.csv
data/processed/elliptic_edges.csv

These CSVs contain normalized features, unique transaction IDs (`tx_id`), and labels (`label`) for supervised learning.

---

## Use for Baseline Machine Learning Models

The node CSVs, containing normalized features and labels, can be used directly as tabular datasets for classical machine learning models such as:

- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Decision Trees  
- Random Forests  

Each row represents a transaction with its relevant features, and the `label` column indicates fraud status.  
These datasets allow quick training, evaluation, and benchmarking of classical models, helping to establish baselines and identify predictive features.

---

## Use for Graph Neural Network Models

The combination of **nodes CSVs** and **edges CSVs** provides a complete graph representation suitable for Graph Neural Networks (GNNs) such as:

- Graph Convolutional Networks (GCN)  
- Graph Attention Networks (GAT)  
- GraphSAGE  

- **Nodes:** individual transactions with associated features  
- **Edges:** relationships between transactions (e.g., shared accounts, emails, devices, or fund propagation)  

These files can be converted to PyTorch Geometric or DGL datasets, allowing models to leverage both node attributes and network topology.  
GNNs can capture higher-order patterns like coordinated fraudulent behaviors, which classical tabular models may miss.
