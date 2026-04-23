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
