## Results Overview

This folder contains final evaluation results for **PaySim**, **Elliptic**, and **IEEE-CIS**, comparing baseline models (Logistic Regression, MLP, Random Forest) with GNNs (GCN, GAT, GraphSAGE).

### Files

* **ROC Curves:**
  `*_roc_comparisons.png` – Overlayed ROC curves for all models per dataset.

* **Confusion Matrices:**
  `*_confusion_matrices.png` – Side-by-side matrices showing fraud detection performance.

* **Performance Summary:**
  `model_performance_summary.csv` – Includes AUC-PR, AUC-ROC, F1, Precision, and Recall (fraud class).

### Key Insights

* GNNs outperform baselines on more complex datasets (Elliptic, IEEE-CIS).
* **GraphSAGE (full-batch)** provides the best overall balance of precision and recall.
* Mini-batch GNNs achieve high recall but often suffer from very low precision.
* Random Forest is a strong tabular baseline, especially on PaySim.

### Notes

* Results reflect **severe class imbalance**; AUC-PR and F1 are the most meaningful metrics.
* Threshold tuning was applied using validation data.

### Recommendation

* Use **GraphSAGE (full-batch)** for graph-based fraud detection.
* Use **Random Forest or MLP** for tabular-only settings, especially in more simplistic datasets.
