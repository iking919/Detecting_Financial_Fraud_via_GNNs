## Models Directory Overview

This folder contains all trained **Graph Neural Network (GNN) model checkpoints** used in the project. Models are organized by architecture:

* `graphSAGE/`
* `GCN/`
* `GAT/`

Each subfolder includes trained models for all three datasets:

* **PaySim**
* **Elliptic**
* **IEEE-CIS**

---

## File Structure

Within each model folder, checkpoints follow this naming convention:

```
<Dataset>_<TrainingType>.pth
```

### Example

* `PaySim_Full_batch.pth`
* `PaySim_Mini_batch.pth`
* `Elliptic_Full_batch.pth`
* `Elliptic_Mini_batch.pth`
* `IEEE-CIS_Full_batch.pth`
* `IEEE-CIS_Mini_batch.pth`

---

## Model Variants

### Full-batch

* Trained on the entire graph at once
* More stable and typically better precision
* Preferred for final evaluation and reporting

### Mini-batch

* Uses neighborhood sampling for scalability
* Faster and more memory-efficient
* Often achieves higher recall but lower precision

---

## Usage

To load a model:

```python
model.load_state_dict(torch.load("path_to_model.pth"))
model.eval()
```

Ensure that:

* The model architecture matches the saved checkpoint
* Input feature dimensions are consistent with training

---

## Notes

* These checkpoints correspond to the results reported in the `results/` directory
* Models were trained with **class imbalance handling** (e.g., `pos_weight`) and **threshold tuning**
* Performance may vary if re-trained due to stochasticity (random seeds, sampling)

---

## Summary

* **GraphSAGE** models provide the best overall performance across datasets
* **GCN and GAT** offer competitive alternatives with different trade-offs
* Full-batch models are generally more reliable for fraud detection tasks
