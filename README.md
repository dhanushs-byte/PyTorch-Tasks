# Week 3 - PyTorch Tasks 1 & 2

## Overview
This project covers essential PyTorch concepts, including custom datasets, DataLoaders, neural network training, model saving and loading, prediction, and checkpoint resuming.

## Task 1: Dataset and DataLoader
- Created a custom `CsvDataset` class using `Dataset`.
- Implemented `__init__()`, `__len__()`, and `__getitem__()`.
- Loaded the Breast Cancer Wisconsin dataset containing 569 samples and 30 features.
- Used `DataLoader` with a batch size of 32 and shuffling.
- Split the dataset into training and validation sets using `random_split()`.
- Tested `num_workers=0` and `num_workers=2`.
- Counted the batches in one epoch.

## Task 2: Train, Save and Load a Model
- Built a Multilayer Perceptron (MLP) using PyTorch.
- Trained the model for 30 epochs.
- Evaluated validation accuracy.
- Saved model weights using `torch.save(model.state_dict(), "model.pt")`.
- Loaded the weights into a new model and verified that the accuracy remained identical.
- Implemented a `predict()` function returning a predicted label and probability.

## Bonus Challenge
- Saved a training checkpoint containing model weights, optimizer state, and epoch number.
- Loaded the checkpoint and resumed training for five additional epochs.

## Technologies Used
- Python
- PyTorch
- Pandas
- Scikit-learn
- Joblib

## Project Files
- `task1_task2.py` — Complete implementation of Tasks 1 and 2.
- `breast_cancer.csv` — Dataset used for training and evaluation.
- `model.pt` — Saved model weights.
- `scaler.pkl` — Saved feature scaler for predictions.
- `checkpoint.pt` — Full training checkpoint.
- `screenshot.png` — Training output and model-loading verification.

## How to Run

Install the dependencies:

```bash
pip install torch pandas scikit-learn joblib
```

Run the program:

```bash
python task1_task2.py
```

The script creates the CSV if it does not exist and trains, saves, reloads, and evaluates the model.

## Expected Results
- Dataset size: 569 samples.
- Batch feature shape: `torch.Size([32, 30])`.
- Full-dataset batches per epoch: 18.
- Validation accuracy reported after training.
- Identical validation accuracy before saving and after loading.
- Successful prediction and checkpoint resuming.

*Note: Actual validation accuracy may vary depending on the training setup.*

## Learning Outcomes
This project strengthened my understanding of PyTorch datasets, batching, data loading, neural network training, model serialization, inference, and checkpoint-based training resumption.
