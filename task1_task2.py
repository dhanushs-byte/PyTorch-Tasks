
import os
import multiprocessing
import pandas as pd
import torch
import torch.nn as nn
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader, random_split


# ==================================================
# TASK 1: CSV DATASET AND DATALOADER
# ==================================================

CSV_FILE = "breast_cancer.csv"
MODEL_FILE = "model.pt"
SCALER_FILE = "scaler.pkl"
CHECKPOINT_FILE = "checkpoint.pt"


class CsvDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)

        self.features = torch.tensor(
            df.drop(columns=["label"]).values,
            dtype=torch.float32
        )

        self.labels = torch.tensor(
            df["label"].values,
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


# ==================================================
# TASK 2: MULTILAYER PERCEPTRON
# ==================================================

class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(30, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x).squeeze(1)


def evaluate(model, loader, criterion=None):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for features, labels in loader:
            outputs = model(features)
            predictions = (torch.sigmoid(outputs) >= 0.5).float()

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            if criterion is not None:
                total_loss += criterion(outputs, labels).item() * labels.size(0)

    accuracy = 100.0 * correct / total

    if criterion is not None:
        return accuracy, total_loss / total

    return accuracy


def predict(features):
    """Predict one new row containing 30 raw feature values."""

    scaler = joblib.load(SCALER_FILE)
    model = MLP()

    model.load_state_dict(
        torch.load(
            MODEL_FILE,
            map_location="cpu",
            weights_only=True
        )
    )
    model.eval()

    scaled_features = scaler.transform([features])
    tensor_features = torch.tensor(
        scaled_features,
        dtype=torch.float32
    )

    with torch.no_grad():
        probability = torch.sigmoid(model(tensor_features)).item()

    label = int(probability >= 0.5)

    return label, probability


# ==================================================
# MAIN FUNCTION
# All executable code belongs inside this function
# ==================================================

def main():
    torch.manual_seed(42)

    # Create CSV if it does not exist
    if not os.path.exists(CSV_FILE):
        data = load_breast_cancer()

        df = pd.DataFrame(
            data.data,
            columns=data.feature_names
        )
        df["label"] = data.target
        df.to_csv(CSV_FILE, index=False)

        print("CSV dataset created successfully.")

    # Load CSV dataset
    dataset = CsvDataset(CSV_FILE)

    print("Dataset size:", len(dataset))

    # Split dataset into train and validation subsets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Fit scaler using training samples only
    scaler = StandardScaler()

    train_indices = train_dataset.indices
    train_features = dataset.features[train_indices].numpy()

    scaler.fit(train_features)

    # Transform all features using the training scaler
    dataset.features = torch.tensor(
        scaler.transform(dataset.features.numpy()),
        dtype=torch.float32
    )

    joblib.dump(scaler, SCALER_FILE)

    # DataLoader with batch size 32
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )

    features, labels = next(iter(loader))
    print("Batch features:", features.shape)

    # Test num_workers = 0 and num_workers = 2
    for workers in [0, 2]:
        test_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=workers
        )

        batch_count = sum(1 for _ in test_loader)

        print(
            f"num_workers={workers}: "
            f"{batch_count} batches"
        )

    # Main training DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    # Count batches across the full dataset
    full_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )

    batches = sum(1 for _ in full_loader)

    print("Batches in one full-dataset epoch:", batches)

    # Initialize model, loss and optimizer
    model = MLP()
    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    # Train for 30 epochs
    for epoch in range(30):
        model.train()
        running_loss = 0.0

        for features, labels in train_loader:
            optimizer.zero_grad()

            outputs = model(features)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)

        epoch_loss = running_loss / len(train_dataset)

        if (epoch + 1) % 10 == 0:
            accuracy = evaluate(model, val_loader)

            print(
                f"Epoch [{epoch + 1}/30], "
                f"Loss: {epoch_loss:.4f}, "
                f"Validation Accuracy: {accuracy:.2f}%"
            )

    # Evaluate before saving
    accuracy_before = evaluate(model, val_loader)

    print(
        f"\nAccuracy before saving: "
        f"{accuracy_before:.2f}%"
    )

    # Save model weights
    torch.save(model.state_dict(), MODEL_FILE)

    print("Model saved as model.pt")

    # Create a new model and reload the weights
    loaded_model = MLP()

    loaded_model.load_state_dict(
        torch.load(
            MODEL_FILE,
            map_location="cpu",
            weights_only=True
        )
    )

    loaded_model.eval()

    # Evaluate after loading
    accuracy_after = evaluate(loaded_model, val_loader)

    print(
        f"Accuracy after loading: "
        f"{accuracy_after:.2f}%"
    )

    print(
        "Identical accuracy:",
        abs(accuracy_before - accuracy_after) < 1e-9
    )

    # Predict one new raw sample
    raw_data = load_breast_cancer()
    sample = raw_data.data[0].tolist()

    predicted_label, probability = predict(sample)

    print("\nPrediction label:", predicted_label)
    print(f"Prediction probability: {probability:.4f}")

    print(
        "Predicted class:",
        "Benign" if predicted_label == 1 else "Malignant"
    )

    # ==============================================
    # BONUS: Save full training checkpoint
    # ==============================================

    torch.save(
        {
            "epoch": 30,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        },
        CHECKPOINT_FILE
    )

    print("\nFull checkpoint saved.")

    # Load checkpoint into a new model and optimizer
    resume_model = MLP()

    resume_optimizer = torch.optim.Adam(
        resume_model.parameters(),
        lr=0.001
    )

    checkpoint = torch.load(
        CHECKPOINT_FILE,
        map_location="cpu",
        weights_only=True
    )

    resume_model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    resume_optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    start_epoch = checkpoint["epoch"]

    print("Checkpoint loaded.")
    print("Resuming from epoch:", start_epoch + 1)

    # Resume training for 5 more epochs
    for epoch in range(start_epoch, start_epoch + 5):
        resume_model.train()

        for features, labels in train_loader:
            resume_optimizer.zero_grad()

            outputs = resume_model(features)
            loss = criterion(outputs, labels)

            loss.backward()
            resume_optimizer.step()

    print("Training resumed successfully.")


# ==================================================
# WINDOWS MULTIPROCESSING ENTRY POINT
# ==================================================

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
