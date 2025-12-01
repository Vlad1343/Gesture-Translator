# src/train.py
import json
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from config import DATA_DIR, MODEL_PATH, WINDOW_SIZE, WINDOW_STRIDE
from model import GestureLSTM

BATCH_SIZE = 32
EPOCHS = 45
LR = 1e-3
VAL_SPLIT = 0.2
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("label_map.json", "r") as f:
    label_map = json.load(f)
labels = {v: int(k) for k, v in label_map.items()}


class GestureWindowDataset(Dataset):
    """Slide a fixed window over recordings so training matches inference."""

    def __init__(self, data_dir, window_size, stride):
        self.window_size = window_size
        self.samples = []
        self.targets = []

        for gesture_name in os.listdir(data_dir):
            gesture_dir = os.path.join(data_dir, gesture_name)
            label_idx = labels.get(gesture_name)
            if label_idx is None or not os.path.isdir(gesture_dir):
                continue

            for file in os.listdir(gesture_dir):
                if not file.endswith(".npy"):
                    continue
                seq = np.load(os.path.join(gesture_dir, file))
                if len(seq) < window_size:
                    continue

                for start in range(0, len(seq) - window_size + 1, stride):
                    window = seq[start:start + window_size]
                    self.samples.append(torch.tensor(window, dtype=torch.float32))
                    self.targets.append(label_idx)

        if not self.samples:
            raise RuntimeError("No training samples were found. Record gestures first.")

        self.targets = torch.tensor(self.targets, dtype=torch.long)
        self.num_features = self.samples[0].shape[1]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.targets[idx]


def build_dataloaders(dataset):
    val_len = max(1, int(len(dataset) * VAL_SPLIT))
    train_len = len(dataset) - val_len
    generator = torch.Generator().manual_seed(SEED)
    train_set, val_set = random_split(dataset, [train_len, val_len], generator=generator)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    return train_loader, val_loader


def run_epoch(loader, model, criterion, optimizer=None):
    is_training = optimizer is not None
    model.train(mode=is_training)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        if is_training:
            optimizer.zero_grad()

        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        if is_training:
            loss.backward()
            optimizer.step()

        preds = torch.argmax(outputs, dim=1)
        total_correct += (preds == batch_y).sum().item()
        total_examples += batch_y.size(0)
        total_loss += loss.item()

    avg_loss = total_loss / max(1, len(loader))
    accuracy = total_correct / max(1, total_examples)
    return avg_loss, accuracy


def main():
    dataset = GestureWindowDataset(DATA_DIR, WINDOW_SIZE, WINDOW_STRIDE)
    train_loader, val_loader = build_dataloaders(dataset)

    input_size = dataset.num_features
    num_classes = len(labels)
    model = GestureLSTM(input_size=input_size, num_classes=num_classes).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(train_loader, model, criterion, optimizer)
        val_loss, val_acc = run_epoch(val_loader, model, criterion)
        print(
            f"Epoch {epoch:03d}/{EPOCHS} "
            f"| train loss {train_loss:.4f}, acc {train_acc:.3f} "
            f"| val loss {val_loss:.4f}, acc {val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)

    print(f"Training complete. Best val acc: {best_val_acc:.3f}. Model saved to {MODEL_PATH}.")


if __name__ == "__main__":
    main()
