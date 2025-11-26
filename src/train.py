# src/train.py
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import GestureLSTM

DATA_DIR = "data/raw"
MODEL_DIR = "models"
BATCH_SIZE = 8
EPOCHS = 100
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load label map
import json
with open("label_map.json", "r") as f:
    label_map = json.load(f)  # e.g., {"0":"yes","1":"no",...}
labels = {v:int(k) for k,v in label_map.items()}  # invert mapping

# Dataset for gesture sequences
class GestureDataset(Dataset):
    def __init__(self, data_dir):
        self.sequences = []
        self.targets = []

        for gesture_name in os.listdir(data_dir):
            gesture_dir = os.path.join(data_dir, gesture_name)
            if not os.path.isdir(gesture_dir):
                continue
            for file in os.listdir(gesture_dir):
                if file.endswith(".npy"):
                    seq = np.load(os.path.join(gesture_dir, file))
                    self.sequences.append(seq)
                    self.targets.append(labels[gesture_name])

        self.sequences = [torch.tensor(s, dtype=torch.float32) for s in self.sequences]
        self.targets = torch.tensor(self.targets, dtype=torch.long)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

# collate_fn to pad sequences to same length
def collate_fn(batch):
    sequences, targets = zip(*batch)
    lengths = [len(seq) for seq in sequences]
    max_len = max(lengths)
    padded = [torch.cat([seq, torch.zeros(max_len - len(seq), seq.size(1))], dim=0) for seq in sequences]
    return torch.stack(padded), torch.tensor(targets)

# Prepare dataset
dataset = GestureDataset(DATA_DIR)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# Model
input_size = dataset[0][0].shape[1]  # number of keypoints per frame
num_classes = len(labels)
model = GestureLSTM(input_size=input_size, num_classes=num_classes).to(DEVICE)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(dataloader):.4f}")

# Save model
os.makedirs(MODEL_DIR, exist_ok=True)
torch.save(model.state_dict(), os.path.join(MODEL_DIR, "classifier.pth"))
print("Training complete. Model saved.")




# from dataset import GestureDataset
# dataset = GestureDataset(data_dir="data/raw")
# print(f"Loaded {len(dataset)} sequences")
# print(f"Shape of first sequence: {dataset[0][0].shape}")