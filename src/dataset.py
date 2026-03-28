# src/dataset.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

SEQ_LENGTH = 30  # number of frames per gesture sequence
DATA_DIR = "data/raw"

# map gestures to numeric labels
def load_label_map():
    gestures = sorted(os.listdir(DATA_DIR))
    return {g: i for i, g in enumerate(gestures)}

# custom PyTorch Dataset
class GestureDataset(Dataset):
    def __init__(self, data_dir=DATA_DIR, seq_length=SEQ_LENGTH):
        self.data_dir = data_dir
        self.seq_length = seq_length
        self.label_map = load_label_map()
        self.sequences = []  # list of sequences
        self.labels = []     # corresponding numeric labels
        self.load_data()

    def load_data(self):
        for gesture, label in self.label_map.items():
            gesture_dir = os.path.join(self.data_dir, gesture)
            if not os.path.exists(gesture_dir):
                continue
            for file in os.listdir(gesture_dir):
                if file.endswith(".npy"):
                    seq = np.load(os.path.join(gesture_dir, file), allow_pickle=True)
                    # pad or truncate to fixed seq_length
                    if len(seq) < self.seq_length:
                        pad = np.zeros((self.seq_length - len(seq), seq.shape[1]))
                        seq = np.vstack([seq, pad])
                    elif len(seq) > self.seq_length:
                        seq = seq[:self.seq_length]
                    self.sequences.append(seq)
                    self.labels.append(label)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        X = torch.tensor(self.sequences[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return X, y

# helper to get DataLoader
def get_dataloader(batch_size=8, shuffle=True):
    dataset = GestureDataset()
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)