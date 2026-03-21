import torch
from torch.utils.data import Dataset
import json

class ForensicDataset(Dataset):
    def __init__(self, path):
        with open(path) as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        r = self.data[idx]

        x = torch.tensor([
            r["inode"],
            r["size"],
            r["timestamp"]
        ], dtype=torch.float32)

        return x, x  # autoencoder


class SequenceDataset(Dataset):
    def __init__(self, path):
        with open(path) as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)