import torch
from torch.utils.data import DataLoader

def compute_stats(dataset):
    loader = DataLoader(dataset, batch_size=128)

    all_data = []
    for x in loader:
        all_data.append(x)

    all_data = torch.cat(all_data, dim=0)

    mean = all_data.mean(dim=0)
    std = all_data.std(dim=0) + 1e-8

    return mean, std


def normalize(x, mean, std):
    return (x - mean) / std