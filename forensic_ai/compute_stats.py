from pipeline.dataset import SequenceDataset
from pipeline.normalization import compute_stats
import torch

dataset = SequenceDataset("data/processed/lanl_sequences.json")

mean, std = compute_stats(dataset)

torch.save(mean, "mean.pt")
torch.save(std, "std.pt")

print("Normalization stats saved")