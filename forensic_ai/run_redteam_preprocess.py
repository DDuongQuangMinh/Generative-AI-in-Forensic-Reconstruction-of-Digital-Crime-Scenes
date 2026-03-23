import json
from pipeline.redteam_preprocess import build_redteam_sequences

seqs = build_redteam_sequences("data/raw/redteam.txt")

with open("data/processed/redteam_sequences.json", "w") as f:
    json.dump(seqs, f)

print("Redteam processed")