import pandas as pd
import json
import os

def build_sequences(file_path, seq_len=10, max_rows=10000):
    print("Reading file...")

    df = pd.read_csv(file_path, sep=",", nrows=max_rows)

    print("Rows loaded:", len(df))

    # Encode categorical columns
    for col in df.columns:
        df[col] = df[col].astype("category").cat.codes

    data = df.values

    sequences = []
    for i in range(len(data) - seq_len):
        sequences.append(data[i:i+seq_len].flatten().tolist())

    print("Sequences created:", len(sequences))

    return sequences


# 🚀 THIS PART WAS MISSING / WRONG
if __name__ == "__main__":
    input_path = "data/raw/auth.txt"
    output_path = "data/processed/lanl_sequences.json"

    # Ensure output folder exists
    os.makedirs("data/processed", exist_ok=True)

    seqs = build_sequences(input_path, max_rows=10000)

    with open(output_path, "w") as f:
        json.dump(seqs, f)

    print("✅ Saved to:", output_path)