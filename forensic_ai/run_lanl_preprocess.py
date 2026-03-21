import pandas as pd

def build_sequences(file_path, seq_len=10, max_rows=100000):
    # Load LANL format
    df = pd.read_csv(file_path, sep=",", nrows=max_rows)

    # Encode categorical columns
    for col in df.columns:
        df[col] = df[col].astype("category").cat.codes

    data = df.values

    sequences = []
    for i in range(len(data) - seq_len):
        sequences.append(data[i:i+seq_len].flatten().tolist())

    return sequences