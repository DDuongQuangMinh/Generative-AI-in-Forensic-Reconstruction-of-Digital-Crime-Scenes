import pandas as pd

def build_sequences(csv_path, seq_len=10):
    df = pd.read_csv(csv_path)

    # Encode categorical
    df["source_user"] = df["source_user"].astype("category").cat.codes
    df["destination_user"] = df["destination_user"].astype("category").cat.codes
    df["event_type"] = df["event_type"].astype("category").cat.codes

    data = df[["source_user", "destination_user", "event_type"]].values

    sequences = []
    for i in range(len(data) - seq_len):
        sequences.append(data[i:i+seq_len].flatten().tolist())

    return sequences