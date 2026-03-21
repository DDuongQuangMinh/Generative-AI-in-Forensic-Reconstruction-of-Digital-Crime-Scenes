import pandas as pd

def build_sequences(csv_file, seq_len=10):
    df = pd.read_csv(csv_file)

    # Encode categorical fields
    df["source_user"] = df["source_user"].astype("category").cat.codes
    df["dest_user"] = df["destination_user"].astype("category").cat.codes
    df["event_type"] = df["event_type"].astype("category").cat.codes

    sequences = []

    data = df[["source_user", "dest_user", "event_type"]].values

    for i in range(len(data) - seq_len):
        sequences.append(data[i:i+seq_len].flatten())

    return sequences