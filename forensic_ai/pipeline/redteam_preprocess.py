import pandas as pd

def build_redteam_sequences(file_path, seq_len=10):
    # IMPORTANT: specify column names manually
    df = pd.read_csv(
        file_path,
        header=None,
        names=["time", "user", "source_pc", "destination_pc"]
    )

    # 🔧 Add missing fields to match LANL (9 features)
    df["dst_user"] = df["user"]
    df["auth_type"] = "none"
    df["logon_type"] = "none"
    df["auth_orient"] = "none"
    df["success"] = "success"

    # Reorder to match LANL structure
    df = df[
        [
            "time",
            "user",
            "dst_user",
            "source_pc",
            "destination_pc",
            "auth_type",
            "logon_type",
            "auth_orient",
            "success",
        ]
    ]

    # Encode categorical
    for col in df.columns:
        df[col] = df[col].astype("category").cat.codes

    data = df.values

    sequences = []
    for i in range(len(data) - seq_len):
        sequences.append(data[i:i+seq_len].flatten().tolist())

    return sequences