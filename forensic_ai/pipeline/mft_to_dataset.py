import json

def convert_mft(records, output_file="data/raw/mft.json"):
    dataset = []

    for r in records:
        dataset.append({
            "event_id": r["inode"],  # proxy feature
            "user_id": r["size"],    # proxy feature
            "timestamp": r["mtime"] or 0
        })

    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=2)