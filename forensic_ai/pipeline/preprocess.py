import numpy as np

def vectorize(record):
    return np.array([
        record.get("event_id", 0) or 0,
        record.get("user_id", 0) or 0,
        record.get("timestamp", 0) or 0
    ], dtype=np.float32)

def normalize(data):
    return (data - np.mean(data)) / (np.std(data) + 1e-8)