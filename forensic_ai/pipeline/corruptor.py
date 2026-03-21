import random
import copy

def corrupt_record(record):
    corrupted = copy.deepcopy(record)

    # Simulate timestomping (remove timestamps)
    if random.random() < 0.5:
        corrupted["timestamp"] = None

    # Simulate log deletion
    if random.random() < 0.3:
        corrupted["event_id"] = None

    return corrupted