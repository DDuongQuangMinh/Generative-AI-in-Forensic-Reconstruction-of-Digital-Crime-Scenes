import hashlib

def hash_artifact(data):
    data_bytes = str(data).encode()
    return hashlib.sha256(data_bytes).hexdigest()