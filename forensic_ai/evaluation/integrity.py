import hashlib

def hash_data(x):
    return hashlib.sha256(str(x).encode()).hexdigest()