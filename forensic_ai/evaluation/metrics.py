import numpy as np

def rmse(a, b):
    return np.sqrt(((a - b) ** 2).mean())