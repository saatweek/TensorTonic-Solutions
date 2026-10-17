import numpy as np

def normalize(data: list) -> np.ndarray:
    """
    Returns a float64 matrix standardized independently by column.
    """
    data = np.array(data, dtype=np.float64)
    mean = np.mean(data, axis=0, keepdims=True, dtype=np.float64)
    std = np.std(data, axis = 0, keepdims=True, ddof=0, dtype=np.float64)
    return ((data-mean)/std).astype(np.float64)
