import numpy as np

def row_extremes(data: list) -> np.ndarray:
    """
    Returns float64 rows of maxima, maximum indices, minima, and minimum indices.
    """
    data = np.array(data, dtype = np.float64)
    max_val = np.max(data, axis=1)
    min_val = np.min(data, axis=1)
    max_idx = np.argmax(data, axis=1)
    min_idx = np.argmin(data, axis=1)

    return np.stack([max_val, max_idx, min_val, min_idx])
