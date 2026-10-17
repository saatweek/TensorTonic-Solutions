import numpy as np

def row_summary(data: list, threshold: float) -> np.ndarray:
    """
    Returns a float64 array of shape (3, m, n): mask, any-row, all-row.
    """
    a = np.array(data, dtype=np.float64)
    e_mask = np.where(a>threshold, 1, 0)
    any_mask = a*np.any(a>threshold, axis=1)[:, np.newaxis]
    all_mask = a*np.all(a>threshold, axis=1)[:, np.newaxis]

    return np.stack([e_mask, any_mask, all_mask])
    
