import numpy as np

def compare_correlations(a: list, b: list) -> np.ndarray:
    """
    Returns float64 correlation matrices for a, b, and their combined rows.
    """
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    comb = np.concatenate([a, b], axis=0)
    slice1 = np.corrcoef(a.T)
    slice2 = np.corrcoef(b.T)
    slice3 = np.corrcoef(comb.T)
    return np.stack([slice1, slice2, slice3])
    