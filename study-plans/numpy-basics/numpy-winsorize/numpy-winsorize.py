import numpy as np

def winsorize(data: list, lo_q: float, hi_q: float) -> np.ndarray:
    """
    Returns float64 slices of clipped values, lower mask, and upper mask.
    """
    data = np.array(data, dtype=np.float64)
    lo = np.percentile(data, lo_q, axis=0)
    hi = np.percentile(data, hi_q, axis=0)
    clipped = np.clip(data, lo, hi)
    lo_mask = np.where(data<lo, 1, 0)
    hi_mask = np.where(data>hi, 1, 0)

    return np.stack([clipped, lo_mask, hi_mask])
    
