import numpy as np

def filter_and_extract(data: list, row_start: int, row_stop: int, threshold: float) -> np.ndarray:
    """
    Returns matching values in row-major order as a 1D float64 array.
    """
    arr = np.array(data, dtype=np.float64)
    arr = arr[row_start:row_stop]
    selected = np.where(arr>threshold, arr, np.inf)
    res = selected.flatten()
    res = np.delete(res, np.where(res==np.inf))
    return res