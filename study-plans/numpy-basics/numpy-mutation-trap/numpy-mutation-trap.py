import numpy as np

def original_and_clipped(data: list, row_idx: int, lo: float, hi: float) -> np.ndarray:
    """
    Returns a (2, n) float64 array: original row, then clipped row.
    """
    data = np.array(data, dtype=np.float64)
    selected_row = data[row_idx, :]
    mod_row = np.select([selected_row<lo, selected_row>hi], [lo, hi], selected_row)
    return np.stack([selected_row, mod_row])
