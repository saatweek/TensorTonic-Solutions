import numpy as np

def tile_diff(data: list, reps: int) -> np.ndarray:
    """
    Returns float64 slices of tiled values and next-row differences with a final zero row.
    """
    data = np.array(data, dtype=np.float64)
    rep = np.tile(data, (reps, 1))
    d = rep.copy()
    d = np.diff(d, axis=0)
    d = np.pad(d, ([[0,1], [0,0]]), mode="constant", constant_values=0)
    return np.stack([rep, d])
