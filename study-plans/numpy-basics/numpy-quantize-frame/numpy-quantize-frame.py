import numpy as np

def quantize_and_frame(data: list, decimals: int, pad_width: int) -> np.ndarray:
    """
    Returns float64 slices of rounded, floored, and ceiling-rounded values with zero borders.
    """
    data = np.array(data, dtype=np.float64)
    r = np.round(data, decimals)
    r = np.pad(r, pad_width, mode="constant", constant_values=0)
    f = np.floor(data)
    f = np.pad(f, pad_width, mode="constant", constant_values=0)
    c = np.ceil(data)
    c = np.pad(c, pad_width, mode="constant", constant_values=0)

    return np.stack([r, f, c])
