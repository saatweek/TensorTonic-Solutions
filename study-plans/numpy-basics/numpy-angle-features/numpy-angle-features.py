import numpy as np

def angle_features(angles: list) -> np.ndarray:
    """
    Returns a (3, n) float64 array with sine, cosine, and tangent rows.
    """
    angles = np.array(angles, dtype=np.float64)
    s = np.sin(angles)
    c = np.cos(angles)
    t = np.tan(angles)
    return np.stack([s, c, t])
