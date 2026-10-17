import numpy as np

def generate_random_array(shape: list, kind: str, seed: int) -> np.ndarray:
    """
    Returns a seeded 2D float64 random array with the requested shape.
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(size=shape) if kind=="uniform" else rng.normal(size=shape)
