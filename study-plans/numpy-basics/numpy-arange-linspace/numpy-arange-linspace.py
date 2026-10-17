import numpy as np

def create_sequence(start: float, stop: float, param: float, kind: str) -> np.ndarray:
    """
    Returns a 1D float64 array containing the requested sequence.
    """
    # After reviewing the solution, I realized that putting int(param) in linspace could be important because we can't produce 2.5 numbers between the start and end, so this should stop any potential issues
    return np.linspace(start, stop, int(param), dtype=np.float64) if kind=="linspace" else np.arange(start, stop, param, dtype=np.float64)
