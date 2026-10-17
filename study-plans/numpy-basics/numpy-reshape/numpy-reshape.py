import numpy as np

def reshape_array(data: list, operation: str) -> np.ndarray:
    """
    Returns a float64 array with the shape selected by operation.
    """
    if operation=="flatten":
        return np.array(data, dtype=np.float64).flatten()
    elif operation=="transpose":
        # after checking the solution and reading up on this, it's good practice to type copy() because when you take the transpose, it doesn't really change anything in the memory, it only changes the way it reads the memory
        # so when you make changes to this new variable, it will inadvertently make changes to the original variable as well, since they're both referring to the same memory slots
        return np.array(data, dtype=np.float64).T.copy()
    else:
        a= np.array(data, dtype=np.float64)
        return np.expand_dims(a, axis=0)
