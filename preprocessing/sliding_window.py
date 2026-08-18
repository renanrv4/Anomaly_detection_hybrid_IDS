import numpy as np

# Create sliding windows for embeddings
def create_sliding_windows(ids, n):
    """
    Parameters
    ----------
    ids : array-like
        Sequence of CAN IDs.

    n : int
        Number of IDs in each sliding window.
        Must be an odd number.

    Returns
    -------
    X : np.ndarray
        Input sequences containing n - 1 IDs.

    y : np.ndarray
        Target IDs corresponding to the center ID of each window.
    """
    if n < 3:
        raise ValueError("Value of n must be greater than 3")
    
    if n % 2 == 0:
        raise ValueError("Value of n must be odd")
    
    ids = np.asarray(ids)

    if len(ids) < n:
        raise ValueError("The ID sequence must contain all n elements")
    
    X = []
    y = []

    center = n // 2

    for i in range(len(ids) - n + 1):
        window = ids[i:i+n]

        target = window[center]
        input_ids = np.delete(window, center)
        X.append(input_ids)
        y.append(target)

    return np.asarray(X), np.asarray(y)

