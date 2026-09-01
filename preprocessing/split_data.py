def train_validation_split(X, y, validation_ratio=0.2):
    """
    Split the dataset into training and validation sets
    while preserving temporal order.

    Parameters
    ----------
    X : np.ndarray
        Input sequences.

    y : np.ndarray
        Target IDs.

    validation_ratio : float
        Fraction of samples used for validation.

    Returns
    -------
    X_train, X_val, y_train, y_val
    """

    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1")

    split_index = int(len(X) * (1 - validation_ratio))

    X_train = X[:split_index]
    X_val = X[split_index:]

    y_train = y[:split_index]
    y_val = y[split_index:]

    return X_train, X_val, y_train, y_val
