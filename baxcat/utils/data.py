
def determine_feature_types(X):
    """
    Determines (guesses) the data types for all features (columns) in X.
    *NOTE: Currently can only guess continuous or categorical.

    Inputs:
        - X (list of list): The data. X[f] is a 1D list of data representing a column.

    Returns:
        - (list of str): The datatype for each column.
    """
    return [determine_feature_type(X_f) for X_f in X]


def determine_feature_type(X_f):
    """
    Determines (guesses) the data type of the data column X_f
    *NOTE: Currently can only guess continuous or categorical.

    Inputs:
        - X (list): The data associated with a specific column.

    Returns:
        - (string): The datatype
    """
    if isinstance(X_f, int) or isinstance(X_f, str):
        return "categorical"
    else:
        return "continuous"
