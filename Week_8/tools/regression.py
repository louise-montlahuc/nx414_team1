from sklearn.linear_model import LinearRegression, Ridge

def fit(x, y, method):
    """Fits a regression model to the activations of the model.

    Args:
        x (np.ndarray): the input data to the model.
        y (np.ndarray): the target variable to fit the regression model to.
        method (str): which method to use for the regression model.
            'linear' for linear regression, 'ridge' for ridge regression, and 'mlp' for MLP.

    Raises:
        ValueError: if the method is not supported.

    Returns:
        model: the fitted regression model, that possess the method `predict`.
    """
    if method == 'linear':
        return _fit_linear_regression(x, y)
    elif method == 'ridge':
        return _fit_ridge_regression(x, y)
    elif method == 'mlp':
        return _fit_mlp_regression(x, y)
    else:
        raise ValueError(f"Unknown method: {method}. Supported methods are 'linear', 'ridge', and 'mlp'.")

def _fit_linear_regression(x, y):
    """
    Fits a regression model to the activations of the model.
    """
    linreg = LinearRegression() 
    linreg.fit(x, y)
    return linreg
        
def _fit_ridge_regression(x, y):
    """
    Fits a regression model to the activations of the model.
    """
    ridge = Ridge()
    ridge.fit(x, y)
    return ridge

def _fit_mlp_regression(x, y):
    """
    Trains an MLP to fit the activations of the model.
    The hook can be 'all' for all activations or 'pca' for 1000 principal components.
    """
    raise NotImplementedError("MLP regression is not implemented yet.")