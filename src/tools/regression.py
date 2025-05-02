from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor

def fit(x, y, method, seed):
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
        return _fit_ridge_regression(x, y, seed)
    elif method == 'mlp':
        return _fit_mlp_regression(x, y, seed)
    else:
        raise ValueError(f"Unknown method: {method}. Supported methods are 'linear', 'ridge', and 'mlp'.")

def _fit_linear_regression(x, y):
    """
    Fits a linear regression model to the activations of the model.
    Args:
        x (np.ndarray): the input data to the model.
        y (np.ndarray): the target variable to fit the regression model to.
    Returns:
        model: the fitted regression model, that possess the method `predict`.
    """
    linreg = LinearRegression() 
    linreg.fit(x, y)
    return linreg
        
def _fit_ridge_regression(x, y, seed):
    """
    Fits a ridge regression model to the activations of the model.
    Args:
        x (np.ndarray): the input data to the model.
        y (np.ndarray): the target variable to fit the regression model to.
    Returns:
        model: the fitted regression model, that possess the method `predict`.
    """
    ridge = Ridge(random_state=seed)
    ridge.fit(x, y)
    return ridge

def _fit_mlp_regression(x, y, seed): # TODO MLP does not work at all, find out why
    """
    Trains an MLP to fit the activations of the model.
    Args:
        x (np.ndarray): the input data to the model.
        y (np.ndarray): the target variable to fit the regression MLP to.
    Returns:
        model: the fitted regression model, that possess the method `predict`.
        scaler: the scaler used to scale the input data.
    """
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200, early_stopping=True, verbose=True, random_state=seed) # max_iter is the number of epochs
    mlp.fit(x_scaled, y)
    return mlp, scaler