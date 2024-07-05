import pandas as pd
import numpy as np
import cvxpy as cv

def compute_min_variance_weights(covariance_matrix: pd.DataFrame) -> pd.Series:
    n = covariance_matrix.shape[0]
    weights = cv.Variable(n, nonneg=True)
    portfolio_volatility = cv.quad_form(weights, covariance_matrix)
    problem = cv.Problem(
        cv.Minimize(portfolio_volatility),
        [
            weights.sum() == 1.0,
            weights >= 0
        ]
    )
    problem.solve()
    return pd.Series(weights.value, index=covariance_matrix.index)


def compute_returns(df: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    return (df - df.shift(1)) / df.shift(1)