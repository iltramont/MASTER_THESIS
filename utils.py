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


def compute_single_period_returns(period_stocks_returns: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    index = period_stocks_returns.index
    columns = period_stocks_returns.columns
    matrix = np.empty(period_stocks_returns.shape)
    result = pd.DataFrame(matrix, index=index, columns=columns)
    # result.iloc[0, :] = weights * period_stocks_returns.iloc[0, :]
    result.iloc[0, :] = weights
    for i in range(1, len(index)):
        result.iloc[i, :] = result.iloc[i - 1, :] * (1 + period_stocks_returns.iloc[i, :])
    return result
