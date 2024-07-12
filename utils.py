import datetime

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


def compute_single_period_cum_returns(period_stocks_returns: pd.DataFrame, weights: pd.Series,
                                      initial_amount: float = 1.0) -> pd.DataFrame:
    index = period_stocks_returns.index
    columns = period_stocks_returns.columns
    matrix = np.empty(period_stocks_returns.shape)
    result = pd.DataFrame(matrix, index=index, columns=columns)
    result.iloc[0, :] = weights * (1 + period_stocks_returns.iloc[0, :])
    # result.iloc[0, :] = weights  # For a different version
    for i in range(1, len(index)):
        result.iloc[i, :] = result.iloc[i - 1, :] * (1 + period_stocks_returns.iloc[i, :])
    return result * initial_amount


def compute_strategy_daily_cum_returns(stocks_returns: pd.DataFrame, df_weights: pd.DataFrame,
                                       initial_amount: float = 1.0) -> pd.DataFrame:
    periods = []
    action_days = df_weights.index
    _initial_amount = initial_amount
    for i in range(1, len(action_days)):
        # Select period
        start_period = action_days[i - 1]
        end_period = action_days[i] - datetime.timedelta(days=1)  # Exclude action day
        period_stocks_returns = stocks_returns.loc[start_period: end_period]
        # Select weights for the period
        w = df_weights.loc[start_period]
        # Compute cumulative returns for period
        period_cum_ret = compute_single_period_cum_returns(period_stocks_returns, w, _initial_amount)
        periods.append(period_cum_ret)
        # Compute new initial amount
        _initial_amount = period_cum_ret.iloc[-1, :].sum().item()
    return pd.concat(periods)