#@title metrics.py

import numpy as np
import pandas as pd
from typing import Dict, Optional
from darts import TimeSeries
from darts.metrics import mase as darts_mase

# -----------------------
# Core Metrics
# -----------------------

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Weighted Absolute Percentage Error.
    WAPE = sum(|y - y_pred|) / sum(|y|)
    """
    numerator = np.sum(np.abs(y_true - y_pred))
    denominator = np.sum(np.abs(y_true))
    if denominator == 0:
        return np.nan
    return float(numerator / denominator)


def forecast_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Forecast Bias (FB).
    FB = 100 * (sum(y_true) - sum(y_pred)) / sum(y_true)
    """
    if np.sum(y_true) == 0:
        return np.nan
    return float(100.0 * (np.sum(y_true) - np.sum(y_pred)) / np.sum(y_true))


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: Optional[np.ndarray] = None,
) -> float:
    """
    Mean Absolute Scaled Error using darts.metrics.mase.
    Requires in-sample training series (lag=1 scaling).
    """
    if y_train is None:
        return np.nan

    n_train = len(y_train)
    n_test = len(y_true)

    # Build indices so train ends before test
    idx_train = pd.RangeIndex(start=0, stop=n_train, step=1)
    idx_test = pd.RangeIndex(start=n_train, stop=n_train + n_test, step=1)

    ts_train = TimeSeries.from_times_and_values(idx_train, np.asarray(y_train, dtype=float))
    ts_true  = TimeSeries.from_times_and_values(idx_test, np.asarray(y_true, dtype=float))
    ts_pred  = TimeSeries.from_times_and_values(idx_test, np.asarray(y_pred, dtype=float))

    return float(darts_mase(actual_series=ts_true, pred_series=ts_pred, insample=ts_train))


# -----------------------
# Metrics Wrapper
# -----------------------

def calculate_metrics(
    y: pd.Series,
    y_pred: pd.Series,
    name: str,
    y_train: Optional[pd.Series] = None,
    decimals: int = 3,
) -> Dict[str, float]:
    """
    Calculate standard forecast evaluation metrics.

    Args:
        y (pd.Series): Actual target values.
        y_pred (pd.Series): Predicted values.
        name (str): Model name or identifier.
        y_train (pd.Series, optional): In-sample training target (for MASE).
        decimals (int): Number of decimals to round results.

    Returns:
        dict: Dictionary with Algorithm name and metrics.
    """
    # Convert to numpy
    y_true = y.to_numpy(dtype=float).ravel()
    y_hat = y_pred.to_numpy(dtype=float).ravel()
    y_train_arr = None if y_train is None else y_train.to_numpy(dtype=float).ravel()

    results = {
        "Algorithm": name,
        "MAE": mae(y_true, y_hat),
        "RMSE": rmse(y_true, y_hat),
        "WAPE": wape(y_true, y_hat),
        "MASE": mase(y_true, y_hat, y_train_arr),
        "Forecast Bias(%)": forecast_bias(y_true, y_hat),
    }

    # Round all float values
    for k, v in results.items():
        if isinstance(v, float):
            results[k] = np.round(v, decimals)

    return results
    
    from sklearn.metrics import mean_squared_error
    from sktime.performance_metrics.forecasting import (
        MeanAbsoluteScaledError,
        MeanAbsolutePercentageError,
    )
    rmse_sklearn = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"RMSE: {rmse_sklearn}")
    
    mape_sktime = MeanAbsolutePercentageError(symmetric=False)
    mape = mape_sktime(y_true, y_pred)
    print(f"MAPE: {mape}")
    
    smape_sktime = MeanAbsolutePercentageError(symmetric=True)
    smape = smape_sktime(y_true, y_pred)
    print(f"SMAPE: {smape}")
    
    mase_sktime = MeanAbsoluteScaledError()
    mase = mase_sktime(y_true, y_pred, y_train=y_train)
    print(f"MASE: {mase}")
