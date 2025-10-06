import warnings
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import pandas as pd
from pandas.api.types import is_list_like
from window_ops.rolling import (
    seasonal_rolling_max,
    seasonal_rolling_mean,
    seasonal_rolling_min,
    seasonal_rolling_std,
)


SEASONAL_ROLLING_MAP = {
    "mean": seasonal_rolling_mean,
    "min": seasonal_rolling_min,
    "max": seasonal_rolling_max,
    "std": seasonal_rolling_std,
}

ALLOWED_AGG_FUNCS = {"mean", "std", "min", "max", "sum", "median"}


def create_lag_features(
    df: pd.DataFrame,
    lags: List[int],
    target_col: str,
    group_col: str = None,
    dropna: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create lag features for a given column and add them to the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        lags (List[int]): List of lag steps to generate (e.g., [1, 2, 3]).
        target_col (str): Name of the column to create lag features for.
        group_col (str, optional): Column name identifying unique time series 
            (for grouped lags). If None, assumes a single time series.
        dropna (bool, optional): Whether to drop rows with NaN values created by lagging.
            Defaults to True.

    Returns:
        Tuple[pd.DataFrame, List[str]]:
            - Updated DataFrame with lag features.
            - List of names of the new lag feature columns.
    """
    # --- Input validation ---
    assert is_list_like(lags), "`lags` must be a list of integers"
    assert target_col in df.columns, f"Column `{target_col}` not found in DataFrame"

    # --- Build lagged features ---
    if group_col is None:
        warnings.warn(
            "No `group_col` specified. Assuming the DataFrame contains a single time series."
        )
        lag_dict = {
            f"{target_col}_lag_{lag}": df[target_col].shift(lag) for lag in lags
        }
    else:
        assert group_col in df.columns, f"Column `{group_col}` not found in DataFrame"
        lag_dict = {
            f"{target_col}_lag_{lag}": df.groupby(group_col)[target_col].shift(lag)
            for lag in lags
        }

    # --- Assign lag features to DataFrame ---
    df = df.assign(**lag_dict)
    new_features = list(lag_dict.keys())

    # --- Optionally drop NaNs caused by lagging ---
    if dropna:
        df = df.dropna().reset_index(drop=True)

    return df, new_features


def create_rolling_features(
    df: pd.DataFrame,
    windows: List[int],
    target_col: str,
    agg_funcs: List[str] = ["mean", "std"],
    group_col: Optional[str] = None,
    shift: int = 1,
    dropna: bool = False,
    fill_strategy: Optional[str] = "min_periods",  # NEW
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create rolling statistical features for a given column.

    Args:
        df (pd.DataFrame): Input DataFrame.
        windows (List[int]): List of window sizes for rolling calculations.
        target_col (str): Column name on which to compute rolling stats.
        agg_funcs (List[str], optional): Aggregations to compute. Defaults to ["mean", "std"].
        group_col (str, optional): Column to group by (for multiple time series). Defaults to None.
        shift (int, optional): Number of steps to shift before computing rolling statistics
                               (useful to avoid data leakage). Defaults to 1.
        dropna (bool, optional): Whether to drop rows with NaN introduced by shifting/rolling.
        fill_strategy (str, optional): How to handle NaNs at the beginning of windows.
            - "min_periods" → use partial windows (min_periods=1)


    Returns:
        Tuple[pd.DataFrame, List[str]]: DataFrame with added rolling features and list of new feature names.
    """

    # --- Validation ---
    if not isinstance(windows, (list, tuple)):
        raise ValueError("`windows` must be a list or tuple of integers.")
    if target_col not in df.columns:
        raise ValueError(f"`{target_col}` not found in DataFrame columns.")
    invalid_funcs = set(agg_funcs) - ALLOWED_AGG_FUNCS
    if invalid_funcs:
        raise ValueError(f"Invalid agg_funcs: {invalid_funcs}. Must be from {ALLOWED_AGG_FUNCS}.")

    # --- Rolling computation ---
    frames = []
    min_periods = 1 if fill_strategy == "min_periods" else None

    if group_col is None:
        warnings.warn("No `group_col` specified. Assuming a single time series in the DataFrame.")
        for w in windows:
            frame = (
                df[target_col]
                .shift(shift)
                .rolling(w, min_periods=min_periods)
                .agg({f"{target_col}_rolling_{w}_{func}": func for func in agg_funcs})
            )
            frames.append(frame)
    else:
        if group_col not in df.columns:
            raise ValueError(f"`{group_col}` not found in DataFrame columns.")
        for w in windows:
            frame = (
                df.groupby(group_col)[target_col]
                .shift(shift)
                .rolling(w, min_periods=min_periods)
                .agg({f"{target_col}_rolling_{w}_{func}": func for func in agg_funcs})
            )
            frames.append(frame)

    rolling_df = pd.concat(frames, axis=1)

    # --- Merge results ---
    df = df.assign(**rolling_df.to_dict("list"))
    new_features = rolling_df.columns.tolist()

    # --- Handle NaNs ---
    if fill_strategy == "bfill":
        df[new_features] = df[new_features].bfill(inplace=True)

    return df, new_features


def create_seasonal_rolling_features(
    df: pd.DataFrame,
    seasonal_periods: List[int],
    windows: List[int],
    target_col: str,
    agg_funcs: List[str] = ["mean", "std"],
    group_col: str = None,
    shift: int = 1,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create seasonal rolling statistical features for a given column.

    Seasonal rolling means we compute rolling statistics across
    lagged seasonal cycles (e.g., values from the same month
    in prior years, or the same weekday in prior weeks).

    Args:
        df (pd.DataFrame): Input DataFrame.
        seasonal_periods (List[int]): List of seasonal cycle lengths (e.g., 7 for weekly seasonality in daily data).
        windows (List[int]): Window sizes for rolling statistics applied on seasonal lags.
        target_col (str): Column on which to compute seasonal rolling stats.
        agg_funcs (List[str], optional): Aggregations to compute. Defaults to ["mean", "std"].
        group_col (str, optional): Column to group by (for multiple time series). Defaults to None.
        shift (int, optional): Number of seasonal shifts before computing rolling stats
                               (useful to avoid data leakage). Defaults to 1.

    Returns:
        Tuple[pd.DataFrame, List[str]]: DataFrame with added seasonal rolling features and list of new feature names.
    """
    # --- Validation ---
    if not isinstance(seasonal_periods, (list, tuple)):
        raise ValueError("`seasonal_periods` must be a list of integers.")
    if not isinstance(windows, (list, tuple)):
        raise ValueError("`windows` must be a list of integers.")


    # Map agg functions to seasonal rolling implementations
    agg_funcs_map = {agg: SEASONAL_ROLLING_MAP[agg] for agg in agg_funcs}

    added_features = []

    for sp in seasonal_periods:
        if group_col is None:
            warnings.warn(
                "No `group_col` specified. Assuming a single time series in the DataFrame."
            )
            base = df[target_col]
        else:
            if group_col not in df.columns:
                raise ValueError(f"`{group_col}` not found in DataFrame columns.")
            base = df.groupby(group_col)[target_col]

        # Build new seasonal rolling features
        col_dict = {
            f"{target_col}_{sp}_seasonal_rolling_{w}_{name}":
                base.transform(
                    lambda x, agg=agg, w=w: pd.Series(
                        agg(
                            x.shift(shift * sp).values,
                            season_length=sp,
                            window_size=w
                        ),
                        index=x.index # Ensure the new Series has the same index
                    )
                )
            for name, agg in agg_funcs_map.items() # Use agg_funcs_map here
            for w in windows
        }

        df = df.assign(**col_dict)
        added_features.extend(col_dict.keys())

    return df, added_features


def create_ewma_features(
    df: pd.DataFrame,
    target_col: str,
    alphas: Optional[List[float]] = None,
    spans: Optional[List[int]] = None,
    group_col: Optional[str] = None,
    shift: int = 1,
    dropna: bool = False
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create Exponentially Weighted Moving Average (EWMA) features for a target column.

    Args:
        df (pd.DataFrame): Input DataFrame.
        target_col (str): Column to compute EWMA features on.
        alphas (List[float], optional): List of alpha values (smoothing parameters).
        spans (List[int], optional): List of span values (periods).
                                     If spans are provided, alphas are ignored.
        group_col (str, optional): Column to group by for multiple time series.
                                   If None, assumes a single time series.
        shift (int, optional): Number of time steps to shift before computing EWMA
                               (to prevent leakage). Default is 1.
        dropna (bool, optional): Whether to drop rows with NaN values created by shifting.

    Returns:
        Tuple[pd.DataFrame, List[str]]: DataFrame with new EWMA features and list of feature names.
    """
    if spans is None and alphas is None:
        raise ValueError("You must provide either `spans` or `alphas`.")

    if spans is not None:
        params = spans
        param_type = "span"
    else:
        params = alphas
        param_type = "alpha"

    if target_col not in df.columns:
        raise ValueError(f"Column `{target_col}` not found in DataFrame.")

    # Choose grouping method
    if group_col:
        grouped = df.groupby(group_col)[target_col]
    else:
        grouped = df[target_col]

    # Generate EWMA features
    feature_dict = {}
    for param in params:
        feature_name = f"{target_col}_ewma_{param_type}_{param}"
        feature_dict[feature_name] = (
            grouped.shift(shift)
                   .ewm(span=param if param_type == "span" else None,
                        alpha=param if param_type == "alpha" else None,
                        adjust=False)
                   .mean()
        )

    df = df.assign(**feature_dict)
    if dropna:
        df = df.dropna().reset_index(drop=True)

    return df, list(feature_dict.keys())



"""

# 1. Create lag features
df, lag_feats = create_lag_features(df_cap_hour_pd, lags=[1, 2, 3], target_col="feature1", group_col="sn")

# 2. Create rolling features
df, roll_feats = create_rolling_features(
    df, windows=[12, 24], target_col="feature1", agg_funcs=["mean", "std"], group_col="sn"
)

# 3. Create seasonal rolling features (toy example with period=2)
df, seasonal_feats = create_seasonal_rolling_features(
    df, seasonal_periods=[24], windows=[1,2], target_col="feature1", agg_funcs=["mean"], group_col="sn"
)

# 4. Create EWMA features
df, ewma_feats = create_ewma_features(df, target_col="feature1", spans=[12, 24], group_col="sn")

df.dropna()

"""
