import pandas as pd
import numpy as np
from typing import List

def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a summary of missing values in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Table with columns: 'column', 'n_missing', 'pct_missing'
    """
    missing_counts = df.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0]  # keep only columns with missing

    if missing_counts.empty:
        return pd.DataFrame(columns=["column", "n_missing", "pct_missing"])

    result = pd.DataFrame({
        "column": missing_counts.index,
        "n_missing": missing_counts.values,
        "pct_missing": (missing_counts.values / len(df)) * 100
    }).sort_values(by="n_missing", ascending=False).reset_index(drop=True)

    return result


class MissingValueImputer:
    """
    Flexible imputer for handling missing values with different strategies:
    - Backfill (bfill)
    - Forward fill (ffill)
    - Zero fill
    - Defaults: mean for numeric, 'NA' for object/categorical
    """

    def __init__(
        self,
        bfill_columns: List[str] = None,
        ffill_columns: List[str] = None,
        zero_fill_columns: List[str] = None,
    ):
        self.bfill_columns = bfill_columns or []
        self.ffill_columns = ffill_columns or []
        self.zero_fill_columns = zero_fill_columns or []

    def _apply_bfill(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in self.bfill_columns if c in df.columns]
        if cols:
            df[cols] = df[cols].fillna(method="bfill")
        return df

    def _apply_ffill(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in self.ffill_columns if c in df.columns]
        if cols:
            df[cols] = df[cols].fillna(method="ffill")
        return df

    def _apply_zero_fill(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in self.zero_fill_columns if c in df.columns]
        if cols:
            df[cols] = df[cols].fillna(0)
        return df

    def _fill_remaining(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle leftover missing values with defaults."""
        missing_cols = df.columns[df.isnull().any()].tolist()

        if not missing_cols:
            return df

        numeric_cols = [c for c in missing_cols if pd.api.types.is_numeric_dtype(df[c])]
        object_cols = [c for c in missing_cols if df[c].dtype == "object"]

        if numeric_cols:
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        if object_cols:
            df[object_cols] = df[object_cols].fillna("NA")

        return df

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the full imputation pipeline."""
        df = df.copy()
        df = self._apply_bfill(df)
        df = self._apply_ffill(df)
        df = self._apply_zero_fill(df)
        df = self._fill_remaining(df)
        return df
