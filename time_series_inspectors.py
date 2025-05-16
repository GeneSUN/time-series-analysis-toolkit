# time_series_inspectors.py

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.stats.diagnostic import het_arch


class BaseTimeSeriesInspector:
    def __init__(self, df, datetime_col, value_col):
        self.df = df.copy()
        self.datetime_col = datetime_col
        self.value_col = value_col
        self.df[self.datetime_col] = pd.to_datetime(self.df[self.datetime_col])
        self.df = self.df.sort_values(self.datetime_col).set_index(self.datetime_col)
        self.series = self.df[self.value_col]

    def plot_series(self, title="Original Series"):
        plt.figure(figsize=(14, 4))
        plt.plot(self.series, label=self.value_col)
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel(self.value_col)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def rolling_mean_std(self, window=24):
        rolling_mean = self.series.rolling(window=window).mean()
        rolling_std = self.series.rolling(window=window).std()

        plt.figure(figsize=(14, 4))
        plt.plot(self.series, label="Original")
        plt.plot(rolling_mean, label="Rolling Mean", color="red")
        plt.plot(rolling_std, label="Rolling Std Dev", color="green")
        plt.title(f"Rolling Statistics (window={window})")
        plt.xlabel("Time")
        plt.ylabel(self.value_col)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


class TrendInspector(BaseTimeSeriesInspector):
    def __init__(self, df, datetime_col, value_col, period=24):
        super().__init__(df, datetime_col, value_col)
        self.period = period

    def decompose_stl(self):
        stl = STL(self.series, period=self.period)
        result = stl.fit()
        result.plot()
        plt.suptitle("STL Decomposition")
        plt.tight_layout()
        plt.show()

    def plot_acf(self, lags=100):
        plot_acf(self.series.dropna(), lags=lags)
        plt.title("Autocorrelation (ACF)")
        plt.tight_layout()
        plt.show()

    def adf_test(self):
        result = adfuller(self.series.dropna())
        stat, pval, used_lag, nobs, crit_vals, _ = result

        print("=== Augmented Dickey-Fuller (ADF) Test ===")
        print(f"ADF Statistic     : {stat:.4f}")
        print(f"p-value           : {pval:.4f}")
        print(f"Used Lag          : {used_lag}")
        print(f"Number of Obs     : {nobs}")
        print("Critical Values   :")
        for key, value in crit_vals.items():
            print(f"   {key}: {value:.4f}")

        if pval < 0.05:
            print("\n\u2705 Interpretation: The null hypothesis is rejected.")
            print("\u2192 The time series is likely stationary (no significant trend).")
        else:
            print("\n\u274C Interpretation: The null hypothesis cannot be rejected.")
            print("\u2192 The time series is likely non-stationary (trend is present).")

    def run_all(self):
        self.plot_series("Trend Detection: Original Series")
        self.rolling_mean_std()
        self.decompose_stl()
        self.plot_acf()
        self.adf_test()


class SeasonalityInspector(BaseTimeSeriesInspector):
    def __init__(self, df, datetime_col, value_col, period=24):
        super().__init__(df, datetime_col, value_col)
        self.period = period

    def plot_seasonal_box(self, by="hour"):
        df_temp = self.df.copy()
        df_temp["hour"] = df_temp.index.hour
        df_temp["weekday"] = df_temp.index.weekday

        if by not in df_temp.columns:
            raise ValueError(f"'{by}' must be one of: 'hour', 'weekday'")

        plt.figure(figsize=(12, 5))
        sns.boxplot(x=by, y=self.value_col, data=df_temp)
        plt.title(f"Seasonal Box Plot by {by.capitalize()}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def decompose_stl(self):
        stl = STL(self.series, period=self.period)
        result = stl.fit()
        result.plot()
        plt.suptitle("STL Decomposition (Seasonal Component)")
        plt.tight_layout()
        plt.show()

    def plot_acf(self, lags=100):
        plot_acf(self.series.dropna(), lags=lags)
        plt.title("Autocorrelation Function (ACF)")
        plt.tight_layout()
        plt.show()

    def kpss_test(self, regression='c', nlags='auto'):
        print("=== KPSS Test ===")
        print(f"Settings: regression='{regression}', nlags='{nlags}'")

        try:
            stat, pval, lags, crit = kpss(self.series.dropna(), regression=regression, nlags=nlags)

            print(f"\nKPSS Statistic   : {stat:.4f}")
            print(f"p-value          : {pval:.4f}")
            print(f"Lags Used        : {lags}")
            print("Critical Values  :")
            for key, val in crit.items():
                print(f"   {key}: {val}")

            if pval < 0.05:
                print("\n\u274C Interpretation: The null hypothesis (stationarity) is rejected.")
                print("\u2192 The series is likely **non-stationary**, possibly due to trend or seasonality.")
            else:
                print("\n\u2705 Interpretation: The null hypothesis is not rejected.")
                print("\u2192 The series is likely **stationary** under the tested conditions.")

        except Exception as e:
            print("Error during KPSS test:", e)

    def run_all(self):
        self.plot_series("Seasonality Detection: Original Series")
        self.plot_seasonal_box(by="hour")
        self.decompose_stl()
        self.plot_acf()
        self.kpss_test()


class HeteroscedasticityInspector(BaseTimeSeriesInspector):
    def __init__(self, df, datetime_col, value_col, lags=12):
        super().__init__(df, datetime_col, value_col)
        self.lags = lags

    def plot_residuals(self):
        model = AutoReg(self.series.dropna(), lags=1).fit()
        residuals = model.resid

        plt.figure(figsize=(14, 4))
        plt.plot(residuals, label="Residuals")
        plt.title("Model Residuals")
        plt.xlabel("Time")
        plt.ylabel("Residual")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def arch_test(self):
        print("=== ARCH Test for Heteroscedasticity ===")
        residuals = AutoReg(self.series.dropna(), lags=1).fit().resid
        test_stat, pval, _, _ = het_arch(residuals, nlags=self.lags)

        print(f"ARCH Test Statistic : {test_stat:.4f}")
        print(f"p-value             : {pval:.4f}")

        if pval < 0.05:
            print("\n\u274C Interpretation: Reject null hypothesis of homoscedasticity.")
            print("\u2192 The variance is likely **not constant** over time (heteroscedastic).")
        else:
            print("\n\u2705 Interpretation: Fail to reject null hypothesis.")
            print("\u2192 The variance is likely **constant** (homoscedastic).")

    def plot_transforms(self):
        fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
        axes[0].plot(np.log1p(self.series), label="Log Transform", color='blue')
        axes[1].plot(np.sqrt(self.series), label="Sqrt Transform", color='green')

        axes[0].set_title("Log-Transformed Series")
        axes[1].set_title("Square-Root-Transformed Series")

        for ax in axes:
            ax.grid(True)
            ax.legend()
            ax.set_ylabel(self.value_col)

        plt.xlabel("Time")
        plt.tight_layout()
        plt.show()

    def run_all(self):
        self.plot_series("Heteroscedasticity Detection: Original Series")
        self.rolling_mean_std()
        self.plot_residuals()
        self.arch_test()
        self.plot_transforms()
