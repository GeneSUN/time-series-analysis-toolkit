import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller, kpss, acf
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.stats.diagnostic import het_arch

class BaseTimeSeriesInspector:
    """
    Base class for time series analysis.
    Provides shared methods such as plotting and rolling statistics.
    """
    def __init__(self, df, datetime_col, value_col):
        """
        Initialize with a DataFrame and target column.
        """
        self.df = df.copy()
        self.datetime_col = datetime_col
        self.value_col = value_col
        self.df[self.datetime_col] = pd.to_datetime(self.df[self.datetime_col])
        self.df = self.df.sort_values(self.datetime_col).set_index(self.datetime_col)
        self.series = self.df[self.value_col]

    def plot_series(self, title="Original Series"):
        """
        Plot the raw time series for visual inspection.
        """
        print("\n--- 🔍 Step: Visual Inspection ---")
        print("Purpose: Observe overall shape, amplitude, and possible patterns.\n")
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
        """
        Plot rolling mean and standard deviation to assess local trends and variance shifts.
        """
        print(f"\n--- 📉 Step: Rolling Mean and Standard Deviation (window={window}) ---")
        print("Purpose: Detect shifting average or variability over time.\n")
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
    """
    Detects trend in time series using decomposition, ACF, and statistical tests.
    """
    def __init__(self, df, datetime_col, value_col):
        super().__init__(df, datetime_col, value_col)


    def plot_acf(self, lags=100):
        """
        Plot the autocorrelation function to identify persistence in the signal.
        """
        print(f"\n--- 🔁 Step: Autocorrelation Function (ACF, lags={lags}) ---")
        print("Purpose: Slow decay in ACF suggests trend.\n")
        plot_acf(self.series.dropna(), lags=lags)
        plt.title("Autocorrelation (ACF)")
        plt.tight_layout()
        plt.show()

    def adf_test(self):
        """
        Perform Augmented Dickey-Fuller test for trend non-stationarity.
        """
        print("\n--- 🧪 Step: Augmented Dickey-Fuller (ADF) Test ---")
        print("Purpose: Test for stationarity. Null hypothesis: series is non-stationary (has a trend).\n")
        result = adfuller(self.series.dropna())
        stat, pval, used_lag, nobs, crit_vals, _ = result

        print(f"ADF Statistic     : {stat:.4f}")
        print(f"p-value           : {pval:.4f}")
        print(f"Used Lag          : {used_lag}")
        print(f"Number of Obs     : {nobs}")
        print("Critical Values   :")
        for key, value in crit_vals.items():
            print(f"   {key}: {value:.4f}")

        if pval < 0.05:
            print("\n✅ Interpretation: The null hypothesis is rejected.")
            print("→ The series is likely stationary (no significant trend).")
        else:
            print("\n❌ Interpretation: The null hypothesis cannot be rejected.")
            print("→ The series is likely non-stationary (trend is present).")

    def run_all(self):
        """
        Run all trend detection methods in order.
        """
        print("\n" + "="*80)
        print("📊 TREND INSPECTION".center(80))
        print("="*80)
        self.plot_series("Trend Detection: Original Series")
        self.rolling_mean_std()
        self.plot_acf()
        self.adf_test()

class SeasonalityInspector(BaseTimeSeriesInspector):
    """
    Detects seasonality through decomposition, box plots, ACF, and KPSS test.
    """
    def __init__(self, df, datetime_col, value_col, group_by, seasonal_period=None):
        """
        :param df: Input DataFrame
        :param datetime_col: Name of datetime column
        :param value_col: Name of the value column
        :param seasonal_period: Optional. If None, it will be auto-detected.
        :param group_by: datetime attribute to group boxplot by (e.g., 'hour', 'weekday', 'month')
        """
        super().__init__(df, datetime_col, value_col)
        self.group_by = group_by
        self.period = seasonal_period or self._infer_seasonal_period()
        
        print(f"[INFO] Detected seasonal period: {self.period}")

    def _infer_seasonal_period(self, max_lag=168):
        """
        Automatically detect seasonal period using ACF.
        Looks for the first significant peak after lag 0.
        """
        print("[INFO] Attempting to infer seasonal period using ACF...")
        series_clean = self.series.dropna()
        acf_vals = acf(series_clean, nlags=max_lag)

        # ignore lag 0; find first peak above a threshold (e.g., 0.5)
        threshold = 0.5
        for lag in range(1, len(acf_vals)):
            if acf_vals[lag] >= threshold:
                print(f"[INFO] ACF peak detected at lag: {lag}")
                return lag

        print("[WARN] No strong seasonality detected; defaulting to 24.")
        return 24


    def plot_seasonal_box(self):
        """
        Create a box plot grouped by a datetime attribute defined during initialization.

        Raises:
            ValueError: If the datetime index is not valid or group_by is not a valid attribute.
        """
        by = self.group_by
        print(f"\n--- 📦 Step: Seasonal Box Plot by {by.capitalize()} ---")
        print(f"Purpose: Reveal repeated patterns in data by grouping by {by}.\n")

        df_temp = self.df.copy()

        if not pd.api.types.is_datetime64_any_dtype(df_temp.index):
            raise ValueError("Datetime index is required for seasonal box plots.")

        try:
            df_temp[by] = getattr(df_temp.index, by)
        except AttributeError:
            raise ValueError(f"'{by}' is not a valid datetime attribute. Try 'hour', 'weekday', 'month', etc.")

        plt.figure(figsize=(12, 5))
        sns.boxplot(x=by, y=self.value_col, data=df_temp)
        plt.title(f"Seasonal Box Plot by {by.capitalize()}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    def decompose_stl(self):
        """
        Decompose the series using STL to isolate the seasonal component.
        """
        print("\n--- 🧩 Step: STL Decomposition ---")
        print("Purpose: Extract seasonal component from the time series.\n")
        stl = STL(self.series, period=self.period)
        result = stl.fit()
        result.plot()
        plt.suptitle("STL Decomposition (Seasonal Component)")
        plt.tight_layout()
        plt.show()

    def plot_acf(self, lags=100):
        """
        Plot ACF to identify periodic autocorrelation.
        """
        print(f"\n--- 🔁 Step: Autocorrelation Function (ACF, lags={lags}) ---")
        print("Purpose: Repeating spikes suggest presence of seasonality.\n")
        plot_acf(self.series.dropna(), lags=lags)
        plt.title("Autocorrelation Function (ACF)")
        plt.tight_layout()
        plt.show()

    def kpss_test(self, regression='c', nlags='auto'):
        """
        Perform KPSS test for stationarity.
        """
        print("\n--- 🧪 Step: KPSS Test ---")
        print("Purpose: Test for stationarity. Null hypothesis: series is stationary.\n")
        try:
            stat, pval, lags, crit = kpss(self.series.dropna(), regression=regression, nlags=nlags)

            print(f"KPSS Statistic   : {stat:.4f}")
            print(f"p-value          : {pval:.4f}")
            print(f"Lags Used        : {lags}")
            print("Critical Values  :")
            for key, val in crit.items():
                print(f"   {key}: {val}")

            if pval < 0.05:
                print("\n❌ Interpretation: The null hypothesis (stationarity) is rejected.")
                print("→ The series is likely non-stationary (seasonality may exist).")
            else:
                print("\n✅ Interpretation: The null hypothesis is not rejected.")
                print("→ The series is likely stationary.")

        except Exception as e:
            print("Error during KPSS test:", e)

    def run_all(self):
        """
        Run all seasonality detection methods in order.
        """
        print("\n" + "="*80)
        print("🔁 SEASONALITY INSPECTION".center(80))
        print("="*80)
        self.plot_series("Seasonality Detection: Original Series")
        self.plot_seasonal_box()
        self.decompose_stl()
        self.plot_acf()
        self.kpss_test()


class HeteroscedasticityInspector(BaseTimeSeriesInspector):
    """
    Detects heteroscedasticity (changing variance) in time series.
    """
    def __init__(self, df, datetime_col, value_col, lags=12):
        super().__init__(df, datetime_col, value_col)
        self.lags = lags

    def plot_residuals(self):
        """
        Fit a simple AR model and plot residuals to inspect variance over time.
        """
        print("\n--- 🔍 Step: Residual Plot ---")
        print("Purpose: Check if residual variance changes over time.\n")
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
        """
        Perform ARCH test to statistically detect heteroscedasticity.
        """
        print("\n--- 🧪 Step: ARCH Test ---")
        print(f"Purpose: Detect time-varying volatility using {self.lags} lags.\n")
        residuals = AutoReg(self.series.dropna(), lags=1).fit().resid
        test_stat, pval, _, _ = het_arch(residuals, nlags=self.lags)

        print(f"ARCH Test Statistic : {test_stat:.4f}")
        print(f"p-value             : {pval:.4f}")

        if pval < 0.05:
            print("\n❌ Interpretation: Reject null hypothesis of homoscedasticity.")
            print("→ The variance is likely **not constant** (heteroscedastic).")
        else:
            print("\n✅ Interpretation: Fail to reject null hypothesis.")
            print("→ The variance is likely **constant** (homoscedastic).")

    def plot_transforms(self):
        """
        Plot log and square-root transforms to visually assess if variance stabilizes.
        """
        print("\n--- 🔁 Step: Variance-Stabilizing Transforms ---")
        print("Purpose: Apply log/sqrt transforms to reduce changing variance effects.\n")
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
        """
        Run all heteroscedasticity detection methods in order.
        """
        print("\n" + "="*80)
        print("📈 HETEROSCEDASTICITY INSPECTION".center(80))
        print("="*80)
        self.plot_series("Heteroscedasticity Detection: Original Series")
        self.rolling_mean_std()
        self.plot_residuals()
        self.arch_test()
        self.plot_transforms()
