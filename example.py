from statsmodels.datasets import sunspots
data = sunspots.load_pandas().data
from time_series_inspectors import TrendInspector, SeasonalityInspector, HeteroscedasticityInspector

import pandas as pd
from statsmodels.datasets import sunspots

import numpy as np

# Simulate a value with hourly + weekday seasonal pattern + noise
date_range = pd.date_range(start="2023-01-01", periods=24 * 7, freq="H")

np.random.seed(42)
hourly_pattern = np.sin(2 * np.pi * date_range.hour / 24)      # daily cycle
weekday_pattern = np.cos(2 * np.pi * date_range.dayofweek / 7) # weekly cycle
noise = np.random.normal(0, 0.3, len(date_range))
value = 10 + 2 * hourly_pattern + 1 * weekday_pattern + noise

df_hourly = pd.DataFrame({
    "timestamp": date_range,
    "value": value
})


inspector = SeasonalityInspector(
    df=df_hourly,
    datetime_col="timestamp",
    value_col="value",
    seasonal_period=24,       # Daily seasonality
    group_by="hour"           # Group by hour for boxplot
)
inspector.run_all()


from statsmodels.datasets import sunspots

# Load data
data = sunspots.load_pandas().data
data['DATE'] = pd.to_datetime(data['YEAR'], format='%Y')

df_sunspots = data[['DATE', 'SUNACTIVITY']]

trend_inspector = TrendInspector(df_sunspots, datetime_col='DATE', value_col='SUNACTIVITY')
trend_inspector.run_all()




# Download daily stock prices
import yfinance as yf

data = yf.download("AAPL", start="2020-01-01", end="2024-01-01")

data['log_return'] = np.log(data['Close']).diff()
df_returns = data[['log_return']].dropna().reset_index()
df_returns.columns = ['DATE', 'log_return']

inspector = HeteroscedasticityInspector(df_returns, datetime_col='DATE', value_col='log_return')
inspector.run_all()