from statsmodels.datasets import sunspots
data = sunspots.load_pandas().data
from time_series_inspectors import TrendInspector, SeasonalityInspector, HeteroscedasticityInspector

import pandas as pd
from statsmodels.datasets import sunspots

# Load data
data = sunspots.load_pandas().data
data['DATE'] = pd.to_datetime(data['YEAR'], format='%Y')

# Keep only DATE and SUNACTIVITY
df_sunspots = data[['DATE', 'SUNACTIVITY']]


# Trend detection
trend_inspector = TrendInspector(df_sunspots, datetime_col='DATE', value_col='SUNACTIVITY', period=11)
trend_inspector.run_all()

# Seasonality detection
seasonality_inspector = SeasonalityInspector(df_sunspots, datetime_col='DATE', value_col='SUNACTIVITY', period=11)
seasonality_inspector.run_all()

# Heteroscedasticity detection
hetero_inspector = HeteroscedasticityInspector(df_sunspots, datetime_col='DATE', value_col='SUNACTIVITY', lags=12)
hetero_inspector.run_all()

