# 🕵️‍♂️ Time Series Inspectors
[How to Detect Stationarity in Time Series: Trend, Seasonality, and Variance](https://medium.com/@injure21/how-to-detect-stationarity-in-time-series-trend-seasonality-and-variance-66c37d71b9a4)

[A Beginner’s Guide to Time Series Analysis](https://medium.com/@injure21/a-beginners-guide-to-time-series-analysis-9f68a8078233)

[Fundamental of ARIMA](https://medium.com/@injure21/arima-for-anomaly-detection-85bfdef5d585)

[Transform Time Series Data for Supervised Learning: From Sequence to Samples](https://medium.com/@injure21/transform-time-series-data-for-supervised-learning-from-sequence-to-samples-a7b12306b077)

This Python module provides an extendable framework for inspecting key properties of time series data, including:

**Stationarity**  
- 📈 Trend  
- 🔁 Seasonality  
- 📉 Heteroscedasticity  

It leverages libraries like `pandas`, `matplotlib`, `seaborn`, `statsmodels`, and `numpy` to provide visualizations and statistical tests to support time series diagnostics.

---
## 📂 Module Overview

The module contains the following classes:

### `BaseTimeSeriesInspector`

A foundational class that provides common methods for:

- Initializing with a DataFrame  
- Time parsing  
- Rolling statistics  
- General plotting utilities  

---

### `TrendSeasonalityInspector`

Inherits from `BaseTimeSeriesInspector` and provides:

- STL decomposition  
- Seasonality and trend plotting  
- Seasonal autocorrelation visualization  

---

### `StationarityInspector`

Inherits from `BaseTimeSeriesInspector` and provides:

- Augmented Dickey-Fuller (ADF) test  
- KPSS test  
- Stationarity summary report  

---

### `HeteroscedasticityInspector`

Inherits from `BaseTimeSeriesInspector` and provides:

- ARCH test  
- Visualization of residual variance  
- Histogram and Q-Q plots for residuals  



## 📦 Installation

Make sure the following packages are installed:

```bash
pip install pandas numpy matplotlib seaborn statsmodels
