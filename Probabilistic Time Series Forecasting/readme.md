# Threshold-Risk Forecasting Methods for Time Series

This document summarizes three major approaches for threshold-risk forecasting in time series:


1. [Direct Exceedance Probability Forecasting (Regression-to-Classification)](#1direct-exceedance-probability-forecasting-regression-to-classification)
2. [Prediction intervals using conformal prediction](#2prediction-intervals-using-conformal-prediction)
3. [Probabilistic forecasting-Conditional Distribution Modeling](#3-probabilistic-forecasting-conditional-distribution-modeling)
4. [Metrics](https://github.com/GeneSUN/time-series-analysis-toolkit/blob/main/Probabilistic%20Time%20Series%20Forecasting/readme.md#4-metrics)

These approaches differ in how much information about the future distribution they preserve and how well they quantify uncertainty.

## 1.Direct Exceedance Probability Forecasting (Regression-to-Classification)

Exceedance probability forecasting can be implemented by transforming a continuous forecasting problem into a binary classification problem, where the model directly predicts the probability that a future value crosses a predefined threshold.

```
Method = "Exceedance probability forecasting via event classification"

Step 1) Convert continuous target → binary event label (threshold crossing)
Step 2) Train a time-series classifier to output P(event=1 | past window)

```

### 1.1 Fundamental limitations of direct exceedance probability forecasting via binary labels.

Direct exceedance modeling simplifies the forecasting problem by converting a continuous target into a binary event, but this transformation **discards magnitude information.** As a result, small exceedances and extreme exceedances are treated identically during training, even though they correspond to very different levels of certainty and risk.

| True value Y | Binary label |
| ------------ | ------------ |
| 80           | 0            |
| 95           | 0            |
| 110          | 1            |
| 150          | 1            |
| 1000         | 1            |

110  → label 1 <br>
1000 → label 1

- both are treated equally strong evidence
- loss function treats them identically
- model receives no signal that 1000 is a much stronger exceedance.

**This is the key information loss**

- Binary transformation keeps only: Did it exceed threshold?
- but removes: How much did it exceed threshold? (severity of exceedance)

### 1.2 How to define Threshold

- You may use the x% of previous point/window statistics
- Domain knowledge


---

## 2.Prediction intervals using conformal prediction
Conformal prediction uses a regression model to produce point forecasts, and then uses empirical forecast errors (nonconformity scores) to construct prediction intervals with guaranteed coverage.

```
Regression model → point forecast
              +
Empirical residual quantiles → interval width
              =
Conformal prediction interval
```

### 2.1. ARIMA

```python forecasts = sf.forecast(h=HORIZON, level=[95])```

the interval comes from residual

| True y | Predicted y | Residual |
| ------ | ----------- | -------- |
| 1000   | 980         | 20       |
| 1100   | 1050        | 50       |
| 900    | 950         | -50      |

From residuals, ARIMA estimates, $$𝜎^2 =variance  of  residuals$$

- If you use variance of residual, $$s^2 = Σ(r_i - r̄)^2 / (n-1)$$
  
  - For ŷ=980  -> [930, 1030]
  - For ŷ=1050 -> [1000, 1100]
  - For ŷ=950  -> [900, 1000]

### 2.2. General Method
```
q_level    = np.ceil((1 - ALPHA) * (len(residuals) + 1)) / len(residuals)
q_hat      = np.quantile(residuals, q_level)
print(f'Conformal quantile q̂ (α={ALPHA}): {q_hat:.3f}')
```
You may end un with really shallow interval

---

## 3. Probabilistic forecasting-Conditional Distribution Modeling

**Predicting distribution parameters (μ and σ), Heteroscedastic uncertainty**


| Aspect | Point Forecasting | Probabilistic Forecasting |
|-------|------------------|-----------------------------|
| Model Output | Single value | Distribution parameters |
| Prediction | $\hat{y}$ | $\mu(X), \sigma(X)$ |
| Meaning | Best estimate of future value | Expected value and uncertainty |
| Distribution Assumption | Not modeled explicitly | $Y \mid X \sim \text{Normal}(\mu(X), \sigma(X)^2)$ |
| Model Learns | Conditional expectation | Conditional distribution |
| Mathematical Form | $E(Y \mid X)$ | $P(Y \mid X)$ |
| Information Provided | Most likely value only | Most likely value + uncertainty |
| Forecast Type | Deterministic | Probabilistic |


**Probabilistic loss function (Negative Log Likelihood (NLL))**
```python
dist = torch.distributions.Normal(mu, sigma)
loss = -dist.log_prob(y).mean()
```

$$Loss=−logP(y∣μ,σ)$$


- https://github.com/PacktPublishing/Deep-Learning-for-Time-Series-Data-Cookbook/blob/main/Chapter_7/7.4_probabilistic_forecasting_LSTM.py

**libaray**
- **TemporalFusionTransformer**
- **DeepAR**
  - https://github.com/PacktPublishing/Deep-Learning-for-Time-Series-Data-Cookbook/blob/main/Chapter_7/7.5_deepar_probforecasting.py

```
Probabilistic Time Series Forecasting Libraries
│
├── GluonTS (Amazon)
│     ├── DeepAR
│     ├── MQ-RNN
│     ├── Transformer models(TemporalFusionTransformer)
│     └── others
│
├── NeuralForecast (Nixtla)
│     ├── DeepAR
│     ├── LSTM
│     ├── TFT
│     ├── N-BEATS
│     └── others
│
└── PyTorch Forecasting
      ├── TFT
      ├── N-BEATS
      └── others
```

comparison between **DeepAR** and **TemporalFusionTransformer**

| Feature Type        | DeepAR    | TemporalFusionTransformer |
| ------------------- | --------- | ------------------------- |
| Static categorical  | ✅ Yes     | ✅ Yes                     |
| Static numerical    | ✅ Yes     | ✅ Yes                     |
| Dynamic numerical   | ✅ Yes     | ✅ Yes                     |
| Dynamic categorical | ⚠ Limited | ✅ Yes                     |
| Embeddings          | ✅ Yes     | ✅ Yes                     |
| Feature selection   | ❌ No      | ✅ Yes                     |
| Attention           | ❌ No      | ✅ Yes                     |


## 4. Metrics

The prediction contains below information

```python
    return dict(
        ds       = actuals['ds'].values,
        actual   = actuals['prb'].values,
        median   = np.percentile(sample_paths_orig, 50, axis=0),
        p10      = np.percentile(sample_paths_orig, 10, axis=0),
        p90      = np.percentile(sample_paths_orig, 90, axis=0),
        p5       = np.percentile(sample_paths_orig,  5, axis=0),
        p95      = np.percentile(sample_paths_orig, 95, axis=0),
        samples  = sample_paths_orig
    )
```

### 4.1  Point forecast metrics

```python
metrics_point = {
    'MAE' : mae(y_act, y_med),
    'RMSE': rmse(y_act, y_med),
    'MAPE': mape(y_act, y_med),
    'WAPE': wape(y_act, y_med),
    'MASE': mase(y_act, y_med),
}
```

### 4.2  Classification forecast metrics
**Brier score:**

**Brier score: good for one threshold event, but throws away magnitude once the label is created.**

$$Brier=(p−y)^2$$

**Precision/Recall or ROC-AUC**

### 4.3  Probabilistic metrics
```python
def interval_coverage(y, lo, hi) -> float:
    return np.mean((y >= lo) & (y <= hi))

def interval_sharpness(lo, hi) -> float:
    """Average width — smaller is sharper."""
    return np.mean(hi - lo)

def crps_empirical(y: np.ndarray, samples: np.ndarray) -> float:
    """
    Continuous Ranked Probability Score.
    Lower is better. Approximated via samples.
    CRPS(F, y) = E|X - y| - 0.5 * E|X - X'|
    """
    crps_vals = []
    for i, yi in enumerate(y):
        si = samples[:, i]
        term1 = np.mean(np.abs(si - yi))
        term2 = np.mean(np.abs(si[:, None] - si[None, :])) * 0.5
        crps_vals.append(term1 - term2)
    return np.mean(crps_vals)

def pinball_loss(y: np.ndarray, q_forecast: np.ndarray, q: float) -> float:
    """Quantile loss (pinball loss) at level q."""
    err = y - q_forecast
    return np.mean(np.where(err >= 0, q * err, (q - 1) * err))
```


### 4.4 Multi-step probabilistic forecast via autoregressive sampling

#### 1. The model is single-step only

```python

self.fc_mu       = nn.Linear(hidden, 1)   # output size = 1
self.fc_logsigma = nn.Linear(hidden, 1)   # output size = 1


It only ever predicts **one future timestep** — the distribution of `y_{t+1}` given the past `SEQ_LEN` steps. That's it.



## So how does it predict 96 steps ahead?

The trick is called **autoregressive sampling** — it feeds its own predictions back as inputs. Here's the loop stripped down:

Given:  [t-96, t-95, ..., t-1]  ← real historical context

Step 1: predict ŷ_t     → sample from N(μ, σ) → append to context
Step 2: predict ŷ_{t+1} → sample from N(μ, σ) → append to context
Step 3: predict ŷ_{t+2} → ...
...
Step 96: predict ŷ_{t+95}


Each step the window **slides forward by 1**, dropping the oldest real value and adding the newly sampled prediction:

Step 1 input:  [REAL,   REAL,   REAL,   ..., REAL  ]   → predict ŷ_1
Step 2 input:  [REAL,   REAL,   REAL,   ..., ŷ_1   ]   → predict ŷ_2
Step 3 input:  [REAL,   REAL,   REAL,   ..., ŷ_1, ŷ_2] → predict ŷ_3
```

#### 2. How DeepAR handles it internally

**DeepAR's fixTrains autoregressively from the start**— the model learns to handle its own noisy predictions as inputs, making it much more robust for long horizons

During training, DeepAR already feeds its own outputs back as inputs at each step:
```
Training input:   [y_1, y_2, y_3, ..., y_t]
                        ↓
                   LSTM cell → predicts distribution of y_{t+1}
                        ↓
                   feeds y_{t+1} back in → predicts y_{t+2}
                        ↓
                   ... all the way to y_{t+h}
```
This means the model learns during training that its inputs may be noisy predicted values, not just clean real values. That's the fix for the exposure bias problem I mentioned.


During prediction, ```trajectory_samples``` does the sampling

```
deepar_model = DeepAR(
    trajectory_samples = 100,   # ← this is the equivalent of our n_samples=500
    ...
)


Internally DeepAR runs 100 autoregressive sample paths and returns the percentiles as your prediction interval columns:

DeepAR-lo-90  =  5th  percentile across 100 paths
DeepAR-lo-80  = 10th  percentile across 100 paths
DeepAR        = 50th  percentile across 100 paths  (median)
DeepAR-hi-80  = 90th  percentile across 100 paths
DeepAR-hi-90  = 95th  percentile across 100 paths
```

**Side by side comparison**

<img width="750" height="301" alt="Screenshot 2026-03-04 at 3 51 46 PM" src="https://github.com/user-attachments/assets/e3b3966a-112f-4522-ba91-d60fee9d9cd3" />


















