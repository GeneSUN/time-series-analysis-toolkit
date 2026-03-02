# Threshold-Risk Forecasting Methods for Time Series

This document summarizes three major approaches for threshold-risk forecasting in time series:


1. [Direct Exceedance Probability Forecasting (Regression-to-Classification)](#1direct-exceedance-probability-forecasting-regression-to-classification)
2. [Prediction intervals using conformal prediction](#2prediction-intervals-using-conformal-prediction)
3. [Probabilistic forecasting-Conditional Distribution Modeling](#3-probabilistic-forecasting-conditional-distribution-modeling)
   
These approaches differ in how much information about the future distribution they preserve and how well they quantify uncertainty.

## 1.Direct Exceedance Probability Forecasting (Regression-to-Classification)

Exceedance probability forecasting can be implemented by transforming a continuous forecasting problem into a binary classification problem, where the model directly predicts the probability that a future value crosses a predefined threshold.

```
Method = "Exceedance probability forecasting via event classification"

Step 1) Convert continuous target → binary event label (threshold crossing)
Step 2) Train a time-series classifier to output P(event=1 | past window)

```

### Fundamental limitations of direct exceedance probability forecasting via binary labels.

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

### ARIMA

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





