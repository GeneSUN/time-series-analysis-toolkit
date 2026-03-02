## 1. Prophet

Prophet models is a decomposable additive regression model (a statistical model).

  $$y(t)=g(t)+s(t)+h(t)+ϵ$$

Where:
- g(t) = trend
- s(t) = seasonality
- h(t) = holidays
- εₜ = noise

### Strength
- Automatic Time-Series Feature Engineering, Few hyperparameters to tune
- Built-in Holiday Effects 
- Works well with default settings

## 2. ARIMA

$$y_t =a_1y_t−1 +a_2 y_t−2 +...$$
Uses: Past values directly on Lag structure

