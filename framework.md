# Common end-to-end procedure for a time series forecasting project (refined + expanded)

## 0) Frame the forecasting task (BEFORE touching models)
   A. Forecast object / granularity
      - What is one “observation”? (hourly/daily/weekly; per user/store/tower/etc.)
      - What is the forecast horizon H? (e.g., next 7 days, next 24 steps)
      - What is the decision it supports? (reorder, staffing, alerting, SLA, pricing)
   B. Forecast type
      - Univariate vs multivariate (exogenous X)
      - Single-step vs multi-step (direct vs recursive vs multi-output)
      - Single-series vs many-series (local vs global model)
      - Point forecast vs probabilistic forecast (P50/P90, prediction intervals)
   C. Success criteria
      - Pick metrics tied to the decision + cost of errors
      - Define the evaluation protocol: train/val/test + backtesting windows
   (These “basic steps of a forecasting task” align with the classic workflow: define problem → gather info → explore → model → evaluate/use.) :contentReference[oaicite:0]{index=0}

## 1) Data understanding + “forecastability” checks (EDA for time series)

   A. Visual EDA (single series + aggregated across many series)
      - Level, **trend**, **seasonality**, changepoints/structural breaks
      - Outliers (one-off spikes vs regime changes)
      - **Heteroscedasticity** (variance changing over time)
   B. Statistical diagnostics (useful but not a silver bullet)
      - Stationarity checks and autocorrelation structure (ACF/PACF intuition)
      - If using ARIMA-family: whether differencing/seasonal differencing may help
   C. lagged correlation (Cross-Correlation)
      - Compute corr( y(t), x(t−k) ) over lags k = 0…K (and negative lags too). 
        - ```statsmodels.tsa.stattools.ccf```
      - ```statsmodels.tsa.stattools.grangercausalitytests```
      - The most “real” quick test: time-aware ablation with backtesting (feature adds value or not)

   
   
## 2) Split strategy + leakage prevention (design evaluation early)
   A. Holdout that respects time
      - Never shuffle randomly; train must be strictly earlier than validation/test
      - Decide if you need a “gap” between train and test (to prevent leakage via lag features)
   B. Backtesting (time series cross-validation)
      - Expanding window: train grows each fold
      - Rolling window: fixed-size train slides forward
      - Choose folds that match how the model will be used in production
   References:
      - scikit-learn TimeSeriesSplit (expanding-window style) :contentReference[oaicite:1]{index=1}
      - StatsForecast cross-validation (sliding window backtesting concept + implementation) :contentReference[oaicite:2]{index=2}

## 3) Preprocessing (time-aware)
   A. Missingness handling (time-series specific)
      - “Missing timestamp” vs “missing value at existing timestamp” are different problems
      - Methods: forward/back fill, interpolation, seasonal imputation, model-based imputation
      - Avoid using future information (e.g., centered rolling stats can leak)
   B. Transformations (when appropriate)
      - Stabilize variance (log/Box-Cox), remove seasonality, differencing for certain models
      - Scaling/normalization: fit on train only, apply to val/test
   C. Outlier strategy
      - Keep if it represents reality you must forecast (e.g., holiday spikes)
      - Cap/remove only if it’s data error and you’re confident it won’t recur


## 4) Feature engineering (mainly for ML/global models)
   A. Time index features (known in the future)
      - day-of-week, month, holiday flags, cyclical encoding
   B. Lag/rolling features (careful about leakage)
      - lags: y(t-1), y(t-7), etc.
      - rolling: rolling mean/std, EWM, seasonal rolling
   C. Exogenous variables (X)
      - Only include if they’re available at prediction time (or you have a plan to forecast X too)
      - For cold-start/new entity: build features from static metadata + hierarchy/region embeddings + similarity

## 5) Modeling (choose based on task constraints, not fashion)
   A. Classical statistical (strong baselines; interpretable)
      - ETS, ARIMA/SARIMA, TBATS, etc.
      - Great when per-series patterns are stable and data is limited
   B. “Tabular ML” via supervised framing
      - Convert to (features at time t) → (target at t+h)
      - Often strong in practice (handles many covariates; scalable)
   C. Deep learning
      - Useful for large multi-series/global settings and complex nonlinearities
      - Needs careful regularization + backtesting; can struggle with regime shifts
   D. Ensembles / combinations
      - Often improves robustness because “no single model wins everywhere”
      - Strong evidence from forecasting competitions that combinations and hybrid approaches perform very well on average. :contentReference[oaicite:5]{index=5}

## 6) Metrics (pick what matches business pain + horizon)
   A. Point forecast metrics
      - MAE / RMSE (scale-dependent)
      - MAPE / sMAPE (scale-free-ish; beware zeros and small denominators)
      - WAPE/WMAPE (often better behaved in ops settings)
   B. Probabilistic metrics (if you need uncertainty)
      - Pinball loss for quantiles (P50/P90)
      - CRPS / interval coverage + interval width tradeoff
   C. Horizon-aware reporting
      - Report metric by horizon (h=1…H), not just averaged

## 7) Model selection + validation report (make it decision-ready)
   - Compare models on the SAME backtesting folds and SAME metrics
   - Error slicing: by season, by region/entity type, by demand level, during promos/holidays
   - Stability: does the model degrade sharply on recent windows (concept drift)?

## 8) Multi-series / global modeling (your special case)
   A. Decide: local vs global vs grouped-hierarchical
      - Local: one model per series (simple but expensive at scale)
      - Global: one model for all series (better data sharing; handles cold-start better if features exist)
      - Grouped/hierarchical: one model per segment (middle ground)
   B. Practical requirements
      - Consistent IDs, aligned timestamps, robust missing handling across series
      - Per-series scaling (often needed), plus global features
   C. Evaluate fairly
      - Aggregate metrics across series (weighted + unweighted)
      - Also inspect “long tail” series (sparse/noisy) separately

## 9) Productionization (often where projects succeed/fail)
   A. Training/serving parity
      - Same feature definitions, same time cutoffs, same transformations
   B. Inference design
      - Batch vs streaming; latency constraints; forecast update frequency
   C. Monitoring
      - Data quality (missing timestamps, feature drift)
      - Forecast quality (backtest-like live evaluation with lag)
      - Alerting on drift/regime change
   D. Retraining policy
      - Schedule-based + trigger-based (performance drop, drift detection)
