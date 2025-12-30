# 🕵️‍♂️ time-series-analysis-toolkit
Repository for time-series learning, forecasting, evaluation, and visualization.  
For an introduction, see:  [A Beginner’s Guide to Time Series Analysis](https://github.com/GeneSUN/time-series-analysis-toolkit/blob/main/A%20Beginner%E2%80%99s%20Guide%20to%20Time%20Series%20Analysis.md)

## Outline

1. [Analyzing and Visualizing Time Series Data](#1-analyzing-and-visualizing-time-series-data)
2. [Preprocess](#2-preprocess)
3. [Baseline Model](#3-baseline-model)
4. [Machine Learning for Time Series Forecasting](#4-machine-learning-for-time-series-forecasting)
5. [Evaluation](#5-evaluation)  
    - [Metrics](#51-metrics)  
    - [Validation](#52-validation-strategy)
6. [Deep Learning for Time Series Forecasting](#6-deep-learning-for-time-series-forecasting)  
    - [Multivariant](#61-multivariant)  
    - [Global Model](#62-global-model)
7. [Time Series Classification](#7-time-series-classification)
8. [Time Series Clustering](#8-time-series-clustering)
9. [Fusion/Ensemble](#9-fusionensemble)
10. [Libraries](#10-libraries)
---

## 1. Analyzing and Visualizing Time Series Data
[How to Detect Stationarity in Time Series: Trend, Seasonality, and Variance](https://medium.com/@injure21/how-to-detect-stationarity-in-time-series-trend-seasonality-and-variance-66c37d71b9a4)  
- https://github.com/GeneSUN/time-series-analysis-toolkit/blob/main/src/EDA/time_series_inspectors.py  
- https://colab.research.google.com/drive/1A2LJE5tnQiz4--1tqcsVIrSPu51Q0NX8  

<img width="530" height="375" alt="image" src="https://github.com/user-attachments/assets/339deb23-d073-447b-a620-73418d01829e" />

---

## 2. Preprocess
Tools for transforming raw time series into supervised learning samples.  
<img width="600" height="351" alt="image" src="https://github.com/user-attachments/assets/c1a23c0f-068b-4355-bda9-4d44a07c9575" />

[Transform Time Series Data for Supervised Learning: From Sequence to Samples](https://medium.com/@injure21/transform-time-series-data-for-supervised-learning-from-sequence-to-samples-a7b12306b077)  
- https://github.com/GeneSUN/time-series-analysis-toolkit/blob/main/src/FeatureEngineering/TimeSeries_FeatureEngineering.py  
- https://colab.research.google.com/drive/1CrwQJIqmtFIbJRxs5cHv2g3RD86WWCbj#scrollTo=qj6PNAXcxxbu

---

## 3. Baseline Model
[Fundamentals of ARIMA](https://medium.com/@injure21/arima-for-anomaly-detection-85bfdef5d585)

---

## 4. Machine Learning for Time Series Forecasting
[Machine Learning for Time Series: Workflows That Scale](https://medium.com/@injure21/machine-learning-for-time-series-prediction-c0136bd321a9)  
- https://colab.research.google.com/drive/1Zxx4BDxjp32SoKa8mzNOyXjUSMJLhiMg

---

## 5. Evaluation

### 5.1. Metrics
[Time Series Forecasting Metrics — Practical Guide](https://medium.com/@injure21/time-series-forecasting-metrics-a-practical-guide-72bba61fc2da)
- https://colab.research.google.com/drive/1lJlJ5o_tZG2KZoqArEm4tvl2d7uizg34#scrollTo=HU6XYuOaop-L  
- https://github.com/GeneSUN/time-series-analysis-toolkit/blob/main/src/Evaluation/metrics.py

---

### 5.2. Validation Strategy
<img width="566" height="300" alt="image" src="https://github.com/user-attachments/assets/b1535736-dde4-4893-b419-a85a80fa1d3e" />

- https://github.com/GeneSUN/time-series-analysis-toolkit/blob/main/Evaluation/Validation_Strategies.ipynb  
- https://colab.research.google.com/drive/1qtw6kypnQ51Iq92gZuU1DID1AjR1UQVS

---

## 6. Deep Learning for Time Series Forecasting
[Deep Learning for Forecasting with PyTorch Lightning & PyTorch Forecasting](https://medium.com/@injure21/building-deep-learning-forecasting-models-a59ada25564f)  
- https://colab.research.google.com/drive/12Hsf-5w2tDwZcsJSPBnH8GZSOFxcB7iG

### 6.1. Multivariant
Handles correlated temporal signals across multiple features.

### 6.2. Global Model
<img width="800" height="392" alt="image" src="https://github.com/user-attachments/assets/c5b7caa1-7d5f-4c9c-afe1-0c11f69724bf" />

[From Local to Global Forecasting](https://github.com/GeneSUN/time-series-analysis-toolkit/blob/main/Model/From%20Local%20to%20Global%3A%20How%20One%20Model%20Can%20Forecast%20Thousands%20of%20Time%20Series.md)  
- https://colab.research.google.com/drive/17HKMecEdrzReMsiTd6ITEMR6izDeY3Nw

---

## 7. Time Series Classification
<img width="1465" height="596" alt="image" src="https://github.com/user-attachments/assets/16cc03fa-b10c-4607-aa78-cdbaac558eea" />

[Time-Series Classification: A Practical Guide](https://medium.com/@injure21/time-series-classification-a-practical-field-guide-with-a-telco-churn-walkthrough-271fa59b9bd0)
- https://colab.research.google.com/drive/1CGFJHqtr3R6KMDE4qNyd7sHLn0A4eg61
- https://github.com/GeneSUN/time-series-analysis-toolkit/blob/main/Model/Time-Series%20Classification.md
- [Time-Series Churn Classification](https://medium.com/@injure21/time-series-classification-churn-c33f85a038fd)
- https://github.com/GeneSUN/5g-home-churn/tree/main
  
---

## 8. Time Series Clustering
<img width="603" height="390" alt="image" src="https://github.com/user-attachments/assets/be3ebd1e-cef3-488a-8c53-4c048c0a2116" />

[Time Series Clustering — DTW to Deep Embeddings (TS2Vec, Autoencoders)](https://medium.com/@injure21/time-series-clustering-from-dtw-to-deep-embeddings-ts2vec-autoencoders-f1c1517d9025)  
- https://colab.research.google.com/drive/1v805hpfMX8Z5xYkLTMQ3N3EINDks5OAJ

---

## 9. Fusion/Ensemble
<img width="800" height="246" alt="image" src="https://github.com/user-attachments/assets/9d1e1d3b-f329-44c9-a867-ba281ea771ae" />

[Beyond Single-Source Learning: Fusion of Time Series & Static Features](https://medium.com/@injure21/beyond-single-source-learning-how-fusion-models-combine-time-series-and-static-features-f1627b7c7e55)
- https://colab.research.google.com/drive/13OhKqtGc1RDjN8DdyGhyDSN7oY0upVpa  
- https://github.com/GeneSUN/time-series-analysis-toolkit/blob/main/src/Ensemble/ensemble.py

## 10. Libraries


| Library | Best For | Key Features |
|---------|----------|--------------|
| `statsmodels.tsa` | Classical Time Series | ARIMA, SARIMA, ETS, statistical tests |
| `nixtla.statsforecast` | High-performance Classical Models | Lightning-fast ARIMA, AutoETS, optimized for scale |
| `prophet` | Business Forecasting | Interpretable, trend/seasonality modeling |
| `darts` | End-to-end Modeling | ARIMA → Deep Learning → Ensembling |
| `sktime` | Time Series Classification & Transformation | Clustering, pipelines, sklearn-style API |
| `pytorch-forecasting` | Deep Learning with PyTorch | TFT, DeepAR, seq2seq, attention-based forecasting |



---

## 📂 Repository Structure

```text
time-series-analysis-toolkit/
│
├── src/
│   ├── EDA/
│   │   └── time_series_inspectors.py
│   │
│   ├── Evaluation/
│   │   ├── feature_importance.py
│   │   ├── metrics.py
│   │   └── plot_predictions_vs_actuals.py
│   │
│   ├── FeatureEngineering/
│   │   └── TimeSeries_FeatureEngineering.py
│   │
│   ├── Imputation/
│   │   └── missing_values.py
│   │
│   └── Ensemble/
│       └── ensemble.py
│
└── notebooks / examples / datasets (user added)
