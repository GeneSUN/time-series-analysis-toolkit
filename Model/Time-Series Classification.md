# Time-Series Classification — A Practical Field Guide  
### *(with a Telco Churn Walkthrough)*

---

## 📌 Introduction: From Static Snapshots to Evolving Sequences

Most real-world data doesn’t stand still — it **evolves over time**.  
> Customer behavior, sensor readings, network performance, and stock prices are all examples of **time series data**.

**Time-Series Classification (TSC)** is the task of assigning a **single label to an entire sequence**, rather than to a static snapshot.

- Each **entity (e.g., customer)** has multiple features  
- Each feature is observed **across time**
- All sequences together map to **one label**

<img width="1465" height="596" alt="image" src="https://github.com/user-attachments/assets/4158aac0-4324-482f-883a-812698f82aa1" />

In contrast, classical classification treats every sample as a fixed-length vector of independent features.

- Each record represents a static moment — for instance, predicting churn from a customer’s current demographics and plan attributes.
- Time-series classification, however, learns from how those attributes change over time — for example, a gradual decline in usage or increasing latency leading to churn.

<img width="701" height="516" alt="image" src="https://github.com/user-attachments/assets/ff7a1b91-f52c-4d2c-8779-e3a2e7c62461" />


By capturing *temporal dynamics* — trends, drops, cycles, and regime shifts — TSC enables models to detect **behavioral evolution**, not just static correlations.  
> **Classical classification sees the state.  
> Time-series classification sees the story.**

This is critical in domains such as healthcare monitoring, predictive maintenance, fraud detection, and **telecom churn forecasting**.

---

## 📑 Table of Contents

- [Toy Telco Example](#toy-telco-example)
- [Modeling Approaches](#modeling-approaches)
  - [1️⃣ Feature-Based Classifiers](#1️⃣-feature-based-classifiers-fast--interpretable)
  - [2️⃣ Distance-Based Nearest Neighbors](#2️⃣-distance-based-nearest-neighbors-surprisingly-strong-baselines)
  - [3️⃣ Deep Sequence Models](#3️⃣-deep-sequence-models-learn-the-shape-for-you)
    - [LSTM / GRU](#a-lstm--gru)
    - [1D CNNs — InceptionTime](#b-1d-cnns-inceptiontime)
    - [ResNetClassifier](#c-resnetclassifier)
- [Fusion: Sequential + Static Features](#fusion-combining-sequential-and-static-features)
- [Conclusion](#conclusion)
- [Popular Time Series Packages](#popular-time-series-packages)

---

## 🧪 Toy Telco Example

A [Google Colab notebook](https://colab.research.google.com/drive/1CGFJHqtr3R6KMDE4qNyd7sHLn0A4eg61) accompanies this article and contains all runnable code:


This notebook synthesize customer‑day panels (90 days, ~10 behavioral features such as Consumption_amount, Data_volume, etc.). Churners show a gradual decline before the label is assigned. We evaluate multiple families:

- Feature‑based ML: Random Forest on R/F/M‑style aggregates.
- Distance‑based: k‑NN with Dynamic Time Warping (DTW).
- Deep sequence models: LSTM and InceptionTime (a strong 1D‑CNN).
- Fusion: combine time‑series with static features (age, plan, tenure).

---

## 🔧 Modeling Approaches

> Before diving into time-series classification, it’s worth recalling how classical machine learning typically approaches the problem.

### 1️⃣ Feature-Based Classifiers (Fast & Interpretable)


Instead of feeding raw sequences into a model, we **summarize each time series into a fixed-length feature vector**.

Common transformations:
- **RFM-style features** (recency, frequency, magnitude)
- **Rolling statistics** (means, stds, medians over windows)
- **Temporal dynamics** (slopes, deltas, volatility, trend breaks)

<img width="701" height="516" alt="image" src="https://github.com/user-attachments/assets/fc16a15b-22e0-48ee-9276-462481541024" />


This compresses 90 days of data into a tabular format usable by:
- Random Forest
- XGBoost
- Logistic Regression

<img width="642" height="68" alt="image" src="https://github.com/user-attachments/assets/234909aa-5acd-4e2e-936c-69d5c107d112" />


---

### 2️⃣ Distance-Based Nearest Neighbors (Surprisingly Strong Baselines)

A simple yet powerful approach is **k-NN with Dynamic Time Warping (DTW)**.

- DTW compares **entire sequences**
- Handles temporal shifts and misalignment
- Distance-weighted voting improves robustness

```python
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
```

> sktime is a powerful Python library for time series analysis — particularly strong in classification tasks, where it offers far greater versatility and breadth of model options compared to packages like Darts or PyTorch Forecasting.

<img width="650" height="64" alt="image" src="https://github.com/user-attachments/assets/170e93f0-0839-416b-86e9-acb6971d7055" />


---

### 3️⃣ Deep Sequence Models (Learn the Shape for You)

#### a) LSTM / GRU

Recurrent Neural Networks excel when patterns unfold gradually over time.

- Capture long-range dependencies
- Learn smooth degradation patterns
- Preserve temporal order

Both **custom PyTorch LSTM models** and `sktime`’s `LSTMFCNClassifier` perform well.

<img width="545" height="64" alt="image" src="https://github.com/user-attachments/assets/b2dac649-6412-4fd2-a524-f0b626fe8888" />


---

#### b) 1D CNNs — InceptionTime

**InceptionTime** is a deep learning architecture designed specifically for time-series classification, inspired by the Inception modules from computer vision (originally used in Google’s InceptionNet).

<img width="545" height="72" alt="image" src="https://github.com/user-attachments/assets/ca2963b1-0837-4894-bd72-59cdba1d102c" />

---

#### c) ResNetClassifier

ResNet adapts residual connections from computer vision to time series:

- Skip connections enable deep architectures
- uses residual blocks — shortcuts that connect earlier layers directly to later ones
- Captures both local and global temporal patterns

<img width="619" height="197" alt="image" src="https://github.com/user-attachments/assets/cd87c82c-5408-4ace-be5e-320fa93e12fb" />

---

## 🔀 Fusion: Combining Sequential and Static Features

Behavioral features tell *how* behavior evolves —  
Static features explain *who* the entity is.

Examples of static features:
- Age
- Plan type
- Location
- Tenure

### Fusion Strategies

So far, our models have focused on behavioral features — those that evolve over time, such as data consumption, signal strength, or latency.

However, in many real-world problems, we also have static features — attributes that remain constant for each entity, such as a customer’s age, gender, location, or plan type. These features often carry valuable contextual information that complements the temporal signals.

> how can we combine static features with time-series data in a single model?

There are several strategies to achieve this fusion.

1. **Broadcasting**  
   Static features are repeated across time as additional channels

2. **Dual-Input Architecture**
   - One branch for sequences (LSTM / CNN / ROCKET)
   - One branch for static features (MLP)
   - Outputs concatenated before classification

```python

# --- Model: LSTM branch + Static MLP branch ---
ts_in = Input(shape=(90, 10), name="ts_in")
x = LSTM(128, return_sequences=True)(ts_in)
x = Dropout(0.3)(x)
x = LSTM(64)(x)
x = Dropout(0.3)(x)

stat_in = Input(shape=(X_static.shape[1],), name="static_in")
s = Dense(64, activation='relu')(stat_in)
s = Dropout(0.3)(s)

z = Concatenate()([x, s])
z = Dense(64, activation='relu')(z)
z = Dropout(0.2)(z)
out = Dense(1, activation='sigmoid')(z)

model = Model([ts_in, stat_in], out)
```

<img width="1205" height="91" alt="image" src="https://github.com/user-attachments/assets/4a258935-938a-4c19-9acb-be214133b4d1" />




```python
from pytorch_forecasting import TimeSeriesDataSet
TimeSeriesDataSet(
    data,
    group_ids=["customer_id"],
    time_varying_unknown_reals=["usage", "latency"],
    static_reals=["age", "tenure"],
    static_categoricals=["plan_type", "region"],
)

```


This fusion enables richer representations and often leads to **significant accuracy gains**.

---

## 🏁 Conclusion

- Time-series classification captures **behavioral evolution**, not static correlation
- Feature-based ML provides fast, interpretable baselines
- Distance-based methods are strong but hard to scale
- Deep learning models (LSTM, InceptionTime, ResNet) significantly outperform classical approaches
- Combining **static + temporal features** yields the most powerful models

---

## 📦 Popular Time Series Packages

- `sktime` — strongest ecosystem for time-series classification
- `tsai` — deep learning for TSC
- `PyTorch Forecasting` — sequence models with embeddings
- `Darts` — end-to-end time series pipelines

---

**Topics:**  
Time-Series Classification · Telecom Churn · Deep Learning · LSTM · CNN · DTW
