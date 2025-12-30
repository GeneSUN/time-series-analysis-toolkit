# From Local to Global: How One Model Can Forecast Thousands of Time Series

---

This article focuses on **Global Forecasting Models (GFMs)**, meta-feature encoding, and embedding-based representations for scalable time series forecasting.

A [Google Colab notebook](https://colab.research.google.com/drive/17HKMecEdrzReMsiTd6ITEMR6izDeY3Nw) is provided to **reproduce and practice** all concepts discussed here.

---
## 📑 Table of Contents

- [Local vs Global](#-local-vs-global)
  - [Local Models (Traditional Approach)](#local-models-traditional-approach)
  - [Global Models (Modern Approach)](#-global-models-modern-approach)
  - [Core Assumption](#-core-assumption)
- [Advantages of Global Forecasting Models](#-advantages-of-global-forecasting-models)
  - [Cross-Learning](#1️⃣-cross-learning)
    - [1.1 Data Pooling — Learn What You Don’t Have](#11-data-pooling--learn-what-you-dont-have)
    - [1.2 Regularization Through Sharing](#12-regularization-through-sharing)
  - [Engineering Simplicity](#2️⃣-engineering-simplicity)
- [Representation Learning for GFMs](#-representation-learning-for-gfms)
  - [Why Meta-Features Matter](#why-meta-features-matter)
- [Meta-Feature Conditioning](#-meta-feature-conditioning)
  - [For Machine Learning Models](#for-machine-learning-models)
- [Embedding-Based Representations (Deep Learning)](#-embedding-based-representations-deep-learning)
  - [Why Embeddings?](#why-embeddings)
  - [Example](#-example)
- [Semi-Global Models (Hybrid Strategy)](#-semi-global-models-hybrid-strategy)
- [Conclusion — Key Takeaways](#-conclusion--key-takeaways)


---

## 🚀Local vs Global

Imagine building a temperature forecasting system for **thousands of weather stations worldwide**.

<img width="1960" height="808" alt="image" src="https://github.com/user-attachments/assets/bc33a2cc-50c4-45a1-ac07-683569d45177" />


### Local Models (Traditional Approach)

The traditional approach treats each station as an isolated problem — you train one model per location, learning how that particular city’s temperature behaves.

**Problems:**
- Re-learns the same patterns repeatedly (daily cycles, seasonality), Wastes time on shared information
- Managing thousands of models is operationally impractical

---

### 🌍 Global Models (Modern Approach)

A **Global Forecasting Model (GFM)** trains **one single model across all time series**.

**Why It Works**

- Learns **shared patterns** across series
- Adapts to **local variations**
- Scales efficiently
- Often achieves *better generalization and accuracy*

---

### 🧠 Core Assumption

All time series are **related** and generated from the same underlying **Data Generating Process (DGP)**.

Examples:
- Retail sales share holiday seasonality
- Energy usage responds similarly to temperature
- Weather stations share climate dynamics

These dependencies allow global models to **borrow strength across series**.

---

## ✅ Advantages of Global Forecasting Models

### 1️⃣ Cross-Learning

#### 1.1 Data Pooling — Learn What You Don’t Have

In many real-world forecasting tasks, each time series is short, making it difficult to train reliable models without overfitting. Building a separate model for every individual series is often impractical.

**Example:** 

<img width="1400" height="792" alt="image" src="https://github.com/user-attachments/assets/a2c79248-4962-4f92-92a4-c8a8c64f34d2" />


imagine trying to forecast inventory levels in the Turkish market, which has only a few months of historical data because the market was launched recently. 
- A local model would have too little information to learn meaningful patterns.

> Instead of training one model per series, a Global Forecasting Models (GFMs) learns from many related time series across similar contexts.

**Example:**
using inventory data from multiple European markets, the model can capture shared patterns such as seasonality, promotions, and regional demand cycles.
- By generalizing this collective knowledge, the GFM can make accurate predictions even for new or data-scarce markets like Turkey.

#### 1.2 Regularization Through Sharing

Cross-learning acts as a **regularizer**:
- Smooths noise
- Reinforces stable global patterns (e.g., yearly seasonality)

For example, imagine forecasting daily temperatures for several nearby cities.
<img width="1400" height="813" alt="image" src="https://github.com/user-attachments/assets/453f3ea0-d5b7-450e-934d-c5e043b1643a" />

- Each city’s data may contain short-term fluctuations due to local weather events,
- but when the GFM learns seasonality collectively across all cities, it identifies the broader yearly cycle — cold winters, warm summers — without being misled by local noise.


In essence, GFMs transform individual, data-scarce problems into a collective learning task, improving both robustness and accuracy.


---

### 2️⃣ Engineering Simplicity

| Aspect | Local Models | Global Models |
|------|-------------|--------------|
| Training | Thousands of models | One model |
| Tuning | Repeated per series | Centralized |
| Monitoring | Operational nightmare | Simple |
| Scalability | Poor | Excellent |

---

## 🧩 Representation Learning for GFMs

### Why Meta-Features Matter

A naive global model treats all series as identical 
-  That works fine for homogeneous datasets (e.g., sensors measuring the same phenomenon),
-  but not for heterogeneous datasets where different series follow distinct dynamics (e.g., sales of different product categories).

Introducing meta-features (like product category, region, or store ID) allows the model to condition its forecasts on the characteristics of each series.

---

## 🏷️ Meta-Feature Conditioning

- Add categorical or numerical meta-features describing each series (e.g., region, product category, store ID).
- The model uses these as identifiers or context signals, enabling it to learn both shared global patterns and series-specific nuances.
- This makes the model flexible: it can generalize across groups while specializing within them.


### For Machine Learning Models

```
pip install category-encoders
```

**Encoding Methods:**

1. **One-Hot Encoding**
   - Simple baseline
   - High dimensional, sparse
   - No notion of similarity

2. **Target / Mean Encoding**
   - Uses target statistics
   - Powerful for tree models
   - Must avoid data leakage (use CV)

3. **Native Categorical Handling**
   - CatBoost, LightGBM, XGBoost
   - Strong tabular baselines
   - No explicit embeddings needed

**Limitations of One-Hot:**
- The number of new dimensions equals the cardinality (number of unique values) of the categorical variable.
- Each encoded vector contains mostly zeros — only one position is “1.”
- This results in a sparse and memory-inefficient representation.
- Lack of Semantic Information. One-hot encoding treats all categories as equally distant, ignoring similarities or relationships among them.


---

## 🧬 Embedding-Based Representations (Deep Learning)

Embeddings provide **dense, low-dimensional representations** of categorical features.

### Why Embeddings?

- Compact and efficient
- Learn semantic similarity automatically
- Scales to high-cardinality features
- Widely used in NLP and recommender systems

<img width="2000" height="723" alt="image" src="https://github.com/user-attachments/assets/e56ec36c-6590-4d31-b95b-a1e5942b737c" />


---

### 🧪 Example: 

Deep learning–based time series models incorporate embeddings directly as part of the model parameters.

```python
# PyTorch Forecasting
from pytorch_forecasting import TimeSeriesDataSet

embedding_sizes = {
    "store_id": (100, 12),
    "region_id": (5, 4),
    "product_id": (2000, 32),
}

ds = TimeSeriesDataSet(
    data,
    group_ids=["series_id"],
    static_categoricals=["store_id", "region_id", "product_id"],
    time_varying_known_categoricals=["dow", "month"],
    embedding_sizes=embedding_sizes,
)
```


```python
# GluonTS (DeepAR)
from gluonts.model.deepar import DeepAREstimator

estimator = DeepAREstimator(
    prediction_length=14,
    freq="D",
    cardinality=[100, 5, 2000],
)
```

---

## 🧠 Semi-Global Models (Hybrid Strategy)

**Idea:**  
Cluster similar time series, then train **one model per cluster**.

- Group similar time series based on statistical or learned representations (e.g., correlation, dynamic time warping, or embeddings).
- Train one semi-global model per cluster, so that each model focuses on series with similar dynamics.
- This hybrid approach balances data efficiency (like GFMs) with specificity (like LFMs).

- https://medium.com/@injure21/time-series-clustering-from-dtw-to-deep-embeddings-ts2vec-autoencoders-f1c1517d9025
- https://colab.research.google.com/drive/1v805hpfMX8Z5xYkLTMQ3N3EINDks5OAJ



## Challenges

### 1️⃣ Heterogeneity Across Time Series (The Core Challenge)

Global models assume that time series are related, but in practice they are often only partially related.

**Problems:**

- Different series may have:
  - Different seasonality (weekly vs yearly)
  - Different scales and noise levels
  - Different trend behaviors
- A single global model can over-generalize, hurting performance on unique series

**Typical Symptoms**
- Strong performance on “average” series
- Poor performance on rare or edge-case series


### 2️⃣ Evaluation Is Harder Than Local Models

Evaluating a single series is easy. Evaluating thousands is not.

**Problems**

- Aggregated metrics hide per-series failures
- Good average performance ≠ good per-entity performance

**Best Practices**

- Per-series metrics (quantiles, error distributions)
- Segment-based evaluation (by region, category)
- Worst-case and tail performance analysis

### 3️⃣ Interpretability & Debugging

Global models—especially deep learning ones—can feel like black boxes.

**Issues**

- Hard to explain why a forecast is wrong for a specific series
- Difficult to trace errors back to data or representation






