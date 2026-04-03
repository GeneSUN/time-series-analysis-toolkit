# Time Series Clustering — From DTW to Deep Learning Embeddings  
### *(TS2Vec, Autoencoders, and Practical Trade-offs)*

---

This repository accompanies the article **“Time Series Clustering — From DTW to Deep Learning Embeddings.”**  
It focuses on **time-aware clustering methods** that respect temporal structure, with practical guidance on *when to use what*.

A **[Google Colab notebook](https://colab.research.google.com/drive/1v805hpfMX8Z5xYkLTMQ3N3EINDks5OAJ)** is provided to reproduce and practice everything discussed here.


---

## 📑 Table of Contents

- [Why Time Series Clustering Is Different](#why-time-series-clustering-is-different)
- [1️⃣ Distance-Based Clustering (DTW)](#1️⃣-distance-based-clustering-time-aware)
- [2️⃣ Feature-Learning / Model-Based Clustering](#2️⃣-feature-learning-model-based-clustering)
- [3️⃣ Autoencoder Embeddings](#3️⃣-autoencoder-embeddings-deep-learned)
- [4️⃣ Pretrained Deep Embeddings (TS2Vec)](#4️⃣-pretrained-deep-learning-embeddings-ts2vec)
- [Final Notes & Practical Guidance](#final-notes--practical-guidance)

---

## Why Time Series Clustering Is Different

Clustering time series is **not the same** as clustering tabular feature vectors.

- A common mistake is to treat a length‑T time series as a T‑dimensional point and apply Euclidean clustering.  
- This ignores the **ordered and dependent** nature of temporal data.

### Why Euclidean Distance Fails

- In tabular data, features are independent and unordered
- In time series, observations are **sequentially correlated**
- So, a 60-length series ≠ a 60-dimensional feature vector. Because ```x_t``` depends on ```x_{t-1}```, and temporal structure matters.

**Intuition Example**
- Series A: a clean sine wave  
- Series B: the same sine wave, shifted 3 time steps  

<p align="center">
  <img src="https://github.com/user-attachments/assets/6954144c-60e7-4c44-a2d5-dedab3ce4f2e" width="48%" />
  <img src="https://github.com/user-attachments/assets/577c0a96-edf0-4cc2-84e1-95b7c9a38cdb" width="48%" />
</p>



Visually identical — yet Euclidean distance is large.  
This motivates **time-aware distances or learned representations**.

---

## 1️⃣ Distance-Based Clustering (Time-Aware)

Instead of Euclidean distance, use distances designed for sequences.

### Dynamic Time Warping (DTW)

<img width="630" height="475" alt="image" src="https://github.com/user-attachments/assets/a396cec5-1f3f-4c98-8c11-7c2050e3f2b9" />


**What it does**
- Aligns two sequences non-linearly in time
- Handles local shifts and stretching

<img width="603" height="790" alt="image" src="https://github.com/user-attachments/assets/3b33ce7c-ad0a-446d-bcbf-ff72f2f18242" />


**When to use**
- Short to mid-length series
- Moderate dataset sizes

**How to cluster**
- k-medoids (PAM)
- Hierarchical clustering on a DTW distance matrix

```python
from tslearn.clustering import TimeSeriesKMeans

model = TimeSeriesKMeans(metric="dtw", n_clusters=5)
```

**Pros**
- Robust to phase shifts
- Intuitive and widely used

**Cons**
- Pairwise DTW is expensive: O(N²)
- Poor scalability without approximations

---

## 2️⃣ Feature-Learning / Model-Based Clustering

Instead of raw sequences, **fit a model to each series** and cluster the learned parameters.

### Idea

- Fit ARIMA / VAR / HMM per series
- Use model parameters as feature vectors
- Cluster in parameter space

**Why it works**
- Parameters encode dynamics (trend, persistence, regimes)

### Example: ARIMA Representation

For an AR(3):

$$y_t = φ₀ + φ₁y_{t−1} + φ₂y_{t−2} + φ₃y_{t−3} + ε_t$$

- The coefficients [φ₀, φ₁, φ₂, φ₃] capture the series’ dynamics (trend, persistence, etc.).
- Thus, each time series can be represented compactly by these few parameters, turning an entire sequence into a small feature vector for clustering.

<img width="419" height="121" alt="image" src="https://github.com/user-attachments/assets/6bf50ebb-9739-489d-9ef8-6a8a60b36de2" />

<img width="590" height="790" alt="image" src="https://github.com/user-attachments/assets/3bb588a2-d60e-4a22-bbfa-341f18a2d027" />


**Trade-offs**
- Requires reliable model fits
- Sensitive to noise and misspecification
- Computationally heavy

---

## 3️⃣ Autoencoder Embeddings (Deep Learned)

Instead of hand-crafted features, **learn representations automatically**.

### Workflow

1. Train a sequence autoencoder (LSTM or 1D CNN)
2. Extract the latent (bottleneck) representation
3. Cluster embeddings with k-means or DBSCAN

<img width="1349" height="438" alt="image" src="https://github.com/user-attachments/assets/d02a5238-670e-486c-913a-65430f9adacd" />

<img width="502" height="547" alt="image" src="https://github.com/user-attachments/assets/5c8b872c-4448-4337-934d-3fa63d85f477" />


**Pros**
- Learns task-agnostic temporal features
- Effective for large, related datasets

**Cons**
- Requires training and tuning
- Architecture-sensitive
- More effort than classical methods

Instead of training from scratch, **pretrained models** can be used.

---

## 4️⃣ Pretrained Deep Learning Embeddings (TS2Vec)

### TS2Vec — Towards Universal Representation of Time Series

- GitHub: https://github.com/zhihanyue/ts2vec  
- Paper: https://arxiv.org/abs/2106.10466  

**What TS2Vec does**
- Produces fixed-length embeddings for entire time series
- Captures trends, periodicity, and shape patterns
- You can later apply Clustering on the latent vectors.

**Key Idea**
> Distance in embedding space ≈ semantic similarity of time series behavior

### NLP Analogy

A time series can be viewed as a sentence — both are ordered sequences where context matters.

- In natural language processing (NLP), models such as BERT or Word2Vec learn embeddings that capture the semantic meaning of words within their context.
- Similarly, in time series analysis, models like TS2Vec learn embeddings that capture temporal dynamics — such as trends, periodicity, and shape patterns — across time steps.

Therefore, TS2Vec functions much like a pretrained embedding model in NLP, providing fixed-length vector representations that encode the “meaning” or behavior of entire time series.

---

## 🏁 Final Notes & Practical Guidance
**Treat a time series as a sequence, not a vector**
- Need alignment robustness? → **DTW**
- Many related series? → **Autoencoders or TS2Vec**
- Need interpretability? → **Model-based features**
- Need scalability? → **Pretrained embeddings**

---

**Topics:**  
Time Series Clustering · DTW · Autoencoders · TS2Vec · Representation Learning
