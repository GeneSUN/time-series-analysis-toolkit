# Beyond Single-Source Learning-How Fusion Models Combine Time Series and Static Features

*Feature Engineering · Model Fusion · Ensemble Learning*

---

In real-world prediction problems, data rarely comes from a single source.

A telecom customer, for example, has:
- **Temporal behaviors**: daily data usage, latency, signal strength  
- **Static attributes**: age, device type, plan tier  

Traditional models struggle to combine these effectively — they either:
- flatten time series into coarse aggregates, or  
- ignore static profile context entirely  

This is where **fusion models** come in.

A **Google Colab notebook** accompanies this article to demonstrate all fusion strategies in practice:  
👉 https://colab.research.google.com/drive/13OhKqtGc1RDjN8DdyGhyDSN7oY0upVpa <br>
👉 https://github.com/GeneSUN/time-series-analysis-toolkit/blob/main/src/Ensemble/ensemble.py

---

## 📑 Table of Contents

- [What Is Fusion?](#what-is-fusion)
- [Levels of Fusion](#levels-of-fusion)
  - [1️⃣ Early (Feature-Level) Fusion](#1️⃣-early-feature-level-fusion--mix-before-you-cook)
  - [2️⃣ Intermediate (Representation-Level) Fusion](#2️⃣-intermediate-representation-level-fusion--cook-each-component-then-combine)
  - [3️⃣ Late (Decision-Level) Fusion / Ensembles](#3️⃣-late-decision-level-fusion-ensemble--vote-after-cooking-all-dishes)
    - [Greedy Hill Climbing](#a-greedy-hill-climbing)
    - [Stochastic Hill Climbing](#b-stochastic-hill-climbing)
    - [Optimal Weighted Ensemble / Stacking](#c-optimal-weighted-ensemble--stacking)
- [Key Takeaway](#-key-takeaway)

---

## What Is Fusion?

Fusion is **not a single architecture** — it is a **design philosophy**.

> Fusion is about *how and when* different information streams interact.

It means combining information from multiple sources (modalities or views) so a model can make better predictions than it could from any single source alone.


### Examples

- Text + image (visual question answering)

<img width="2000" height="346" alt="image" src="https://github.com/user-attachments/assets/7aab7f83-3cc7-4236-8f24-073ba3de3592" />

- Time series + static profile (telecom churn)

### Why Fusion Helps
Different sources carry **complementary signals**:
- One source may be noisy or missing
- Others can compensate

**Analogy:**  
Think of each modality as a person in a meeting.  
Fusion defines combine different person opinion together before making a decision.

---

## Levels of Fusion

Fusion strategies can be grouped into **three levels**, from simplest to most flexible.

1). Early/feature-level fusion


2). Intermediate/representation fusion

3). Late decision fusion/Ensemble


---

## 1️⃣ Early (Feature-Level) Fusion — *“Mix Before You Cook”*

### Idea
Concatenate static features with temporal features **before modeling**.  
Static attributes are treated as extra channels repeated across time.

```python
X_time = (n_samples, time_steps, n_features)
X_static = repeat(static_features, repeats=time_steps, axis=1)
X_fused = concatenate([X_time, X_static], axis=-1)
```

<img width="1992" height="1272" alt="image" src="https://github.com/user-attachments/assets/71b15feb-fb88-4335-b0cc-c0581c887dcd" />


### Pros
- Works with off-the-shelf time-series models (ResNet, InceptionTime)
- Simple to implement
- No architecture changes required

### Cons
- Static data is repeated at every timestep (redundant)
- Model may misinterpret static features as dynamic context
- Higher risk of overfitting

---

## 2️⃣ Intermediate (Representation-Level) Fusion — *“Cook Each Component, Then Combine”*

### Idea
Build **separate encoders** for each data source, then merge learned representations.

```python
model_time = LSTM(64)(time_series_input)
model_stat = Dense(32, activation="relu")(static_input)

fusion = Concatenate()([model_time, model_stat])
output = Dense(1, activation="sigmoid")(fusion)
```

<img width="2000" height="245" alt="image" src="https://github.com/user-attachments/assets/026ed731-621e-44a7-a2d2-75de5dbf42ca" />


### Pros
- Clear modularity
- Each modality learns its own representation
- Supports attention, gating, and weighting
- Strong balance between performance and interpretability

### Cons
- Requires custom model design
- More tuning effort

---

## 3️⃣ Late (Decision-Level) Fusion / Ensemble — *“Vote After Cooking All Dishes”*

### Idea
Train **separate models per modality**, then combine predictions.

```python
p1 = model_time_series.predict(X_time)
p2 = model_static.predict(X_static)

final_pred = 0.7 * p1 + 0.3 * p2
```

This approach is flexible and often very strong in practice.

---

### a) Greedy Hill Climbing

<img width="1400" height="1146" alt="image" src="https://github.com/user-attachments/assets/1e2b02be-aff4-4a1e-9015-5e8a0b05033d" />

1). Start simple:
Evaluate all individual candidate models and choose the one with the best performance (lowest error). This becomes your starting point.

2). Add one model at a time:
Combine the current ensemble with each of the remaining candidates, one by one, and evaluate the new performance.

3). Keep the best improvement:
If adding one of the models lowers the error, accept it and move forward with this new ensemble.
If none improves it, stop — you’ve reached a local optimum.

4). Repeat until no gain:
Continue adding models greedily (always picking the one that gives the biggest improvement at each step) until further additions no longer help.

5). Outcome:
You end up with a small, high-performing subset of models. It may not be the global best combination (since it doesn’t explore all possibilities), but it’s fast, interpretable, and often performs surprisingly well.


**Outcome:**  
A small, high-performing ensemble that is fast and interpretable.

---

### b) Stochastic Hill Climbing

<img width="1228" height="1336" alt="image" src="https://github.com/user-attachments/assets/26044554-ba33-4db5-9aff-c487230dba20" />



1. Start from the best single model  
2. Generate neighbors by adding/removing one model  
3. Randomly select a neighbor  
4. Accept it only if performance improves  
5. Repeat until convergence  

**Why stochastic?**  
Randomness helps escape local optima and improves robustness.

---

### c) Optimal Weighted Ensemble / Stacking

- Collect base models (e.g., LightGBM, Lasso, XGBoost, Theta)
- Train a **meta-learner** (e.g., logistic regression)
- Optimize non-negative weights under a simplex constraint
- Compute final prediction as a weighted sum

<img width="1754" height="1288" alt="image" src="https://github.com/user-attachments/assets/3f43c99c-cf82-4297-8c74-61a8aac37d2b" />


This approach often yields **near-optimal ensembles** when enough validation data is available.

---

## 🚀 Key Takeaway

Fusion is **not about stacking more layers** — it’s about **structuring information flow**.

**Orchestra analogy:**
- Early fusion → mix instruments into one melody  
- Intermediate fusion → harmonize distinct parts  
- Late fusion → let each instrument solo, then blend  

Choosing the right fusion level depends on:
- Data diversity
- Interpretability requirements
- Engineering budget

Understanding these three levels gives you a **clear mental map** for designing modern predictive systems that go beyond single-source learning.
