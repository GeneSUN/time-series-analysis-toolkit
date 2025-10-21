#https://colab.research.google.com/drive/13OhKqtGc1RDjN8DdyGhyDSN7oY0upVpa#scrollTo=JCMKm6ZpfX1o

#@title Function: greedy_ensemble_selection
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

def greedy_ensemble_selection(
    pred_wide_val,
    target_col,
    evaluate_fn=None,
    verbose=True,
):
    """
    Greedy (hill-climbing) ensemble selector with flexible user-defined evaluation.

    Parameters
    ----------
    pred_wide_val : pd.DataFrame
        DataFrame containing model predictions and a target column.
        Each row corresponds to a timestamp or sample.
    target_col : str
        Name of the target column in pred_wide_val.
    evaluate_fn : callable, optional
        A function taking (y_true, y_pred) -> float (smaller is better).
        If not provided, defaults to RMSE.
    verbose : bool
        If True, prints progress messages.

    Returns
    -------
    best_models : list of str
        Names of model columns selected for the ensemble.
    best_score : float
        The best (lowest) score achieved by the ensemble.
    """

    # -------------------------------------------------
    # Default evaluation = RMSE if not provided
    # -------------------------------------------------
    if evaluate_fn is None:
        def evaluate_fn(y_true, y_pred):
            return np.sqrt(mean_squared_error(y_true, y_pred))
        if verbose:
            print("Using default metric: RMSE")

    # -------------------------------------------------
    # Initialization
    # -------------------------------------------------
    model_cols = [c for c in pred_wide_val.columns if c != target_col]
    y_true = pred_wide_val[target_col].values

    # Step 1: Evaluate single models
    scores = {m: evaluate_fn(y_true, pred_wide_val[m]) for m in model_cols}
    best_model = min(scores, key=scores.get)
    best_models = [best_model]
    best_score = scores[best_model]
    remaining_models = [m for m in model_cols if m != best_model]

    if verbose:
        print(f"Start with: {best_model} (score={best_score:.5f})")

    # -------------------------------------------------
    # Step 2: Iteratively add models
    # -------------------------------------------------
    improved = True
    while improved and remaining_models:
        improved = False
        stage_results = []

        for m in remaining_models:
            temp_models = best_models + [m]
            y_pred = pred_wide_val[temp_models].mean(axis=1)
            score = evaluate_fn(y_true, y_pred)
            stage_results.append((m, score))

        # Pick model that gives the lowest score
        m_best, score_best = min(stage_results, key=lambda x: x[1])

        if score_best < best_score:
            best_models.append(m_best)
            best_score = score_best
            remaining_models.remove(m_best)
            improved = True
            if verbose:
                print(f"  + Added {m_best}, score improved to {best_score:.5f}")
        else:
            if verbose:
                print("No further improvement — stopping.")

    if verbose:
        print(f"\n✅ Final Ensemble: {best_models}")
        print(f"✅ Final Score: {best_score:.5f}")

    return best_models, best_score


#@title Function: stochastic_ensemble_selection
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import random

def stochastic_ensemble_selection(
    pred_wide_val,
    target_col,
    evaluate_fn=None,
    max_iterations=20,
    random_state=None,
    verbose=True,
):
    """
    Stochastic hill-climbing ensemble selector.
    Randomly picks a candidate model each step and accepts it if the
    new ensemble improves the score.

    Parameters
    ----------
    pred_wide_val : pd.DataFrame
        DataFrame containing model predictions and a target column.
        Each row corresponds to a timestamp or sample.
    target_col : str
        Name of the target column in pred_wide_val.
    evaluate_fn : callable, optional
        A function taking (y_true, y_pred) -> float (smaller is better).
        If not provided, defaults to RMSE.
    max_iterations : int, default=50
        Maximum number of random trials.
    random_state : int, optional
        Random seed for reproducibility.
    verbose : bool, default=True
        If True, prints progress messages.

    Returns
    -------
    best_models : list of str
        Models included in the best ensemble found.
    best_score : float
        Best (lowest) score achieved.
    history : list of dict
        Iteration-wise log of scores, candidates, and acceptance decisions.
    """

    # -------------------------------------------------
    # Setup
    # -------------------------------------------------
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)

    if evaluate_fn is None:
        def evaluate_fn(y_true, y_pred):
            return np.sqrt(mean_squared_error(y_true, y_pred))
        if verbose:
            print("Using default metric: RMSE")

    model_cols = [c for c in pred_wide_val.columns if c != target_col]
    y_true = pred_wide_val[target_col].values

    # Step 1: Start from the best single model
    scores = {m: evaluate_fn(y_true, pred_wide_val[m]) for m in model_cols}
    best_model = min(scores, key=scores.get)
    best_models = [best_model]
    best_score = scores[best_model]

    if verbose:
        print(f"Start with: {best_model} (score={best_score:.5f})")

    # -------------------------------------------------
    # Step 2: Randomized search (stochastic hill climbing)
    # -------------------------------------------------
    history = []
    for i in range(max_iterations):
        remaining_models = [m for m in model_cols if m not in best_models]
        if not remaining_models:
            if verbose:
                print("All models have been used — stopping.")
            break

        candidate = random.choice(remaining_models)
        temp_models = best_models + [candidate]
        y_pred = pred_wide_val[temp_models].mean(axis=1)
        score = evaluate_fn(y_true, y_pred)

        accepted = False
        if score < best_score:
            # Accept the move
            best_models.append(candidate)
            best_score = score
            accepted = True
            if verbose:
                print(f"[Iter {i+1:02d}] Randomly picked {candidate}: improved ✓ (score={best_score:.5f})")
        else:
            # Reject the move
            if verbose:
                print(f"[Iter {i+1:02d}] Randomly picked {candidate}: no improvement ✗ (score={score:.5f})")

        history.append({
            "iteration": i + 1,
            "candidate": candidate,
            "accepted": accepted,
            "score": score,
            "best_score": best_score,
            "current_ensemble": best_models.copy(),
        })

    if verbose:
        print(f"\n✅ Final Ensemble: {best_models}")
        print(f"✅ Final Score: {best_score:.5f}")

    return best_models, best_score, history


from scipy import optimize
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

def find_optimal_combination(
    candidates,
    pred_wide,
    target,
    metric_fn=mean_absolute_error,
    verbose=True,
):
    """Find optimal non-negative weights (sum=1) minimizing forecast error."""

    def loss_function(weights):
        fc = np.sum(pred_wide[candidates].values * np.array(weights), axis=1)
        return metric_fn(pred_wide[target].values, fc)

    n = len(candidates)
    init_w = np.ones(n) / n

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = [(0.0, 1.0)] * n

    res = optimize.minimize(
        loss_function,
        x0=init_w,
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
        options={"ftol": 1e-9, "disp": False},
    )

    weights = res.x

    if verbose:
        print(f"Optimization success: {res.success}")
        print(f"Optimal Weights: {dict(zip(candidates, np.round(weights, 4)))}")
        print(f"Minimum {metric_fn.__name__}: {res.fun:.5f}")

    return weights
