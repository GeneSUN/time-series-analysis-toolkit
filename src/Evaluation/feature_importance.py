
import matplotlib.pyplot as plt
import pandas as pd

def plot_feature_importance(model, top_n=20, title="Top Features by Coefficient"):
    """
    Plot top N most important features from a fitted sklearn Pipeline with
    ColumnTransformer + linear model (e.g., Ridge, Lasso).

    Parameters
    ----------
    model : sklearn Pipeline
        Fitted pipeline with named steps: 'preprocessor' and 'regressor'.
    top_n : int, optional (default=20)
        Number of top features to display.
    title : str, optional
        Title for the plot.

    Returns
    -------
    coef_df : pd.DataFrame
        DataFrame with feature names and coefficients (sorted by importance).
    """
    # --- get feature names after preprocessing ---
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    
    # --- get coefficients from final estimator ---
    coefs = model.named_steps["regressor"].coef_
    
    # --- combine into DataFrame ---
    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefs,
        "abs_importance": abs(coefs)
    }).sort_values(by="abs_importance", ascending=False).head(top_n)

    # --- plotting ---
    plt.figure(figsize=(10, 6))
    plt.barh(coef_df["feature"], coef_df["coefficient"])
    plt.xlabel("Coefficient")
    plt.title(title)
    plt.gca().invert_yaxis()  # highest on top
    plt.show()

    return coef_df
