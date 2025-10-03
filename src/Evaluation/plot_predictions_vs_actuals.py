import plotly.graph_objects as go
import numpy as np
import pandas as pd

def plot_predictions_vs_actuals(
    y_pred,
    y_true,
    timestamps,
    title="Model Predictions vs Actuals",
    metrics: dict = None,
    extra_title: str = None,
    date_range: tuple = None
):
    """
    Create an interactive line plot comparing predictions vs actuals, with error curve.

    Parameters
    ----------
    y_pred : np.ndarray
        Predicted values.
    y_true : array-like
        Actual observed values.
    timestamps : array-like
        Time index corresponding to the values.
    title : str, optional
        Base title of the plot.
    metrics : dict, optional
        Dictionary of metrics (e.g., {"MAE": 0.1, "RMSE": 0.2}).
        Rendered in the plot title.
    extra_title : str, optional
        Additional text appended to the title.
    date_range : tuple, optional
        (start_date, end_date) as strings or pd.Timestamp.
        Filters the plot to this date range.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive figure object (not shown automatically).
    """

    # Convert inputs to arrays/pd.Series
    y_pred = np.asarray(y_pred).ravel()
    y_true = np.asarray(y_true).ravel()
    timestamps = pd.to_datetime(timestamps)

    # Validate equal length
    if not (len(y_pred) == len(y_true) == len(timestamps)):
        raise ValueError(
            f"Length mismatch: y_pred={len(y_pred)}, y_true={len(y_true)}, timestamps={len(timestamps)}"
        )

    # Build DataFrame for easier handling
    df_plot = pd.DataFrame({
        "timestamp": timestamps,
        "actual": y_true,
        "predicted": y_pred
    })
    df_plot["error"] = df_plot["actual"] - df_plot["predicted"]

    # Apply date filtering if provided
    if date_range is not None:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        df_plot = df_plot[(df_plot["timestamp"] >= start) & (df_plot["timestamp"] <= end)]

    # Build plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_plot["timestamp"], y=df_plot["actual"],
        mode="lines", name="Actual",
        line=dict(color="black")
    ))

    fig.add_trace(go.Scatter(
        x=df_plot["timestamp"], y=df_plot["predicted"],
        mode="lines", name="Predicted",
        line=dict(color="red", dash="dot")
    ))

    fig.add_trace(go.Scatter(
        x=df_plot["timestamp"], y=df_plot["error"],
        mode="lines", name="Error (Actual - Predicted)",
        line=dict(color="blue", dash="dash")
    ))

    # Construct dynamic title
    if metrics is not None:
        metrics_text = " | ".join([f"{k}: {v:.3f}" if isinstance(v, (float, np.floating)) else f"{k}: {v}" 
                                   for k, v in metrics.items()])
        title += f"<br><sup>{metrics_text}</sup>"

    if extra_title is not None:
        title += f"<br><sup>{extra_title}</sup>"

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Energy Consumption",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(x=0, y=1.05, orientation="h")
    )

    return fig


if __name__ == "__main__":
    # some example usage or test
    pass
    """
    fig = plot_predictions_vs_actuals(
                y_pred,
                sample_test_df["energy_consumption"],
                sample_test_df["timestamp"],
                metrics=metrics_dict,
                extra_title="Validation set: Household A",
                date_range=("2014-01-01", "2014-01-08")
                )   """