# Evaluation & plotting
import pandas as pd
import numpy as np
from darts.metrics import mape, rmse, mae, smape
from darts.metrics import mql, mic
import matplotlib.pyplot as plt

# ---- helpers (same shapes as your CV code) ----
def _ts_to_df(ts, name="y"):
    arr = ts.values(copy=False)
    if arr.ndim == 3:
        arr = arr[:, 0, 0]  # (time, component, sample)
    elif arr.ndim == 2:
        arr = arr[:, 0]     # (time, component)
    else:
        arr = np.asarray(arr)
    return pd.DataFrame({name: arr}, index=ts.time_index)

def _safe_mape(y_true, y_pred, eps=1e-1):
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_pred - y_true) / denom)) * 100.0)

def _smape(y_true, y_pred, eps=1e-1):
    denom = np.maximum((np.abs(y_true) + np.abs(y_pred)) / 2.0, eps)
    return float(np.mean(np.abs(y_pred - y_true) / denom) * 100.0)


# ---- Model Evaluation / Plotting after fitting----
def eval_model(model, n, actual_series, val_series, past_covs_scaled, future_covs_scaled, well_idx, y_scaler, num_samples=100, 
               low_q=0.1, high_q=0.9, lowest_q=0.05, highest_q=0.95, 
               label_q_inner='50%-90% prediction interval', label_q_outer='5%-95% prediction interval', 
               figsize=(10, 6)):
    """
    Evaluate and plot a single well's time series prediction with quantile intervals and debug info.
    """
    # Extract the specific well's series
    actual_well = actual_series[well_idx]
    val_well = val_series[well_idx] if val_series is not None and isinstance(val_series, list) else None
    
    # Extract well-specific covariates
    past_cov_well = past_covs_scaled[well_idx] if past_covs_scaled is not None else None
    future_cov_well = future_covs_scaled[well_idx] if future_covs_scaled is not None else None
    
    # Predict for the selected well with covariates
    pred_series = model.predict(n=n, num_samples=num_samples, series=actual_well,
                                past_covariates=past_cov_well, future_covariates=future_cov_well)
    
    # Inverse transform to original scale
    actual_well_unscaled = y_scaler.inverse_transform(actual_well)
    pred_unscaled = y_scaler.inverse_transform(pred_series)
    val_unscaled = y_scaler.inverse_transform(val_well) if val_well is not None else None
    
    # Debug: Inspect quantile values
    q10 = pred_unscaled.quantile(0.1)
    q50 = pred_unscaled.quantile(0.5)
    q90 = pred_unscaled.quantile(0.9)
    print(f"Q10 Values: {q10.values()}")
    print(f"Q50 Values: {q50.values()}")
    print(f"Q90 Values: {q90.values()}")
    
    # Debug: Inspect raw prediction values
    pred_values = pred_unscaled.values()
    print(f"Raw Prediction Values Shape: {pred_values.shape}")
    print(f"Raw Prediction Values: {pred_values}")  # Should show multiple samples or quantiles if probabilistic
    
    # Plot
    plt.figure(figsize=figsize)
    actual_well_unscaled[:pred_unscaled.end_time()].plot(label="Actual")
    
    # Plot prediction with quantile ranges
    try:
        pred_unscaled.plot(low_quantile=lowest_q, high_quantile=highest_q, label=label_q_outer)
        pred_unscaled.plot(low_quantile=low_q, high_quantile=high_q, label=label_q_inner)
        print("Quantile plotting succeeded, indicating probabilistic output.")
    except ValueError as e:
        pred_unscaled.plot(label="Prediction")
        print(f"Quantile plotting failed: {e}")
    
    plt.title(f"Well {well_idx}")
    plt.xlabel("Date")
    plt.ylabel("Oil Rate (bpd)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Print prediction details
    print(f"Prediction Components for Well {well_idx}: {pred_unscaled.components}")
    print(f"Prediction Values Shape: {pred_unscaled.values().shape}")

# ---- main: metrics after cross-validation ----

def metrics_from_cv(cv_list, series_list_scaled, y_scaler, quantiles=[0.10, 0.50, 0.90], compute_global=False):
    n_series = len(cv_list)
    if n_series == 0:
        return {
            "per_series_window": pd.DataFrame(columns=["series_idx", "window_idx", "rmse", "mae", "mape", "smape", "mql_0_50", "mic_10_90"]),
            "overall": {} if compute_global else None
        }

    n_windows = max((len(cv_list[i]) for i in range(n_series)), default=0)
    if n_windows == 0:
        return {
            "per_series_window": pd.DataFrame(columns=["series_idx", "window_idx", "rmse", "mae", "mape", "smape", "mql_0_50", "mic_10_90"]),
            "overall": {} if compute_global else None
        }

    rows = []
    for i in range(n_series):
        for j in range(n_windows):
            if j >= len(cv_list[i]):
                continue

            # Inverse transform prediction and truth
            pred_ts = y_scaler.inverse_transform([cv_list[i][j]])[0]
            y_true_ts = y_scaler.inverse_transform([series_list_scaled[i].slice(pred_ts.start_time(), pred_ts.end_time())])[0]

            # Convert to DataFrames and align
            pred_df = _ts_to_df(pred_ts.quantile(0.50), name="y_hat")
            truth_df = _ts_to_df(y_true_ts, name="y_true")
            aligned = pred_df.join(truth_df, how="inner")
            if aligned.empty:
                print(f"Empty alignment for series {i}, window {j}. Pred: {pred_df.index}, Truth: {truth_df.index}")
                continue

            y_hat = aligned["y_hat"].to_numpy()
            y_true = aligned["y_true"].to_numpy()

            # Compute deterministic metrics
            rmse = float(np.sqrt(np.mean((y_hat - y_true) ** 2)))
            mae = float(np.mean(np.abs(y_hat - y_true)))
            mape = float(_safe_mape(y_true, y_hat))
            smape = float(_smape(y_true, y_hat))

            # Compute quantile metrics
            mql_50 = float(mql(y_true_ts, pred_ts, 0.50))
            mic_val = float(mic(y_true_ts, pred_ts, q_interval=(0.10, 0.90)))

            row = {
                "series_idx": i,
                "window_idx": j,
                "rmse": rmse,
                "mae": mae,
                "mape": mape,
                "smape": smape,
                "mql_0_50": mql_50,
                "mic_10_90": mic_val,
                "n_points": len(y_true),
            }
            rows.append(row)

    if not rows:
        print("No valid rows computed")
        return {
            "per_series_window": pd.DataFrame(columns=["series_idx", "window_idx", "rmse", "mae", "mape", "smape", "mql_0_50", "mic_10_90"]),
            "overall": {} if compute_global else None
        }

    df_metrics = pd.DataFrame(rows)

    result = {"per_series_window": df_metrics}
    if compute_global:
        overall = df_metrics[["rmse", "mae", "mape", "smape", "mql_0_50", "mic_10_90"]].mean()
        result["overall"] = overall

    return result

# ---- main: metrics for single warm-start/horizon predictions ----

def metrics_from_results(results, q_interval=(0.10, 0.90)):
    """
    Compute metrics for predictions vs truth.

    Parameters
    ----------
    results : list of dicts
        Output from predict_with_truth()
    q_interval : tuple
        Interval for MIC (default 10%-90%)

    Returns
    -------
    dict with:
        - per_series : DataFrame of metrics per well
        - overall_mean : simple average of metrics
        - overall_weighted : weighted average by #points
    """
    rows = []

    for r in results:
        i     = r["series_idx"]
        pred  = r["pred"]
        truth = r["truth"]

        # Align truth and pred
        pred_df  = _ts_to_df(pred.quantile(0.50), "y_hat")
        truth_df = _ts_to_df(truth, "y_true").reindex(pred_df.index).dropna()
        if truth_df.empty:
            continue  # nothing to score for this well

        pred_df = pred_df.loc[truth_df.index]
        y_hat = pred_df["y_hat"].to_numpy()
        y_true = truth_df["y_true"].to_numpy()
        n_pts = len(y_true)

        # deterministic metrics
        rmse   = float(np.sqrt(np.mean((y_hat - y_true) ** 2)))
        mae    = float(np.mean(np.abs(y_hat - y_true)))
        mape_  = _safe_mape(y_true, y_hat)
        smape_ = _smape(y_true, y_hat)

        # probabilistic metrics
        pred_overlap  = pred.slice(truth.start_time(), truth.end_time())
        truth_overlap = truth.slice(truth.start_time(), truth.end_time())

        mql_050   = float(mql(truth_overlap, pred_overlap, 0.50))
        mic_10_90 = float(mic(truth_overlap, pred_overlap, q_interval=q_interval))

        rows.append({
            "series_idx": i,
            "n_points": n_pts,
            "rmse": rmse,
            "mae": mae,
            "mape": mape_,
            "smape": smape_,
            "mql_0_50": mql_050,
            "mic_10_90": mic_10_90,
        })

    # --- new guard ---
    if not rows:
        empty = pd.DataFrame(columns=[
            "series_idx","n_points","rmse","mae","mape","smape","mql_0_50","mic_10_90"
        ])
        return {
            "per_series": empty,
            "overall_mean": pd.Series(dtype=float),
            "overall_weighted": pd.Series(dtype=float),
        }

    per_series = pd.DataFrame(rows).sort_values("series_idx").reset_index(drop=True)

    out = {"per_series": per_series}
    # simple mean
    out["overall_mean"] = per_series[["rmse","mae","mape","smape","mql_0_50","mic_10_90"]].mean()

    # weighted mean
    w = per_series["n_points"].to_numpy()
    w = w / w.sum()
    out["overall_weighted"] = pd.Series({
        "rmse_w": np.sum(w * per_series["rmse"].to_numpy()),
        "mae_w": np.sum(w * per_series["mae"].to_numpy()),
        "mape_w": np.sum(w * per_series["mape"].to_numpy()),
        "smape_w": np.sum(w * per_series["smape"].to_numpy()),
        "mql_0_50_w": np.sum(w * per_series["mql_0_50"].to_numpy()),
        "mic_10_90_w": np.sum(w * per_series["mic_10_90"].to_numpy()),
    })
    return out

