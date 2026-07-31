"""Shared helpers for UCI real-data TL-DVCM experiments."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from tl_dvcm import fit_transfer_dvcm, gaussian_kernel


def zscore_safe(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return s * 0.0
    return (s - s.mean()) / sd


def fit_linear_theta(X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> np.ndarray:
    model = LinearRegression()
    model.fit(X, y, sample_weight=sample_weight)
    return np.r_[model.intercept_, model.coef_]


def predict(theta: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.c_[np.ones(len(X)), X] @ theta


def fit_scalar_ridge(
    X: np.ndarray,
    y: np.ndarray,
    theta_anchor: np.ndarray,
    lam: float,
    theta_start: np.ndarray,
) -> np.ndarray:
    design = np.c_[np.ones(len(X)), X]

    def obj(theta: np.ndarray) -> Tuple[float, np.ndarray]:
        resid = design @ theta - y
        diff = theta - theta_anchor
        loss = 0.5 * float(np.mean(resid**2)) + 0.5 * lam * float(diff @ diff)
        grad = design.T @ resid / len(y) + lam * diff
        return loss, grad

    opt = minimize(
        lambda t: obj(t)[0],
        theta_start,
        jac=lambda t: obj(t)[1],
        method="L-BFGS-B",
        options={"maxiter": 1000},
    )
    if not opt.success:
        raise RuntimeError(f"scalar ridge failed: {opt.message}")
    return opt.x


def tune_scalar_ridge(
    X: np.ndarray,
    y: np.ndarray,
    theta_anchor: np.ndarray,
    theta_start: np.ndarray,
    lambda_grid: Iterable[float],
    seed: int,
) -> tuple[np.ndarray, float]:
    idx = np.arange(len(y))
    train_idx, val_idx = train_test_split(idx, train_size=0.7, random_state=seed)
    best_loss = np.inf
    best_theta = theta_start
    best_lam = float(next(iter(lambda_grid)))
    for lam in lambda_grid:
        theta = fit_scalar_ridge(X[train_idx], y[train_idx], theta_anchor, float(lam), theta_start)
        loss = mean_squared_error(y[val_idx], predict(theta, X[val_idx]))
        if loss < best_loss:
            best_loss = loss
            best_theta = theta
            best_lam = float(lam)
    return best_theta, best_lam


def evaluate_linear_methods_cv_bandwidth(
    df_model: pd.DataFrame,
    feature_cols: list[str],
    *,
    domain_label: str,
    bandwidth_grid: Iterable[float],
    pooled_feature_cols: list[str] | None = None,
    min_target_size: int = 30,
    n_repeats: int = 5,
    degree: int = 1,
    bias_order: int | None = None,
    bias_bandwidth: float | None = None,
    seed0: int = 20240519,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate methods with target-validation bandwidth selection for TL.

    For each target domain and repeat, target rows are split into train/test as
    before. The target train rows are split again into CV-train/CV-validation.
    The bandwidth minimizing TL validation MSE is selected, then local methods
    are refit on the full target train rows and evaluated on held-out test rows.
    """
    U_all = df_model["U"].to_numpy(float)
    X_all = df_model[feature_cols].to_numpy(float)
    if pooled_feature_cols:
        X_pooled_all = df_model[pooled_feature_cols].to_numpy(float)
    else:
        X_pooled_all = np.column_stack([X_all, U_all])
    y_all = df_model["Y"].to_numpy(float)
    u0s = np.sort(df_model["U"].unique())
    bandwidth_grid = [float(h) for h in bandwidth_grid]
    lambda_grid = np.logspace(-5, 1, 7)
    records = []

    for u0 in u0s:
        target_mask = np.isclose(U_all, u0)
        source_mask = ~target_mask
        X0, y0, U0 = X_all[target_mask], y_all[target_mask], U_all[target_mask]
        Xs, ys, Us = X_all[source_mask], y_all[source_mask], U_all[source_mask]
        if len(y0) < min_target_size:
            continue

        domain_value = float(df_model.loc[target_mask, domain_label].iloc[0])
        for rep in range(n_repeats):
            seed = seed0 + rep
            idx = np.arange(len(y0))
            train_idx, test_idx = train_test_split(idx, train_size=2 / 3, random_state=seed)
            cv_train_idx, cv_val_idx = train_test_split(train_idx, train_size=0.75, random_state=seed + 500)

            best_bandwidth = bandwidth_grid[0]
            best_val_loss = np.inf
            for candidate_bandwidth in bandwidth_grid:
                U_cv = np.r_[Us, U0[cv_train_idx]]
                X_cv = np.vstack([Xs, X0[cv_train_idx]])
                y_cv = np.r_[ys, y0[cv_train_idx]]
                target_cv = np.r_[np.zeros(len(ys), dtype=bool), np.ones(len(cv_train_idx), dtype=bool)]
                tl_cv = fit_transfer_dvcm(
                    U_cv,
                    X_cv,
                    y_cv,
                    np.array([u0]),
                    target_mask=target_cv,
                    degree=degree,
                    bandwidth=candidate_bandwidth,
                    family="linear",
                    split_target=True,
                    random_state=seed + 2000,
                    delta=1.0,
                    bias_order=bias_order,
                    bias_bandwidth=bias_bandwidth,
                )
                val_pred = predict(tl_cv.theta_tl, X0[cv_val_idx])
                val_loss = mean_squared_error(y0[cv_val_idx], val_pred)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_bandwidth = candidate_bandwidth

            train1_idx, train2_idx = train_test_split(train_idx, train_size=1 / 2, random_state=seed + 1000)
            theta_lr = fit_linear_theta(X0[train_idx], y0[train_idx])

            U_dvcm = np.r_[Us, U0[train_idx]]
            X_dvcm = np.vstack([Xs, X0[train_idx]])
            y_dvcm = np.r_[ys, y0[train_idx]]
            theta_dvcm = fit_linear_theta(
                X_dvcm,
                y_dvcm,
                sample_weight=gaussian_kernel((U_dvcm - u0) / best_bandwidth),
            )

            pooled_train = np.r_[np.flatnonzero(source_mask), np.flatnonzero(target_mask)[train_idx]]
            theta_pooled = fit_linear_theta(X_pooled_all[pooled_train], y_all[pooled_train])

            U_tl = np.r_[Us, U0[train1_idx], U0[train2_idx]]
            X_tl = np.vstack([Xs, X0[train1_idx], X0[train2_idx]])
            y_tl = np.r_[ys, y0[train1_idx], y0[train2_idx]]
            target_tl = np.r_[
                np.zeros(len(ys), dtype=bool),
                np.ones(len(train1_idx) + len(train2_idx), dtype=bool),
            ]
            tl_res = fit_transfer_dvcm(
                U_tl,
                X_tl,
                y_tl,
                np.array([u0]),
                target_mask=target_tl,
                degree=degree,
                bandwidth=best_bandwidth,
                family="linear",
                split_target=True,
                random_state=seed + 2000,
                delta=1.0,
                bias_order=bias_order,
                bias_bandwidth=bias_bandwidth,
            )
            theta_scalar, best_lam = tune_scalar_ridge(
                X0[train2_idx],
                y0[train2_idx],
                tl_res.theta_pilot,
                theta_lr,
                lambda_grid,
                seed + 3000,
            )

            X_test = X0[test_idx]
            y_test = y0[test_idx]
            theta_by_method = {
                "LR": (theta_lr, X_test),
                "Pooled": (theta_pooled, X_pooled_all[np.flatnonzero(target_mask)[test_idx]]),
                "DVCM": (theta_dvcm, X_test),
                "ScalarRidge": (theta_scalar, X_test),
                "TL": (tl_res.theta_tl, X_test),
            }
            row = {
                "bandwidth": best_bandwidth,
                "selected_bandwidth": best_bandwidth,
                "validation_loss": best_val_loss,
                "u0": u0,
                domain_label: domain_value,
                "rep": rep,
                "m_target": len(y0),
                "ScalarRidge_lambda": best_lam,
            }
            for method, (theta, X_eval) in theta_by_method.items():
                pred = predict(theta, X_eval)
                row[method] = mean_squared_error(y_test, pred)
                row[f"{method}_mae"] = mean_absolute_error(y_test, pred)
                row[f"{method}_r2"] = r2_score(y_test, pred)
            records.append(row)

    rep_df = pd.DataFrame.from_records(records)
    if rep_df.empty:
        raise RuntimeError(f"No domains had at least {min_target_size} observations")

    methods = ["LR", "Pooled", "DVCM", "ScalarRidge", "TL"]
    metric_cols = methods + [f"{m}_{suffix}" for m in methods for suffix in ["mae", "r2"]]
    summary = (
        rep_df.groupby(["u0", domain_label, "m_target"], as_index=False)
        .agg(
            **{col: (col, "mean") for col in metric_cols},
            **{f"{col}_std": (col, "std") for col in metric_cols},
            selected_bandwidth=("selected_bandwidth", "mean"),
            selected_bandwidth_std=("selected_bandwidth", "std"),
            validation_loss=("validation_loss", "mean"),
            ScalarRidge_lambda=("ScalarRidge_lambda", "mean"),
            n_success=("rep", "count"),
        )
        .sort_values("u0")
    )
    summary["bandwidth"] = "cv"
    return summary, rep_df
