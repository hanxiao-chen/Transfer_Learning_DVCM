"""Transfer learning estimators for domain-varying coefficient models.

This module implements the two-step procedure.

1. local-polynomial DVCM/GDVCM pilot estimation around a target domain ``u0``;
2. target-domain fine tuning with an adaptive ridge penalty toward the pilot.

The implementation is intentionally notebook-friendly: import this file from
the existing simulation or real-data notebooks and call ``fit_transfer_dvcm``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


Array = np.ndarray
Kernel = Callable[[Array], Array]


def gaussian_kernel(t: Array) -> Array:
    """Standard Gaussian kernel applied to scalar distances."""
    return np.exp(-0.5 * np.asarray(t) ** 2) / np.sqrt(2.0 * np.pi)


def uniform_kernel(t: Array) -> Array:
    """Uniform kernel on [-1, 1]."""
    return 0.5 * (np.abs(np.asarray(t)) <= 1.0)


def add_intercept(X: Array) -> Array:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    return np.hstack((np.ones((X.shape[0], 1)), X))


def polynomial_features(U_centered_scaled: Array, degree: int) -> Array:
    """Local-polynomial map [1, u, u^2/2!, ...] for one or more U dimensions.

    For multivariate U this uses all monomials up to total degree via sklearn
    when available in the repo environment. Univariate U uses a small local
    implementation to keep the common path transparent.
    """
    U_centered_scaled = np.asarray(U_centered_scaled, dtype=float)
    if U_centered_scaled.ndim == 1:
        U_centered_scaled = U_centered_scaled[:, None]
    if U_centered_scaled.shape[1] == 1:
        t = U_centered_scaled[:, 0]
        return np.column_stack([t**j / factorial(j) for j in range(degree + 1)])

    from sklearn.preprocessing import PolynomialFeatures

    poly = PolynomialFeatures(degree=degree, include_bias=True)
    return poly.fit_transform(U_centered_scaled)


def local_design(
    U: Array,
    X: Array,
    u0: Array,
    degree: int,
    bandwidth: float,
    *,
    intercept: bool = True,
    kernel: Kernel = gaussian_kernel,
) -> Tuple[Array, Array, Array, Array]:
    """Build the Kronecker local-polynomial design and kernel weights."""
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")

    U = np.asarray(U, dtype=float)
    if U.ndim == 1:
        U = U[:, None]
    u0 = np.asarray(u0, dtype=float).reshape(1, -1)
    X_design = add_intercept(X) if intercept else np.asarray(X, dtype=float)

    scaled = (U - u0) / bandwidth
    U_poly = polynomial_features(scaled, degree)
    Z = np.einsum("ij,ik->ijk", U_poly, X_design).reshape(U.shape[0], -1)

    distances = np.linalg.norm(scaled, axis=1)
    weights = kernel(distances)
    return Z, weights, X_design, U_poly


def _family_parts(family: str) -> Dict[str, Callable[[Array, Array], Array]]:
    family = family.lower()
    if family == "linear":
        return {
            "loss": lambda eta, y: 0.5 * (eta - y) ** 2,
            "score": lambda eta, y: eta - y,
            "curvature": lambda eta, y: np.ones_like(eta),
            "mean": lambda eta: eta,
        }
    if family == "logistic":
        return {
            "loss": lambda eta, y: np.logaddexp(0.0, eta) - y * eta,
            "score": lambda eta, y: expit(eta) - y,
            "curvature": lambda eta, y: expit(eta) * (1.0 - expit(eta)),
            "mean": expit,
        }
    if family == "poisson":
        return {
            "loss": lambda eta, y: np.exp(np.clip(eta, -30, 30)) - y * eta,
            "score": lambda eta, y: np.exp(np.clip(eta, -30, 30)) - y,
            "curvature": lambda eta, y: np.exp(np.clip(eta, -30, 30)),
            "mean": lambda eta: np.exp(np.clip(eta, -30, 30)),
        }
    raise ValueError("family must be one of: linear, logistic, poisson")


def _solve_weighted_glm(
    design: Array,
    y: Array,
    weights: Optional[Array],
    family: str,
    *,
    init: Optional[Array] = None,
    ridge: float = 1e-10,
    maxiter: int = 1000,
) -> Array:
    """Fit weighted canonical GLM by minimizing average weighted loss."""
    design = np.asarray(design, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n, p = design.shape
    weights = np.ones(n) if weights is None else np.asarray(weights, dtype=float).ravel()
    weights = np.maximum(weights, 0.0)
    if not np.any(weights > 0):
        raise ValueError("all local kernel weights are zero; increase bandwidth")

    parts = _family_parts(family)
    if init is None:
        init = np.zeros(p)

    if family == "linear":
        sw = np.sqrt(weights)
        lhs = (design * sw[:, None]).T @ (design * sw[:, None]) + ridge * np.eye(p)
        rhs = (design * weights[:, None]).T @ y
        return np.linalg.pinv(lhs) @ rhs

    scale = max(float(np.sum(weights)), 1.0)

    def objective(beta: Array) -> Tuple[float, Array]:
        eta = design @ beta
        loss = np.sum(weights * parts["loss"](eta, y)) / scale
        loss += 0.5 * ridge * float(beta @ beta)
        grad = design.T @ (weights * parts["score"](eta, y)) / scale + ridge * beta
        return float(loss), grad

    res = minimize(
        lambda b: objective(b)[0],
        np.asarray(init, dtype=float),
        jac=lambda b: objective(b)[1],
        method="L-BFGS-B",
        options={"maxiter": maxiter},
    )
    if not res.success:
        raise RuntimeError(f"{family} optimization failed: {res.message}")
    return res.x


def fit_local_dvcm(
    U: Array,
    X: Array,
    y: Array,
    u0: Array,
    *,
    degree: int,
    bandwidth: float,
    family: str = "linear",
    intercept: bool = True,
    kernel: Kernel = gaussian_kernel,
) -> Tuple[Array, Array, Array, Array]:
    """Fit the Step-I local DVCM/GDVCM pilot.

    Returns ``theta_hat, alpha_hat, Z, weights``. ``theta_hat`` is the first
    coefficient block, i.e. the estimate of theta(u0).
    """
    Z, weights, X_design, _ = local_design(
        U, X, u0, degree, bandwidth, intercept=intercept, kernel=kernel
    )
    alpha = _solve_weighted_glm(Z, y, weights, family)
    p = X_design.shape[1]
    return alpha[:p], alpha, Z, weights


def fit_target_glm(
    X: Array,
    y: Array,
    *,
    family: str = "linear",
    intercept: bool = True,
) -> Array:
    design = add_intercept(X) if intercept else np.asarray(X, dtype=float)
    return _solve_weighted_glm(design, y, None, family)


def fit_ridge_finetune(
    X: Array,
    y: Array,
    theta0: Array,
    Q: Array,
    *,
    family: str = "linear",
    intercept: bool = True,
) -> Array:
    """Step-II target-domain ridge fine tuning toward ``theta0``."""
    design = add_intercept(X) if intercept else np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    theta0 = np.asarray(theta0, dtype=float).ravel()
    Q = np.asarray(Q, dtype=float)
    parts = _family_parts(family)

    if family == "linear":
        lhs = design.T @ design / len(y) + Q
        rhs = design.T @ y / len(y) + Q @ theta0
        return np.linalg.pinv(lhs) @ rhs

    def objective(beta: Array) -> Tuple[float, Array]:
        eta = design @ beta
        diff = beta - theta0
        loss = np.mean(parts["loss"](eta, y)) + 0.5 * float(diff @ Q @ diff)
        grad = design.T @ parts["score"](eta, y) / len(y) + Q @ diff
        return float(loss), grad

    res = minimize(
        lambda b: objective(b)[0],
        theta0,
        jac=lambda b: objective(b)[1],
        method="L-BFGS-B",
        options={"maxiter": 1000},
    )
    if not res.success:
        raise RuntimeError(f"ridge fine-tuning failed: {res.message}")
    return res.x


def sandwich_variance(
    Z: Array,
    y: Array,
    alpha: Array,
    weights: Array,
    family: str,
    theta_dim: int,
    *,
    ridge: float = 1e-8,
) -> Array:
    """Finite-sample sandwich covariance for the selected theta(u0) block."""
    parts = _family_parts(family)
    eta = Z @ alpha
    score = parts["score"](eta, y)
    curvature = parts["curvature"](eta, y)
    H = Z.T @ ((weights * curvature)[:, None] * Z)
    B = Z.T @ (((weights * score) ** 2)[:, None] * Z)
    H_inv = np.linalg.pinv(H + ridge * np.eye(H.shape[0]))
    cov_alpha = H_inv @ B @ H_inv
    return cov_alpha[:theta_dim, :theta_dim]


def estimate_scale(
    X: Array,
    y: Array,
    theta: Array,
    *,
    family: str = "linear",
    intercept: bool = True,
) -> float:
    """Estimate target scale/noise variance for the adaptive penalty."""
    design = add_intercept(X) if intercept else np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    eta = design @ theta
    mu = _family_parts(family)["mean"](eta)
    if family == "linear":
        return float(np.mean((y - mu) ** 2))
    if family == "logistic":
        # Canonical Bernoulli has fixed dispersion 1; Pearson residuals can be
        # unstable near probabilities 0/1, so we use the standard GLM scale.
        return 1.0
    denom = np.maximum(mu, 1e-8)
    return float(np.mean((y - mu) ** 2 / denom))


def estimate_bias_outer(
    U: Array,
    X: Array,
    y: Array,
    u0: Array,
    *,
    base_degree: int,
    bandwidth: float,
    family: str,
    intercept: bool = True,
    kernel: Kernel = gaussian_kernel,
    bias_order: Optional[int] = None,
    bias_bandwidth: Optional[float] = None,
) -> Array:
    """Plug-in leading-bias outer product for univariate U.

    Set ``bias_order=None`` to use a zero bias estimate, which is often a
    pragmatic default with undersmoothing.
    """
    if bias_order is None:
        p = add_intercept(X).shape[1] if intercept else np.asarray(X).shape[1]
        return np.zeros((p, p))
    if bias_order <= base_degree:
        raise ValueError("bias_order should exceed base_degree")
    bias_bandwidth = bandwidth if bias_bandwidth is None else float(bias_bandwidth)

    U_arr = np.asarray(U, dtype=float)
    if U_arr.ndim != 1 and U_arr.shape[1] != 1:
        raise NotImplementedError("plug-in bias estimate is implemented for univariate U")
    U_vec = U_arr.ravel()
    u0_scalar = float(np.asarray(u0).ravel()[0])
    t = (U_vec - u0_scalar) / bias_bandwidth
    w = kernel(np.abs(t))
    zeta = float(np.sum((t**bias_order) * w) / max(np.sum(w), 1e-12))

    _, alpha_high, _, _ = fit_local_dvcm(
        U,
        X,
        y,
        u0,
        degree=bias_order,
        bandwidth=bias_bandwidth,
        family=family,
        intercept=intercept,
        kernel=kernel,
    )
    p = add_intercept(X).shape[1] if intercept else np.asarray(X).shape[1]
    derivative = alpha_high[bias_order * p : (bias_order + 1) * p] / (bias_bandwidth**bias_order)
    bias = zeta * derivative * (bandwidth**bias_order) / factorial(bias_order)
    return np.outer(bias, bias)


def estimate_penalty_q(
    U: Array,
    X: Array,
    y: Array,
    u0: Array,
    alpha_pilot: Array,
    weights: Array,
    Z: Array,
    X_target: Array,
    y_target: Array,
    theta_target: Array,
    *,
    degree: int,
    bandwidth: float,
    family: str,
    intercept: bool = True,
    kernel: Kernel = gaussian_kernel,
    delta: float = 1.0,
    bias_order: Optional[int] = None,
    bias_bandwidth: Optional[float] = None,
    ridge: float = 1e-8,
) -> Tuple[Array, Dict[str, Array]]:
    """Estimate the adaptive inverse-MSE penalty matrix Q."""
    theta_dim = add_intercept(X).shape[1] if intercept else np.asarray(X).shape[1]
    var = sandwich_variance(Z, y, alpha_pilot, weights, family, theta_dim, ridge=ridge)
    bias_outer = estimate_bias_outer(
        U,
        X,
        y,
        u0,
        base_degree=degree,
        bandwidth=bandwidth,
        family=family,
        intercept=intercept,
        kernel=kernel,
        bias_order=bias_order,
        bias_bandwidth=bias_bandwidth,
    )
    mse = 0.5 * (var + var.T) + bias_outer + ridge * np.eye(theta_dim)
    scale = estimate_scale(X_target, y_target, theta_target, family=family, intercept=intercept)
    Q = delta * scale / len(y_target) * np.linalg.pinv(mse)
    Q = 0.5 * (Q + Q.T)
    return Q, {
        "variance": var,
        "bias_outer": bias_outer,
        "mse": mse,
        "scale": np.array(scale),
        "bias_bandwidth": np.array(bandwidth if bias_bandwidth is None else bias_bandwidth),
    }


@dataclass
class TransferDVCMResult:
    theta_tl: Array
    theta_pilot: Array
    theta_target: Array
    Q: Array
    alpha_pilot: Array
    target_finetune_mask: Array
    target_pilot_mask: Array
    diagnostics: Dict[str, Array]


def fit_transfer_dvcm(
    U: Array,
    X: Array,
    y: Array,
    u0: Array,
    *,
    target_mask: Optional[Array] = None,
    degree: int = 1,
    bandwidth: float = 0.2,
    family: str = "linear",
    intercept: bool = True,
    kernel: Kernel = gaussian_kernel,
    split_target: bool = True,
    random_state: Optional[int] = 0,
    delta: float = 1.0,
    bias_order: Optional[int] = None,
    bias_bandwidth: Optional[float] = None,
) -> TransferDVCMResult:
    """Fit the adaptive transfer-learning DVCM estimator.

    Parameters
    ----------
    U, X, y:
        Observation-level domain identifiers, covariates, and responses.
    u0:
        Target domain identifier.
    target_mask:
        Boolean mask for target observations. If omitted, rows with ``U == u0``
        are treated as target observations.
    split_target:
        If true, target rows are split: half for the pilot and half for
        fine-tuning, matching the paper's theoretical recipe.
    bias_order:
        Optional integer order for plug-in bias estimation. Leave as ``None``
        for undersmoothed/no-bias penalty estimation.
    bias_bandwidth:
        Optional bandwidth used only for plug-in bias estimation. If omitted,
        the main local-polynomial bandwidth is reused.
    """
    U = np.asarray(U, dtype=float)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if U.ndim == 1:
        U_for_compare = U
    else:
        U_for_compare = np.linalg.norm(U - np.asarray(u0).reshape(1, -1), axis=1)

    if target_mask is None:
        if U.ndim == 1:
            target_mask = np.isclose(U_for_compare, float(np.asarray(u0).ravel()[0]))
        else:
            target_mask = np.isclose(U_for_compare, 0.0)
    target_mask = np.asarray(target_mask, dtype=bool)
    if not np.any(target_mask):
        raise ValueError("target_mask selects no target observations")

    target_idx = np.flatnonzero(target_mask)
    if split_target and len(target_idx) >= 2:
        rng = np.random.default_rng(random_state)
        shuffled = target_idx.copy()
        rng.shuffle(shuffled)
        half = len(shuffled) // 2
        target_pilot_idx = shuffled[:half]
        target_finetune_idx = shuffled[half:]
    else:
        target_pilot_idx = target_idx
        target_finetune_idx = target_idx

    pilot_mask = ~target_mask
    pilot_mask[target_pilot_idx] = True
    finetune_mask = np.zeros_like(target_mask)
    finetune_mask[target_finetune_idx] = True

    theta_pilot, alpha_pilot, Z, weights = fit_local_dvcm(
        U[pilot_mask],
        X[pilot_mask],
        y[pilot_mask],
        u0,
        degree=degree,
        bandwidth=bandwidth,
        family=family,
        intercept=intercept,
        kernel=kernel,
    )
    theta_target = fit_target_glm(
        X[finetune_mask],
        y[finetune_mask],
        family=family,
        intercept=intercept,
    )
    Q, diagnostics = estimate_penalty_q(
        U[pilot_mask],
        X[pilot_mask],
        y[pilot_mask],
        u0,
        alpha_pilot,
        weights,
        Z,
        X[finetune_mask],
        y[finetune_mask],
        theta_target,
        degree=degree,
        bandwidth=bandwidth,
        family=family,
        intercept=intercept,
        kernel=kernel,
        delta=delta,
        bias_order=bias_order,
        bias_bandwidth=bias_bandwidth,
    )
    theta_tl = fit_ridge_finetune(
        X[finetune_mask],
        y[finetune_mask],
        theta_pilot,
        Q,
        family=family,
        intercept=intercept,
    )

    target_pilot_mask = np.zeros_like(target_mask)
    target_pilot_mask[target_pilot_idx] = True
    return TransferDVCMResult(
        theta_tl=theta_tl,
        theta_pilot=theta_pilot,
        theta_target=theta_target,
        Q=Q,
        alpha_pilot=alpha_pilot,
        target_finetune_mask=finetune_mask,
        target_pilot_mask=target_pilot_mask,
        diagnostics=diagnostics,
    )
