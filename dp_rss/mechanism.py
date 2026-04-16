"""
DP-RSS: Differentially Private Refined Sufficient Statistics (Algorithm 3).

Satisfies ε-differential privacy under the add/remove adjacency model (pure DP, δ=0).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from typing import Optional, Tuple


def dp_rss(
    x: ArrayLike,
    y: ArrayLike,
    epsilon: float,
    x_bounds: Tuple[float, float] = (0.0, 1.0),
    y_bounds: Tuple[float, float] = (0.0, 1.0),
) -> Optional[Tuple[float, float]]:
    """
    DP-RSS for simple linear regression (Algorithm 3 in the paper).

    Exploits multidimensional simplex transformations to construct multiple
    independent unbiased estimators of each sufficient statistic, then combines
    them via inverse-variance weighting for up to 4.8x variance reduction over
    independent privatisation — at zero additional privacy cost.

    Parameters
    ----------
    x : array-like, shape (n,)
        Independent variable values.
    y : array-like, shape (n,)
        Dependent variable values.
    epsilon : float
        Total privacy budget (ε > 0).  Split equally: ε₁ = ε₂ = ε/2.
    x_bounds : (x_min, x_max), default (0, 1)
        Known bounds on x.  Data is normalised to [0,1] internally.
    y_bounds : (y_min, y_max), default (0, 1)
        Known bounds on y.  Data is normalised to [0,1] internally.

    Returns
    -------
    (alpha_hat, beta_hat) : tuple of float
        Private estimates of slope and intercept on the *original* scale.
        Returns None if the noisy sample size or determinant is ≤ 0.

    Privacy guarantee
    -----------------
    ε-differential privacy under the add/remove model (pure DP).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    x_min, x_max = x_bounds
    y_min, y_max = y_bounds
    dx = x_max - x_min
    dy = y_max - y_min

    # Normalise to [0, 1]  (Appendix A)
    x_norm = (x - x_min) / dx if dx > 0 else np.zeros_like(x)
    y_norm = (y - y_min) / dy if dy > 0 else np.zeros_like(y)

    result = _dp_rss_unit(x_norm, y_norm, epsilon)
    if result is None:
        return None

    alpha_norm, beta_norm = result

    # Map back to original scale  (Appendix A)
    if dx == 0 or dy == 0:
        return (0.0, (y_min + y_max) / 2.0)

    alpha_hat = (dy / dx) * alpha_norm
    beta_hat = y_min + dy * (beta_norm - (x_min / dx) * alpha_norm)
    return (alpha_hat, beta_hat)


def _dp_rss_unit(
    x: np.ndarray,
    y: np.ndarray,
    epsilon: float,
) -> Optional[Tuple[float, float]]:
    """Core DP-RSS on data already in [0, 1]²."""
    scale = 2.0 / epsilon  # Laplace scale = Δ/ε₁ = 1/(ε/2)

    # ── Group 1: x-statistics  (Eq. 5) ──────────────────────────────────────
    #   x² + (x − x²) + (1 − x) = 1   →   ℓ₁-sensitivity = 1
    s_tilde_x2 = np.sum(x ** 2) + np.random.laplace(0, scale)
    s_tilde_x_minus_x2 = np.sum(x - x ** 2) + np.random.laplace(0, scale)
    s_tilde_1_minus_x = np.sum(1.0 - x) + np.random.laplace(0, scale)

    # ── Group 2: (x, y)-statistics  (Eq. 7) ─────────────────────────────────
    #   xy + (1−x)y + (1−y) = 1   →   ℓ₁-sensitivity = 1
    s_tilde_xy = np.sum(x * y) + np.random.laplace(0, scale)
    s_tilde_1_minus_x_y = np.sum((1.0 - x) * y) + np.random.laplace(0, scale)
    s_tilde_1_minus_y = np.sum(1.0 - y) + np.random.laplace(0, scale)

    # ── Noisy sample sizes  (Eq. 9–10) ──────────────────────────────────────
    nx_tilde = s_tilde_x2 + s_tilde_x_minus_x2 + s_tilde_1_minus_x
    ny_tilde = s_tilde_xy + s_tilde_1_minus_x_y + s_tilde_1_minus_y
    n_tilde = (nx_tilde + ny_tilde) / 2.0

    if n_tilde <= 0:
        return None

    # ── Refined estimators  (Theorems 12, 14, 16, 18) ───────────────────────
    #
    # S_x², S_xy:  σ₁²=8/ε², σ₂²=40/ε²  →  w₁=5/6, w₂=1/6
    # S_x,  S_y:   σ₁²=16/ε², σ₂²=32/ε²  →  w₁=2/3, w₂=1/3

    refined_Sx2 = (5.0 / 6.0) * s_tilde_x2 + (1.0 / 6.0) * (
        ny_tilde - s_tilde_x_minus_x2 - s_tilde_1_minus_x
    )
    refined_Sxy = (5.0 / 6.0) * s_tilde_xy + (1.0 / 6.0) * (
        nx_tilde - s_tilde_1_minus_x_y - s_tilde_1_minus_y
    )
    refined_Sx = (2.0 / 3.0) * (s_tilde_x2 + s_tilde_x_minus_x2) + (1.0 / 3.0) * (
        ny_tilde - s_tilde_1_minus_x
    )
    refined_Sy = (2.0 / 3.0) * (s_tilde_xy + s_tilde_1_minus_x_y) + (1.0 / 3.0) * (
        nx_tilde - s_tilde_1_minus_y
    )

    # ── Solve normal equations  (step 25) ────────────────────────────────────
    det = refined_Sx2 * n_tilde - refined_Sx ** 2
    if det <= 0:
        return None

    alpha = (n_tilde * refined_Sxy - refined_Sx * refined_Sy) / det
    beta = (refined_Sx2 * refined_Sy - refined_Sx * refined_Sxy) / det

    return (alpha, beta)
