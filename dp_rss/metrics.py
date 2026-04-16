"""
Evaluation metrics for differentially private linear regression.

L1 (MAE) and L2 (MSE) error between the true and estimated regression lines,
integrated over the input domain — Equations (1), (2), (11), (12) in the paper.
"""

from __future__ import annotations

import numpy as np


def l1_error(
    true_slope: float,
    true_intercept: float,
    pred_slope: float,
    pred_intercept: float,
    x_lower: float = 0.0,
    x_upper: float = 1.0,
    n_points: int = 1000,
) -> float:
    """Approximate L1 error (MAE) via numerical integration (Eq. 11)."""
    x = np.linspace(x_lower, x_upper, n_points)
    return float(np.mean(np.abs((true_slope - pred_slope) * x + (true_intercept - pred_intercept))))


def l2_error(
    true_slope: float,
    true_intercept: float,
    pred_slope: float,
    pred_intercept: float,
    x_lower: float = 0.0,
    x_upper: float = 1.0,
    n_points: int = 1000,
) -> float:
    """Approximate L2 error (MSE) via numerical integration."""
    x = np.linspace(x_lower, x_upper, n_points)
    return float(np.mean(((true_slope - pred_slope) * x + (true_intercept - pred_intercept)) ** 2))


def l2_error_exact(
    true_slope: float,
    true_intercept: float,
    pred_slope: float,
    pred_intercept: float,
    x_lower: float = 0.0,
    x_upper: float = 1.0,
) -> float:
    """
    Exact L2 error via closed-form integration (Eq. 12).

    MSE = A²/3 + AB + B²   where A = α−α̂, B = β−β̂  (for [0,1]).
    """
    A = true_slope - pred_slope
    B = true_intercept - pred_intercept
    length = x_upper - x_lower
    if length == 0:
        return 0.0
    F = lambda t: A ** 2 * (t ** 3 / 3.0) + A * B * t ** 2 + B ** 2 * t
    return float((F(x_upper) - F(x_lower)) / length)
