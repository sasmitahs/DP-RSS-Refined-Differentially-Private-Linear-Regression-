"""
Synthetic data generation for evaluating DP-RSS.
"""

from __future__ import annotations

import numpy as np


def generate_dataset(
    n: int,
    alpha: float = 0.5,
    beta: float = 0.2,
    sigma: float = 0.1,
    x_min: float = 0.0,
    x_max: float = 1.0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data: y = αx + β + N(0, σ²), clipped to [0, 1].

    Parameters
    ----------
    n      : number of data points
    alpha  : true slope
    beta   : true intercept
    sigma  : noise standard deviation
    x_min, x_max : bounds for Uniform(x_min, x_max)
    seed   : random seed for reproducibility

    Returns
    -------
    (x, y) : arrays of shape (n,)
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(x_min, x_max, size=n)
    y = np.clip(alpha * x + beta + rng.normal(0, sigma, size=n), 0.0, 1.0)
    return x, y
