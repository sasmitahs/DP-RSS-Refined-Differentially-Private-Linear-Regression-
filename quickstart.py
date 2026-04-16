#!/usr/bin/env python3
"""Quick-start: run DP-RSS on synthetic data."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from dp_rss import dp_rss, generate_dataset, l2_error_exact

TRUE_ALPHA, TRUE_BETA = 0.5, 0.2

x, y = generate_dataset(n=5000, alpha=TRUE_ALPHA, beta=TRUE_BETA, sigma=0.1, seed=42)

for eps in [0.1, 0.5, 1.0, 5.0]:
    np.random.seed(0)
    result = dp_rss(x, y, epsilon=eps)
    if result is None:
        print(f"ε={eps:<5}  → degenerate (returned None)")
    else:
        a, b = result
        mse = l2_error_exact(TRUE_ALPHA, TRUE_BETA, a, b)
        print(f"ε={eps:<5}  → α̂={a:+.4f}  β̂={b:+.4f}  MSE={mse:.6f}")
