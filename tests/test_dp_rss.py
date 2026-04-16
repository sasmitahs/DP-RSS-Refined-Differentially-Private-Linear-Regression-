"""
Tests for dp_rss.   Run with:  pytest tests/ -v
"""

import numpy as np
import pytest
from dp_rss.mechanism import dp_rss
from dp_rss.metrics import l1_error, l2_error, l2_error_exact
from dp_rss.data import generate_dataset


# ── Data ─────────────────────────────────────────────────────────────────────

class TestData:
    def test_shape_and_bounds(self):
        x, y = generate_dataset(10_000, alpha=0.9, beta=0.5, sigma=0.5, seed=0)
        assert x.shape == (10_000,)
        assert 0.0 <= y.min() and y.max() <= 1.0

    def test_reproducibility(self):
        assert np.array_equal(*[generate_dataset(50, seed=42)[0] for _ in range(2)])


# ── Metrics ──────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_zero_error(self):
        assert l2_error_exact(1, 0, 1, 0) == pytest.approx(0.0)

    def test_exact_vs_numerical(self):
        assert l2_error_exact(0.5, 0.2, 0.6, 0.15) == pytest.approx(
            l2_error(0.5, 0.2, 0.6, 0.15, n_points=100_000), abs=1e-5
        )


# ── DP-RSS mechanism ─────────────────────────────────────────────────────────

class TestDPRSS:
    def test_returns_tuple(self):
        np.random.seed(1)
        x, y = generate_dataset(5000, seed=0)
        r = dp_rss(x, y, epsilon=2.0)
        assert r is not None and len(r) == 2

    def test_accuracy_high_epsilon(self):
        np.random.seed(42)
        errors = []
        for _ in range(50):
            x, y = generate_dataset(5000, alpha=0.5, beta=0.2, sigma=0.05)
            r = dp_rss(x, y, epsilon=10.0)
            if r:
                errors.append(l2_error_exact(0.5, 0.2, *r))
        assert np.mean(errors) < 0.001

    def test_general_bounds(self):
        np.random.seed(7)
        x = np.random.uniform(10, 20, size=3000)
        y = 0.5 * x + 3.0 + np.random.normal(0, 0.5, size=3000)
        r = dp_rss(x, y, epsilon=5.0, x_bounds=(10, 20), y_bounds=(min(y), max(y)))
        assert r is not None
        assert abs(r[0] - 0.5) < 0.5
