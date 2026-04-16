# DP-RSS: Differentially Private Refined Sufficient Statistics for Linear Regression

Reference implementation for:

> **Refined Differentially Private Simple Linear Regression via Extension of a Free Lunch Result**
> Sasmita Harini S and Anshoo Tandon
> *SeQureDB Workshop @ ACM SIGMOD/PODS 2026, Bengaluru, India*

## Overview

DP-RSS exploits multidimensional simplex transformations on bounded data to construct **multiple independent unbiased estimators** of each sufficient statistic needed for ordinary least squares regression. These estimators are combined via inverse-variance weighting, achieving **up to 4.8× variance reduction** over independent privatisation — at zero additional privacy cost.

The mechanism operates under the **add/remove differential privacy** model with pure (ε, 0)-DP guarantees.

### Variance Reduction (Table 1)

| Statistic | Standard Variance | DP-RSS Variance | Improvement |
|-----------|-------------------|-----------------|-------------|
| n         | 16/ε²             | 12/ε²           | 1.33×       |
| S(x²)     | 32/ε²             | 20/(3ε²)        | **4.80×**   |
| S(xy)     | 32/ε²             | 20/(3ε²)        | **4.80×**   |
| S(x)      | 32/ε²             | 32/(3ε²)        | 3.00×       |
| S(y)      | 32/ε²             | 32/(3ε²)        | 3.00×       |

## Repository Structure

```
dp-rss/
├── dp_rss/
│   ├── __init__.py      # Package entry point
│   ├── mechanism.py     # DP-RSS implementation (Algorithm 3)
│   ├── metrics.py       # L1 / L2 error (Eqs. 1, 2, 11, 12)
│   └── data.py          # Synthetic data generation
├── examples/
│   ├── quickstart.py    # Minimal usage example
│   └── reproduce_paper.py  # Reproduce Figure 1
├── tests/
│   └── test_dp_rss.py   # Unit tests
├── pyproject.toml
├── LICENSE
└── README.md
```

## Installation

```bash
git clone https://github.com/sasmitahs/dp-rss.git
cd dp-rss
pip install -e .
```

Only requires `numpy>=1.20`. For experiments and tests:

```bash
pip install -e ".[dev]"   # adds pytest + matplotlib
```

## Quick Start

```python
from dp_rss import dp_rss, generate_dataset, l2_error_exact

# Generate data: y = 0.5x + 0.2 + noise, clipped to [0, 1]
x, y = generate_dataset(n=5000, alpha=0.5, beta=0.2, sigma=0.1, seed=42)

# Run DP-RSS with ε = 1.0
alpha_hat, beta_hat = dp_rss(x, y, epsilon=1.0)
print(f"α̂ = {alpha_hat:.4f},  β̂ = {beta_hat:.4f}")

# Evaluate
mse = l2_error_exact(0.5, 0.2, alpha_hat, beta_hat)
print(f"MSE = {mse:.6f}")
```

### General Bounded Data

For data in arbitrary rectangles (see Appendix A in the paper):

```python
result = dp_rss(x, y, epsilon=1.0, x_bounds=(10, 50), y_bounds=(0, 100))
```

Data is normalised to [0, 1]² internally; returned parameters are on the original scale.

## API

### `dp_rss(x, y, epsilon, x_bounds=(0,1), y_bounds=(0,1))`

| Parameter  | Description |
|------------|-------------|
| `x, y`     | Data arrays of shape (n,) |
| `epsilon`  | Total privacy budget ε > 0 (split ε/2 per group) |
| `x_bounds` | Known (x_min, x_max) |
| `y_bounds` | Known (y_min, y_max) |
| **Returns** | `(alpha_hat, beta_hat)` or `None` if degenerate |

### Metrics

```python
from dp_rss import l1_error, l2_error_exact

l1_error(true_slope, true_intercept, pred_slope, pred_intercept)       # Eq. 11
l2_error_exact(true_slope, true_intercept, pred_slope, pred_intercept) # Eq. 12
```

## Reproducing Paper Results

```bash
# Full run (1000 iterations, ~30 min)
python examples/reproduce_paper.py

# Quick check (50 iterations, ~2 min)
python examples/reproduce_paper.py --iterations 50

# Single setup
python examples/reproduce_paper.py --setup 1
```

## Tests

```bash
pytest tests/ -v
```

## Citation

```bibtex
@inproceedings{harini2026dprss,
    title     = {Refined Differentially Private Simple Linear Regression
                 via Extension of a Free Lunch Result},
    author    = {Sasmita Harini S and Anshoo Tandon},
    booktitle = {Proceedings of ACM SIGMOD/PODS Workshop on Secure and
                 Private Data Management (SeQureDB)},
    year      = {2026},
    publisher = {ACM},
    address   = {Bengaluru, India}
}
```

## License

MIT
