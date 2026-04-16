"""
DP-RSS: Differentially Private Refined Sufficient Statistics for Linear Regression.

Reference implementation for:

    Sasmita Harini S and Anshoo Tandon. "Refined Differentially Private Simple
    Linear Regression via Extension of a Free Lunch Result." SeQureDB @ ACM
    SIGMOD/PODS 2026.
"""

from dp_rss.mechanism import dp_rss
from dp_rss.metrics import l1_error, l2_error, l2_error_exact
from dp_rss.data import generate_dataset

__version__ = "1.0.0"
__all__ = ["dp_rss", "l1_error", "l2_error", "l2_error_exact", "generate_dataset"]
