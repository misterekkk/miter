"""miter: gradient-boosted decision trees with random oblique splits."""

from miter._core import (
    has_openmp,
    hello,
    lignum_linked,
    openmp_max_threads,
)
from miter.booster import MiterClassifier, MiterRegressor
from miter.data import Dataset, TransformedData

__version__ = "0.1.0"

__all__ = [
    "Dataset",
    "MiterClassifier",
    "MiterRegressor",
    "TransformedData",
    "__version__",
    "has_openmp",
    "hello",
    "lignum_linked",
    "openmp_max_threads",
]
