"""miter: gradient-boosted decision trees with random oblique splits."""

from miter._core import (
    has_openmp,
    hello,
    lignum_linked,
    openmp_max_threads,
)

__version__ = "0.1.0.dev0"

__all__ = [
    "__version__",
    "has_openmp",
    "hello",
    "lignum_linked",
    "openmp_max_threads",
]
