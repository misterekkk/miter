"""Configurations of the speed-up experiments (pre-registrations: results/exp_speedups_*_prereg.md).

Kept free of imports so that the TabArena runner (pandas 2) and the OpenML harness (pandas 3, its
own dataset cache) can both use it without touching each other's environment.
"""

from __future__ import annotations

from typing import Any

BASE: dict[str, Any] = {}  # the library defaults
VARIANTS: dict[str, dict[str, Any]] = {
    "BASE": BASE,
    "E1": {"oblique_min_distinct": 3},
    "E2": {"oblique_sample_rows": 4096},  # stress setting for the quality checks
    "E2_64K": {"oblique_sample_rows": 65536},  # the candidate default, for the speed check
}
