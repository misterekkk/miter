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
    "DEF2": {},  # the current default (E2 adopted); TabArena baseline for E3
    "E3": {"efb": True},  # on top of the current default (E2 adopted)
    # Exploratory ablation of the method's components (relative to the current default).
    "AB_NO_OBLIQUE": {"m_mult": 0},
    "AB_LEAFWISE": {"grow_policy": "leafwise"},
    "AB_M2": {"m_mult": 2},
    "AB_DEPTH4": {"max_depth": 4},
    "AB_K2": {"k_proj": 2},
    "AB_CTR": {"max_ctr_complexity": "auto", "ctr_n_permutations": 4},
    "AB_QAUTO": {"quantizer": "auto"},
    # Ablation of the training-time optimisations (models identical to the default except E2).
    "SP_NONE": {"oblique_sample_rows": 0, "efb": False},
    "SP_E2_ONLY": {"efb": False},
    "SP_EFB_ONLY": {"oblique_sample_rows": 0},
}
