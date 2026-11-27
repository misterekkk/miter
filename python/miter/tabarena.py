"""miter as an AutoGluon / TabArena model (regression, binary, and native softmax multiclass).

Requires ``autogluon.tabular`` (and ``tabarena`` plus ``ConfigSpace`` for the config generators);
neither is a dependency of miter. Usage inside a TabArena experiment::

    from miter.tabarena import MiterModel, generate_configs_miter_v12

The wrapper adds nothing to miter's own behaviour: categorical columns (pandas ``category``, as
AutoGluon produces them; ``object`` columns are cast to ``category``) go to miter's native K1
handling (and optional C++20 online ordered CTRs when configured), NaN stays NaN (miter handles
it natively), and TabArena's ``time_limit`` is forwarded to miter's ``time_limit``. Early stopping
(50 rounds) uses the bagging fold's validation set.

Multiclass uses miter's native C++20 softmax objective (vector leaves).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from autogluon.common.features.types import R_CATEGORY, R_FLOAT, R_INT, R_OBJECT
from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION
from autogluon.core.models import AbstractModel

_DEFAULTS: dict[str, Any] = {
    "n_estimators": 10000,
    "learning_rate": 0.05,
    "early_stopping_rounds": 50,
    "k_proj": 0,
    "preset": "default",
}

# Hyperparameters forwarded to the estimator when present in a config (others are rejected).
_FORWARDED = (
    "n_estimators",
    "learning_rate",
    "num_leaves",
    "max_depth",
    "grow_policy",
    "min_data_in_leaf",
    "min_sum_hessian_in_leaf",
    "lambda_l2",
    "min_gain_to_split",
    "max_bin",
    "max_cat_to_onehot",
    "min_data_per_group",
    "max_categories",
    "cat_smooth",
    "cat_l2",
    "max_cat_threshold",
    "m_mult",
    "k_proj",
    "preset",
    "feature_fraction",
    "bagging_fraction",
    "extra_trees",
    "oblique_sample_rows",
    "efb",
    "quantizer",
    "max_ctr_complexity",
    "ctr_prior_weight",
    "ctr_n_permutations",
    "ctr_max_borders",
    "ctr_model_size_reg",
)


class MiterModel(AbstractModel):  # type: ignore[misc]
    ag_key = "MITER"
    ag_name = "Miter"

    def _set_default_params(self) -> None:
        for key, value in _DEFAULTS.items():
            self._set_default_param_value(key, value)

    @classmethod
    def supported_problem_types(cls) -> list[str]:
        return [REGRESSION, BINARY, MULTICLASS]

    def _get_default_auxiliary_params(self) -> dict[str, Any]:
        params: dict[str, Any] = super()._get_default_auxiliary_params()
        params.update(
            {
                "valid_raw_types": [R_INT, R_FLOAT, R_CATEGORY, R_OBJECT],
                "ignored_type_group_special": [],
            }
        )
        return params

    @staticmethod
    def _prep(X: Any) -> Any:
        obj = [c for c in X.columns if str(X[c].dtype) == "object"]
        if obj:
            X = X.copy()
            for c in obj:
                X[c] = X[c].astype("category")
        return X

    def _estimator(self, params: dict[str, Any], *, has_val: bool, time_limit: float | None, n_jobs: int) -> Any:
        from miter import MiterClassifier, MiterRegressor

        unknown = sorted(set(params) - set(_FORWARDED) - {"early_stopping_rounds"})
        if unknown:
            raise ValueError(f"Unknown miter hyperparameters: {unknown}")
        kwargs = {k: params[k] for k in _FORWARDED if k in params}
        cls = MiterRegressor if self.problem_type == REGRESSION else MiterClassifier
        return cls(
            **kwargs,
            early_stopping_rounds=params.get("early_stopping_rounds") if has_val else None,
            time_limit=time_limit,
            n_jobs=n_jobs,
            random_state=int(self.random_seed or 0),
        )

    def _fit(
        self,
        X: Any,
        y: Any,
        X_val: Any = None,
        y_val: Any = None,
        time_limit: float | None = None,
        num_cpus: int = 1,
        **kwargs: Any,
    ) -> None:
        params = self._get_model_params()
        X = self._prep(self.preprocess(X, is_train=True))
        has_val = X_val is not None
        if self.problem_type == MULTICLASS:
            y_arr = np.asarray(y, dtype=np.int64)
            present = np.unique(y_arr)
            n_total_classes = int(getattr(self, "num_classes", None) or (int(np.max(y_arr)) + 1))
            if len(present) == n_total_classes and np.array_equal(present, np.arange(n_total_classes)):
                self._present_classes: Any = None
            else:
                self._present_classes = present
            self.model = self._estimator(params, has_val=has_val, time_limit=time_limit, n_jobs=int(num_cpus))
            fit_kwargs: dict[str, Any] = {}
            if has_val:
                Xv = self._prep(self.preprocess(X_val))
                yv = np.asarray(y_val, dtype=np.int64)
                if self._present_classes is not None:
                    mask = np.isin(yv, self._present_classes)
                    if np.any(mask):
                        Xv_sub = Xv.iloc[mask] if hasattr(Xv, "iloc") else Xv[mask]
                        fit_kwargs["eval_set"] = [(Xv_sub, yv[mask])]
                else:
                    fit_kwargs["eval_set"] = [(Xv, yv)]
            self.model.fit(X, y_arr, **fit_kwargs)
            self.params_trained["n_estimators"] = int(self.model.best_iteration_)
            return
        self.model = self._estimator(params, has_val=has_val, time_limit=time_limit, n_jobs=int(num_cpus))
        fit_kwargs = {}
        if has_val:
            fit_kwargs["eval_set"] = [(self._prep(self.preprocess(X_val)), np.asarray(y_val, dtype=np.float64))]
        self.model.fit(X, np.asarray(y, dtype=np.float64), **fit_kwargs)
        self.params_trained["n_estimators"] = int(self.model.best_iteration_)

    def _predict_proba(self, X: Any, **kwargs: Any) -> Any:
        X = self._prep(self.preprocess(X, **kwargs))
        if self.problem_type == MULTICLASS:
            proba = self.model.predict_proba(X)
            present = getattr(self, "_present_classes", None)
            if present is None:
                return proba
            n_total = int(getattr(self, "num_classes", len(self.model.classes_)))
            full = np.zeros((len(X), n_total), dtype=np.float64)
            full[:, present] = proba
            return full
        if self.problem_type == REGRESSION:
            return self.model.predict(X)
        return self.model.predict_proba(X)[:, 1]  # AutoGluon expects P(class 1) for binary

    def _get_default_resources(self) -> tuple[int, int]:
        return 1, 0

    def get_minimum_resources(self, is_gpu_available: bool = False) -> dict[str, int]:
        return {"num_cpus": 1, "num_gpus": 0}

    def _more_tags(self) -> dict[str, Any]:
        return {"valid_oof": True, "can_refit_full": True}


def generate_configs_miter(num_random_configs: int = 200, seed: int = 1234) -> list[dict[str, Any]]:
    """Random configurations in the style of TabArena's LightGBM / CatBoost search spaces.

    The first configuration TabArena evaluates is the default (``manual_configs=[{}]``); these are
    the additional random ones for "tuned" and "tuned + ensembled" results.
    """
    from ConfigSpace import (
        Categorical,
        ConfigurationSpace,
        EqualsCondition,
        Float,
        Integer,
    )

    gp = Categorical("grow_policy", ["leafwise", "symmetric"])
    nl = Integer("num_leaves", (4, 128), log=True)
    md = Integer("max_depth", (4, 8))
    space = ConfigurationSpace(
        space=[
            Float("learning_rate", (1e-2, 1e-1), log=True),
            gp,
            nl,
            md,
            Integer("min_data_in_leaf", (1, 64), log=True),
            Float("lambda_l2", (1e-3, 10.0), log=True),
            Categorical("m_mult", [1, 2, 4]),
            Float("feature_fraction", (0.5, 1.0)),
            Categorical("extra_trees", [False, True]),
            Integer("min_data_per_group", (2, 100), log=True),
            Float("cat_l2", (5e-3, 10.0), log=True),
            Float("cat_smooth", (1e-3, 100.0), log=True),
            Integer("max_cat_to_onehot", (4, 64), log=True),
        ],
        seed=seed,
    )
    space.add([EqualsCondition(nl, gp, "leafwise"), EqualsCondition(md, gp, "symmetric")])
    configs = space.sample_configuration(num_random_configs)
    if num_random_configs == 1:
        configs = [configs]
    out = []
    for config in configs:
        out.append({k: v.item() if isinstance(v, np.generic) else v for k, v in dict(config).items() if v is not None})
    return out


def generate_configs_miter_v12(num_random_configs: int = 19, seed: int = 1234) -> list[dict[str, Any]]:
    """v1.2 search space adding ``quantizer`` and native C++ online ordered CTRs."""
    from ConfigSpace import (
        Categorical,
        ConfigurationSpace,
        EqualsCondition,
        Float,
        Integer,
    )

    gp = Categorical("grow_policy", ["leafwise", "symmetric"])
    nl = Integer("num_leaves", (4, 128), log=True)
    md = Integer("max_depth", (4, 8))
    mcc = Categorical("max_ctr_complexity", [0, -1])
    ctr_p = Categorical("ctr_n_permutations", [1, 4])
    ctr_pw = Float("ctr_prior_weight", (0.5, 5.0), log=True)
    ctr_reg = Float("ctr_model_size_reg", (0.1, 2.0), log=True)
    space = ConfigurationSpace(
        space=[
            Float("learning_rate", (1e-2, 1e-1), log=True),
            gp,
            nl,
            md,
            Integer("min_data_in_leaf", (1, 64), log=True),
            Float("lambda_l2", (1e-3, 10.0), log=True),
            Categorical("m_mult", [1, 2, 4]),
            Float("feature_fraction", (0.5, 1.0)),
            Categorical("extra_trees", [False, True]),
            Categorical("quantizer", ["linear", "auto", "quantile"]),
            mcc,
            ctr_p,
            ctr_pw,
            ctr_reg,
            Integer("min_data_per_group", (2, 100), log=True),
            Float("cat_l2", (5e-3, 10.0), log=True),
            Float("cat_smooth", (1e-3, 100.0), log=True),
            Integer("max_cat_to_onehot", (4, 64), log=True),
        ],
        seed=seed,
    )
    space.add(
        [
            EqualsCondition(nl, gp, "leafwise"),
            EqualsCondition(md, gp, "symmetric"),
            EqualsCondition(mcc, gp, "symmetric"),
            EqualsCondition(ctr_p, mcc, -1),
            EqualsCondition(ctr_pw, mcc, -1),
            EqualsCondition(ctr_reg, mcc, -1),
        ]
    )
    configs = space.sample_configuration(num_random_configs)
    if num_random_configs == 1:
        configs = [configs]
    out: list[dict[str, Any]] = []
    for config in configs:
        d = {k: v.item() if isinstance(v, np.generic) else v for k, v in dict(config).items() if v is not None}
        if d.get("max_ctr_complexity") == -1:
            d["max_ctr_complexity"] = "auto"
        out.append(d)
    return out


def config_generator(num_random_configs: int = 200) -> Any:
    """TabArena config generator: the default configuration plus ``num_random_configs`` random ones."""
    from tabarena.utils.config_utils import CustomAGConfigGenerator

    return CustomAGConfigGenerator(
        model_cls=MiterModel,
        search_space_func=lambda num_random_configs=num_random_configs: generate_configs_miter(num_random_configs),
        manual_configs=[{}],
    )
