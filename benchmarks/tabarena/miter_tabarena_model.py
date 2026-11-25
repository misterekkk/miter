"""miter (C++ core) as an AutoGluon/TabArena model, default configuration, regression only.

Differences to the prototype wrapper t2_miter_k1.py (histree, numba):
  * engine is the C++ MiterRegressor (python/miter), not histree.gb.oblique_cat;
  * categoricals: pandas `category` columns (AutoGluon output) go to miter natively (K1 cuts), object columns are
    cast to category; NaN is handled natively by miter (no median imputation, no missing-indicator columns);
  * m = m_mult * ceil(sqrt(p_num)) is computed inside the C++ core (m_mult=2), k_proj=0 (random k in 2..3 per spec);
  * time_limit from TabArena is forwarded to miter's own time_limit (per bagging child); None = unlimited.
Default config: lr 0.05, 31 leaves, min_data_in_leaf 20, lambda_l2 1.0, early stopping 50 on bagging X_val, max 10000 trees.
"""

import tempfile

from autogluon.common.features.types import R_CATEGORY, R_FLOAT, R_INT, R_OBJECT
from autogluon.core.models import AbstractModel


class MiterCppModel(AbstractModel):
    ag_key = "MITER_CPP"
    ag_name = "MiterCpp"

    def _set_default_params(self):
        for k, v in dict(
            n_estimators=10000,
            learning_rate=0.05,
            num_leaves=31,
            min_data_in_leaf=20,
            lambda_l2=1.0,
            early_stopping_rounds=50,
            m_mult=2,
            k_proj=0,
            preset="default",
        ).items():
            self._set_default_param_value(k, v)

    @classmethod
    def supported_problem_types(cls):
        return ["regression"]

    def _get_default_auxiliary_params(self) -> dict:
        p = super()._get_default_auxiliary_params()
        p.update(
            dict(
                valid_raw_types=[R_INT, R_FLOAT, R_CATEGORY, R_OBJECT],
                ignored_type_group_special=[],
            )
        )
        return p

    def _prep(self, X):
        obj = [c for c in X.columns if str(X[c].dtype) == "object"]
        if obj:
            X = X.copy()
            for c in obj:
                X[c] = X[c].astype("category")
        return X

    def _fit(self, X, y, X_val=None, y_val=None, time_limit=None, num_cpus=1, **kwargs):
        from miter import MiterRegressor

        P = self._get_model_params()
        X = self._prep(self.preprocess(X, is_train=True))
        self.model = MiterRegressor(
            n_estimators=P["n_estimators"],
            learning_rate=P["learning_rate"],
            num_leaves=P["num_leaves"],
            min_data_in_leaf=P["min_data_in_leaf"],
            lambda_l2=P["lambda_l2"],
            m_mult=P["m_mult"],
            k_proj=P["k_proj"],
            preset=P["preset"],
            early_stopping_rounds=P["early_stopping_rounds"]
            if X_val is not None
            else None,
            time_limit=time_limit,
            n_jobs=1,
            random_state=int(self.random_seed or 0),
        )
        kw = {}
        if X_val is not None:
            kw["eval_set"] = [
                (self._prep(self.preprocess(X_val)), y_val.to_numpy(dtype="float64"))
            ]
        self.model.fit(X, y.to_numpy(dtype="float64"), **kw)
        self._best_iter = int(self.model.best_iteration_)
        self._blob = self._dump(self.model)

    # miter's C++ Model is not picklable: persist via its JSON save/load, keep the live estimator out of the pickle.
    @staticmethod
    def _dump(est):
        with tempfile.TemporaryDirectory() as d:
            est.save_model(d + "/m.json")
            return open(d + "/m.json", "rb").read()

    def __getstate__(self):
        st = dict(self.__dict__)
        st["model"] = None
        return st

    def _est(self):
        if getattr(self, "model", None) is None:
            from miter import MiterRegressor

            with tempfile.TemporaryDirectory() as d:
                open(d + "/m.json", "wb").write(self._blob)
                self.model = MiterRegressor.load_model(d + "/m.json")
        return self.model

    def _predict_proba(self, X, **kwargs):
        return self._est().predict(self._prep(self.preprocess(X, **kwargs)))

    def _get_default_resources(self):
        return 1, 0

    def get_minimum_resources(self, is_gpu_available=False):
        return dict(num_cpus=1, num_gpus=0)

    def _more_tags(self):
        return {"valid_oof": True}

    @classmethod
    def config_generator(cls):
        from tabarena.utils.config_utils import ConfigGenerator

        return ConfigGenerator(
            model_cls=cls, manual_configs=[{}], search_space={}, name="MiterCpp"
        )


class MiterCppAccModel(MiterCppModel):
    """Same wrapper, `accurate` preset (m = 4 * m0); nothing else changed."""

    ag_key = "MITER_CPP_ACC"
    ag_name = "MiterCppAcc"

    def _set_default_params(self):
        self._set_default_param_value("preset", "accurate")
        super()._set_default_params()

    @classmethod
    def config_generator(cls):
        from tabarena.utils.config_utils import ConfigGenerator

        return ConfigGenerator(
            model_cls=cls, manual_configs=[{}], search_space={}, name="MiterCppAcc"
        )
