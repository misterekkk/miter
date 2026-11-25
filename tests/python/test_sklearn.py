"""scikit-learn compatibility: estimator checks, pickling, clone, pipelines and model selection."""

import pickle

import numpy as np
import pandas as pd
import pytest
from miter import MiterClassifier, MiterRegressor

sklearn = pytest.importorskip("sklearn")
from sklearn.base import clone  # noqa: E402
from sklearn.exceptions import NotFittedError  # noqa: E402
from sklearn.model_selection import GridSearchCV, cross_val_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.utils.estimator_checks import parametrize_with_checks  # noqa: E402


@parametrize_with_checks(
    [MiterRegressor(n_estimators=20, learning_rate=0.3), MiterClassifier(n_estimators=20, learning_rate=0.3)]
)
def test_sklearn_estimator_checks(estimator, check) -> None:  # type: ignore[no-untyped-def]
    check(estimator)


def _same_bits(a: np.ndarray, b: np.ndarray) -> None:
    np.testing.assert_array_equal(np.asarray(a).view(np.uint64), np.asarray(b).view(np.uint64))


def _data(n: int = 800, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    X[rng.random(X.shape) < 0.05] = np.nan
    y = np.nan_to_num(X[:, 0]) * np.nan_to_num(X[:, 1]) + np.sin(np.nan_to_num(X[:, 2]))
    return X, y


def test_pickle_regressor_is_bitwise() -> None:
    X, y = _data()
    est = MiterRegressor(n_estimators=30, early_stopping_rounds=None, n_jobs=1).fit(X, y)
    loaded = pickle.loads(pickle.dumps(est))
    _same_bits(loaded.predict(X), est.predict(X))
    _same_bits(loaded.predict(X, num_iteration=5), est.predict(X, num_iteration=5))
    assert loaded.get_params() == est.get_params()
    np.testing.assert_array_equal(loaded.feature_importances_, est.feature_importances_)


def test_pickle_classifier_keeps_labels_and_proba() -> None:
    X, y = _data(seed=1)
    labels = np.where(y > np.median(y), "yes", "no")
    est = MiterClassifier(n_estimators=20, early_stopping_rounds=None, n_jobs=1).fit(X, labels)
    loaded = pickle.loads(pickle.dumps(est))
    _same_bits(loaded.predict_proba(X), est.predict_proba(X))
    np.testing.assert_array_equal(loaded.predict(X), est.predict(X))
    assert set(loaded.predict(X)) <= {"yes", "no"}


def test_pickle_categorical_dataframe() -> None:
    rng = np.random.default_rng(3)
    n = 1500
    codes = rng.integers(0, 8, size=n)
    df = pd.DataFrame({"c": pd.Categorical(codes.astype(str)), "x": rng.normal(size=n)})
    y = (codes % 3).astype(float) + df["x"].to_numpy()
    est = MiterRegressor(n_estimators=20, min_data_per_group=20, early_stopping_rounds=None, n_jobs=1).fit(df, y)
    loaded = pickle.loads(pickle.dumps(est))
    _same_bits(loaded.predict(df), est.predict(df))


def test_unfitted_estimator_raises_not_fitted() -> None:
    with pytest.raises(NotFittedError):
        MiterRegressor().predict(np.zeros((2, 2)))
    with pytest.raises(NotFittedError):
        pickle.loads(pickle.dumps(MiterClassifier())).predict_proba(np.zeros((2, 2)))


def test_clone_pipeline_and_grid_search() -> None:
    X, y = _data(n=600, seed=4)
    est = MiterRegressor(n_estimators=15, early_stopping_rounds=None, n_jobs=1)
    cloned = clone(est).set_params(num_leaves=7)
    assert cloned.num_leaves == 7 and est.num_leaves == 31
    pipe = make_pipeline(StandardScaler(), est).fit(X, y)
    assert pipe.predict(X).shape == (len(X),)
    grid = GridSearchCV(est, {"num_leaves": [7, 15]}, cv=2).fit(X, y)
    assert grid.best_params_["num_leaves"] in (7, 15)


def test_cross_val_score_in_parallel_processes() -> None:
    X, y = _data(n=600, seed=5)
    est = MiterClassifier(n_estimators=10, early_stopping_rounds=None, n_jobs=1)
    scores = cross_val_score(est, X, (y > 0).astype(int), cv=3, n_jobs=2)  # pickles the estimator
    assert scores.shape == (3,) and np.all(scores > 0.5)


def test_sample_weight_and_input_errors() -> None:
    X, y = _data(n=200, seed=6)
    est = MiterRegressor(n_estimators=5, n_jobs=1)
    with pytest.raises(ValueError, match="non-zero"):
        est.fit(X, y, sample_weight=np.zeros(len(y)))
    with pytest.raises(ValueError, match="sample_weight.shape"):
        est.fit(X, y, sample_weight=np.ones(len(y) + 1))
    with pytest.raises(ValueError, match="Reshape your data"):
        est.fit(X[:, 0], y)
    with pytest.raises(TypeError, match="Sparse"):
        from scipy import sparse

        est.fit(sparse.csr_matrix(np.nan_to_num(X)), y)
    est.fit(X, y)
    with pytest.raises(ValueError, match="is expecting 5 features"):
        est.predict(X[:, :3])
