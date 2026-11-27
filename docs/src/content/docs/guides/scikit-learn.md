---
title: scikit-learn
description: miter estimators in pipelines, grid searches and cross-validation.
---

With scikit-learn installed (`pip install "miter[sklearn]"`), `MiterRegressor` and
`MiterClassifier` are regular scikit-learn estimators:

- **Compatibility:** they pass `sklearn.utils.estimator_checks.check_estimator`.
- **Tools:** they work with `clone`, `Pipeline`, `GridSearchCV`, `cross_val_score(n_jobs=...)` and
  the rest of scikit-learn.
- **Without scikit-learn:** the estimators work the same, with `get_params` and `set_params`
  provided by miter itself.

```python
from sklearn.model_selection import GridSearchCV, cross_val_score
from miter import MiterClassifier

search = GridSearchCV(
    MiterClassifier(n_estimators=2000),
    {"grow_policy": ["leafwise", "symmetric"], "learning_rate": [0.03, 0.1]},
    cv=5,
)
search.fit(X, y)

scores = cross_val_score(MiterClassifier(), X, y, cv=5, n_jobs=5)
```

Early stopping needs an `eval_set`, which scikit-learn's tools do not pass. Inside a grid search,
either set `n_estimators` to a fixed budget, or use `time_limit`.

Input validation follows scikit-learn's conventions:
- sparse, complex, 1-D or empty input raises with scikit-learn's messages;
- predicting before `fit` raises `NotFittedError`;
- a different number of features at prediction time raises `ValueError`.
