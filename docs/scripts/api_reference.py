"""Writes the Python API reference page from the installed package's signatures.

Run before `astro build` (the docs workflow does): python scripts/api_reference.py
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import miter

OUT = Path(__file__).resolve().parent.parent / "src" / "content" / "docs" / "reference" / "python.md"

METHODS = {
    "fit": "Train the model. `eval_set=[(X_val, y_val)]` enables early stopping "
    "(`early_stopping_rounds`); without it all `n_estimators` trees are built. "
    '`categorical_feature` takes indices or column names; `"auto"` uses pandas `category` columns.',
    "predict": "Predictions (regression: values; classification: class labels). `num_iteration` "
    "uses the first trees only (default: the best iteration).",
    "predict_proba": "Class probabilities, shape `(n, n_classes)`.",
    "save_model": "Write the model as miter JSON. Loading it gives bit-identical predictions.",
    "load_model": "Class method: read a model written by `save_model`. A classifier's labels come "
    "back as `0..K-1`; pickle keeps the original labels.",
    "get_params": "scikit-learn: the constructor parameters.",
    "set_params": "scikit-learn: set constructor parameters.",
    "score": "scikit-learn: R² (regressor) or accuracy (classifier).",
}

ATTRIBUTES = {
    "best_iteration_": "Number of trees kept (the best validation iteration with early stopping).",
    "feature_importances_": "Split counts per input feature (oblique splits count for each feature they use).",
    "feature_importances_gain_": "Total gain per input feature.",
    "n_features_in_": "Number of features seen in `fit`.",
    "feature_names_in_": "Column names seen in `fit` (DataFrame input only).",
    "classes_": "Class labels (classifier only).",
    "train_loss_": "Training loss after each iteration.",
    "val_loss_": "Validation loss after each iteration (with `eval_set`).",
}


def signature(name: str, obj) -> str:
    try:
        sig = str(inspect.signature(obj))
    except (TypeError, ValueError):
        sig = "(...)"
    sig = re.sub(r"(: |-> )'([^']*)'", r"\1\2", sig)
    sig = sig.replace("(self, ", "(").replace("(self)", "()")
    # One parameter per line for the long constructor signature.
    if len(sig) > 100:
        inner = sig[1 : sig.rindex(")")]
        tail = sig[sig.rindex(")") :]
        parts = re.split(r", (?![^\[]*\])", inner)
        sig = "(\n    " + ",\n    ".join(parts) + ",\n" + tail
    return f"```python\n{name}{sig}\n```"


def estimator(cls) -> list[str]:
    name = cls.__name__
    parts = [f"## `miter.{name}`", "", inspect.getdoc(cls) or "", "", signature(f"miter.{name}", cls), ""]
    parts += ["See [Parameters](../../guides/parameters/) for what each parameter does.", ""]
    for meth, text in METHODS.items():
        member = getattr(cls, meth, None)
        if member is None:
            continue
        parts += [f"### `{name}.{meth}`", "", signature(f"{name}.{meth}", member), "", text, ""]
    parts += ["### Fitted attributes", "", "| attribute | meaning |", "|---|---|"]
    for attr, text in ATTRIBUTES.items():
        if attr == "classes_" and name != "MiterClassifier":
            continue
        parts.append(f"| `{attr}` | {text} |")
    return parts + [""]


def main() -> None:
    parts = [
        "---",
        "title: Python API",
        "description: The public classes and functions of the miter package.",
        "---",
        "",
        f"Generated from miter {miter.__version__} by `docs/scripts/api_reference.py`.",
        "",
    ]
    parts += estimator(miter.MiterRegressor)
    parts += estimator(miter.MiterClassifier)
    parts += [
        "## Build information",
        "",
        "| function | returns |",
        "|---|---|",
        "| `miter.has_openmp()` | whether the library was built with OpenMP |",
        "| `miter.openmp_max_threads()` | the number of threads `n_jobs=-1` uses |",
        "| `miter.lignum_linked()` | whether predictions run through lignum |",
        "| `miter.__version__` | the package version |",
        "",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(parts).rstrip() + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
