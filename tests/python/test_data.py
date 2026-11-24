import miter
import numpy as np
import pytest


def _reference_q16r(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pure NumPy reference implementing docs/IMPLEMENTATION_PLAN.md Section 2.2."""
    n, p = X.shape
    u_out = np.empty((n, p), dtype=np.uint16)
    for j in range(p):
        col = X[:, j].astype(np.float64)
        finite = col[np.isfinite(col)]
        if finite.size == 0:
            u_out[:, j] = np.where(np.isnan(col), 65535, 0).astype(np.uint16)
            continue
        lo = float(np.quantile(finite, 0.0001))
        hi = float(np.quantile(finite, 0.9999))
        if hi == lo:
            lo = float(np.min(finite))
            hi = float(np.max(finite))
        if hi <= lo:
            u_out[:, j] = np.where(np.isnan(col), 65535, 0).astype(np.uint16)
            continue
        scale = 65534.0 / (hi - lo)
        for i in range(n):
            x = float(col[i])
            if np.isnan(x):
                u_out[i, j] = 65535
            else:
                v = (x - lo) * scale
                if v <= 0.0:
                    u_out[i, j] = 0
                elif v >= 65534.0:
                    u_out[i, j] = 65534
                else:
                    r = np.floor(v + 0.5)
                    u_out[i, j] = np.uint16(min(max(int(r), 0), 65534))
    c_out = (u_out.astype(np.int32) - 32768).astype(np.int16)
    return u_out, c_out


def test_q16r_bitwise_parity_with_spec_reference() -> None:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(2500, 5))
    # Heavy tails / outliers, constant column, NaNs, infinities, fallback column
    X[0, 0] = -1e6
    X[1, 0] = 1e6
    X[:, 1] = 7.25  # constant
    X[10:25, 2] = np.nan
    X[3, 2] = -np.inf
    X[4, 2] = np.inf
    X[:, 3] = 0.0
    X[99, 3] = 12.0  # q(0.0001) == q(0.9999) fallback to min/max

    u_ref, c_ref = _reference_q16r(X)

    ds_c = miter.Dataset(np.ascontiguousarray(X, dtype=np.float64))
    ds_f = miter.Dataset(np.asfortranarray(X, dtype=np.float64))

    np.testing.assert_array_equal(ds_c.codes, c_ref)
    np.testing.assert_array_equal(ds_c.unsigned_codes(), u_ref)
    np.testing.assert_array_equal(ds_f.codes, c_ref)
    np.testing.assert_array_equal(ds_c.bins, ds_f.bins)
    assert ds_c.bins.flags.f_contiguous


def test_q16r_float32_and_transform_with_threshold_rows() -> None:
    rng = np.random.default_rng(7)
    X_train = rng.uniform(-5.0, 5.0, size=(600, 3)).astype(np.float32)
    ds = miter.Dataset(X_train, max_bin=32)

    X_test = rng.uniform(-10.0, 10.0, size=(50, 3)).astype(np.float32)
    X_test[0, 0] = np.nan
    tx = ds.transform(X_test)
    u_ref_test = _reference_q16r_with_params(X_test.astype(np.float64), *ds.quantizer_params())
    c_ref_test = (u_ref_test.astype(np.int32) - 32768).astype(np.int16)
    np.testing.assert_array_equal(tx.codes, c_ref_test)

    # Verify bin(u) <= b <=> u < edges[b] on every test row and feature
    for f in range(ds.n_features):
        edges = ds.bin_edges(f)
        for i in range(tx.n_rows):
            u = int(tx.unsigned_codes()[i, f])
            b_actual = int(tx.bins[i, f])
            if u == 65535:
                assert b_actual == len(edges) + 1
            else:
                expected_bin = int(np.searchsorted(edges, u, side="right"))
                assert b_actual == expected_bin


def _reference_q16r_with_params(
    X: np.ndarray, lo: np.ndarray, scale: np.ndarray, is_const: np.ndarray
) -> np.ndarray:
    n, p = X.shape
    u_out = np.empty((n, p), dtype=np.uint16)
    for j in range(p):
        for i in range(n):
            x = float(X[i, j])
            if np.isnan(x):
                u_out[i, j] = 65535
            elif is_const[j]:
                u_out[i, j] = 0
            else:
                v = (x - float(lo[j])) * float(scale[j])
                if v <= 0.0:
                    u_out[i, j] = 0
                elif v >= 65534.0:
                    u_out[i, j] = 65534
                else:
                    u_out[i, j] = np.uint16(min(max(int(np.floor(v + 0.5)), 0), 65534))
    return u_out


def test_categorical_numpy_and_pandas() -> None:
    pd = pytest.importorskip("pandas")

    df_train = pd.DataFrame(
        {
            "num0": [1.0, 2.0, 3.0, 4.0, np.nan, 6.0],
            "cat_auto": pd.Series(["a", "b", "a", "a", None, "rare"], dtype="category"),
            "cat_str": ["x", "y", "x", "x", "y", None],
        }
    )
    ds = miter.Dataset(
        df_train,
        categorical_feature=["cat_auto", "cat_str"],
        min_data_per_group=2,
    )
    assert ds.feature_names == ["num0", "cat_auto", "cat_str"]
    assert ds.is_categorical == [False, True, True]
    assert ds.n_num_features == 1
    assert ds.n_cat_features == 2
    assert ds.codes.shape == (6, 1)
    assert ds.bins.shape == (6, 3)

    # cat_auto: "a" count 3 -> 0; "b" count 1 < 2 -> OTHER (1); MISSING -> 2
    assert ds.cat_categories(1) == ["a"]
    assert ds.cat_other_code(1) == 1
    assert ds.cat_missing_code(1) == 2
    np.testing.assert_array_equal(ds.bins[:, 1], np.array([0, 1, 0, 0, 2, 1], dtype=np.uint8))

    # cat_str: "x" count 3 -> 0, "y" count 2 -> 1, OTHER = 2, MISSING = 3
    assert ds.cat_categories(2) == ["x", "y"]
    np.testing.assert_array_equal(ds.bins[:, 2], np.array([0, 1, 0, 0, 1, 3], dtype=np.uint8))

    # Transform with unseen category and feature-name check
    df_test = pd.DataFrame(
        {
            "num0": [1.0, 6.0],
            "cat_auto": pd.Series(["unseen", "a"], dtype="category"),
            "cat_str": ["y", "unseen"],
        }
    )
    tx = ds.transform(df_test)
    np.testing.assert_array_equal(tx.bins[:, 1], np.array([1, 0], dtype=np.uint8))
    np.testing.assert_array_equal(tx.bins[:, 2], np.array([1, 2], dtype=np.uint8))

    df_bad_names = df_test.rename(columns={"cat_str": "wrong"})
    with pytest.raises(ValueError, match="Feature names mismatch"):
        ds.transform(df_bad_names)


def test_pandas_auto_category_detection() -> None:
    pd = pytest.importorskip("pandas")

    df = pd.DataFrame(
        {
            "num": [10.0, 20.0, 30.0, 40.0],
            "cat": pd.Series(["p", "p", "q", "q"], dtype="category"),
        }
    )
    ds = miter.Dataset(df, min_data_per_group=2)
    assert ds.is_categorical == [False, True]
    assert ds.cat_categories(1) == ["p", "q"]


def test_numpy_explicit_categorical_feature() -> None:
    X = np.array(
        [
            [1.0, 10.0],
            [2.0, 10.0],
            [3.0, 20.0],
            [4.0, np.nan],
        ],
        dtype=np.float64,
    )
    ds = miter.Dataset(X, categorical_feature=[1], min_data_per_group=2)
    assert ds.is_categorical == [False, True]
    assert ds.cat_categories(1) == ["10"]
    np.testing.assert_array_equal(ds.bins[:, 1], np.array([0, 0, 1, 2], dtype=np.uint8))
