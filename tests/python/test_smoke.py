import numpy as np

import miter
from miter import _core


def test_version_and_hello() -> None:
    assert miter.__version__.startswith("0.1.0")
    assert _core.cpp_version() == miter.__version__
    assert miter.hello() == f"miter {miter.__version__}"


def test_lignum_linked_and_deterministic() -> None:
    assert miter.lignum_linked() is True
    h1 = _core.lignum_smoke_hash("miter")
    h2 = _core.lignum_smoke_hash("miter")
    h3 = _core.lignum_smoke_hash("other")
    assert h1 == h2
    assert h1 != h3


def test_openmp_threads() -> None:
    assert isinstance(miter.has_openmp(), bool)
    assert miter.openmp_max_threads() >= 1


def test_ndarray_zero_copy_shapes() -> None:
    c_arr = np.zeros((4, 3), dtype=np.float64, order="C")
    f_arr = np.zeros((5, 2), dtype=np.float64, order="F")
    assert _core.array_shape(c_arr) == (4, 3)
    assert _core.array_shape(f_arr) == (5, 2)
