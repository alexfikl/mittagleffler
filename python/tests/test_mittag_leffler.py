# SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.linalg as la
import pytest


def test_mittag_leffler_scalar() -> None:
    from pymittagleffler import mittag_leffler

    alpha = beta = 1.0

    types: tuple[type, ...] = (bool, int, float, np.float32, np.float64)
    for cls in types:
        z = cls(1)
        _ = mittag_leffler(z, alpha, beta)
        print(f"({type(z)}, {type(_)}): {_} {np.exp(z + 0j)}")

    types = (complex, np.complex64, np.complex128)
    for cls in types:
        z = cls(1.0, 1.0)  # type: ignore[call-arg]
        _ = mittag_leffler(z, alpha, beta)
        print(f"({type(z)}, {type(_)}): {_} {np.exp(z + 0j)}")

    z = "1.0"
    with pytest.raises(TypeError):
        _ = mittag_leffler(z, alpha, beta)


@pytest.mark.parametrize(
    "etype", [np.int32, np.float32, np.float64, np.complex64, np.complex128]
)
def test_mittag_leffler_vector(etype: Any) -> None:
    from pymittagleffler import mittag_leffler

    dtype = np.dtype(etype)
    rng = np.random.default_rng()
    if issubclass(etype, np.integer):
        z = rng.integers(0, 10, size=128, dtype=dtype)
    elif issubclass(etype, np.floating):
        z = rng.random(size=128, dtype=dtype)
    elif issubclass(etype, np.complexfloating):
        rtype = dtype.type(1.0).real.dtype
        z = rng.random(size=128, dtype=rtype) + 1j * rng.random(size=128, dtype=rtype)
    else:
        raise TypeError(f"Unsupported dtype: {dtype}")

    alpha = beta = 1.0
    result = mittag_leffler(z, alpha, beta)
    ref = np.exp(z)

    assert la.norm(result - ref) < 1.0e-4 * la.norm(ref)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
