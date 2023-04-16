# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from mittagleffler import MittagLefflerAlgorithm, ml


def test_rust_module() -> None:
    r = ml(0, 0, 0, MittagLefflerAlgorithm.Series)
    assert isinstance(r, complex)
    assert abs(r) == 0
