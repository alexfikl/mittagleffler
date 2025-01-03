# SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import Any, overload

import numpy as np

class GarrappaMittagLeffler:
    @property
    def eps(self) -> float: ...
    def __init__(self, *, eps: float | None = None) -> None: ...
    def evaluate(self, z: complex, alpha: float, beta: float) -> complex | None: ...

@overload
def mittag_leffler(
    z: int | float | complex | np.generic, alpha: float, beta: float
) -> complex: ...
@overload
def mittag_leffler(
    z: np.ndarray[tuple[int, ...], np.dtype[Any]], alpha: float, beta: float
) -> np.ndarray[tuple[int, ...], np.dtype[np.complex128]]: ...
