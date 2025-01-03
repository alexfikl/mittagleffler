# SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib
from collections.abc import Callable

import numpy as np
import rich.logging

from pymittagleffler import mittag_leffler
from pymittagleffler.fallback import mittag_leffler_garrappa

log = logging.getLogger(pathlib.Path(__file__).stem)
log.setLevel(logging.ERROR)
log.addHandler(rich.logging.RichHandler())

# {{{ timeit wrapper


def timeit(
    stmt: str | Callable[[], object],
    *,
    repeat: int = 16,
    number: int = 1,
) -> tuple[float, float, float]:
    import timeit as _timeit

    r = np.array(_timeit.repeat(stmt=stmt, repeat=repeat + 1, number=number))
    return np.min(r), np.mean(r), np.std(r, ddof=1)


# }}}


# {{{ benchmark

alpha = np.linspace(0.1, 5.0, 32)
beta = 1.25

arg = np.pi / 2
r = np.linspace(0.0, 12.0, 256)
z = r * np.exp(1j * arg)

result_rust = np.empty((alpha.size, 3))
result_python = np.empty((alpha.size, 3))

for i in range(alpha.size):
    result_rust[i] = timeit(lambda: mittag_leffler(z, float(alpha[i]), beta))
    result_python[i] = timeit(lambda: mittag_leffler_garrappa(z, float(alpha[i]), beta))

filename = pathlib.Path(__file__).with_suffix(".npz")
np.savez(filename, alpha=alpha, result_rust=result_rust, result_python=result_python)
log.info("Saved results in '%s'", filename)

# }}}

# {{{ plot

try:
    import matplotlib.pyplot as mp
except ImportError as exc:
    print("'matplotlib' package is not available for plotting")
    raise SystemExit(0) from exc

from pymittagleffler import _set_recommended_matplotlib  # noqa: E402,RUF100

_set_recommended_matplotlib()
fig = mp.figure()
ax = fig.gca()

mean = result_rust[:, 1] * 1000
std = result_rust[:, 2] * 1000
line, = ax.plot(alpha, mean, label="Rust")
ax.fill_between(alpha, mean + std, mean - std, alpha=0.2, color=line.get_color())

mean = result_python[:, 1] * 1000
std = result_python[:, 2] * 1000
line, = ax.plot(alpha, mean, label="Python")
ax.fill_between(alpha, mean + std, mean - std, alpha=0.2, color=line.get_color())

ax.set_xlabel(r"$\alpha$")
ax.set_ylabel("$Time~(ms)$")
ax.legend()

filename = pathlib.Path(__file__).with_suffix(".png")
fig.savefig(filename)
log.info("Saved plot in '%s'", filename)

# }}}
