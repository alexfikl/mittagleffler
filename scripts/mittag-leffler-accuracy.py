# SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

r"""
This reproduces Figure 4 from [Garrappa2015]_. It evaluates the Mittag-Leffler
function

.. math::

    E_{\frac{1}{2}, 1}(z) = \exp(z^2) \erfc(-z)

In this regime, the series expansion will fail to converge for :math:`|z| > 5`
and the algorithm from Diethelm2005 also fails.

.. [Garrappa2015] R. Garrappa,
    *Numerical Evaluation of Two and Three Parameter Mittag-Leffler Functions*,
    SIAM Journal on Numerical Analysis, Vol. 53, pp. 1350--1369, 2015,
    `DOI <https://doi.org/10.1137/140971191>`__.
"""

from __future__ import annotations

import pathlib

import numpy as np
from scipy.special import erfc

from pymittagleffler import GarrappaMittagLeffler

Array = np.ndarray[tuple[int, ...], np.dtype[np.floating]]


def pointwise_error(ref: Array, a: Array) -> Array:
    # NOTE: this is the error used in Equation 4.1 from [Garrappa2015]_.
    return np.abs(a - ref) / (1 + np.abs(ref))  # type: ignore[no-any-return]


def mittag_leffler(
    ml: GarrappaMittagLeffler, z: Array, alpha: float, beta: float
) -> Array:
    result = np.empty_like(z)
    for i in np.ndindex(*z.shape):
        result[i] = ml.evaluate(z[i], alpha, beta)

    return result


# {{{ evaluate (for Figure 4 in [Garrappa2015])

alpha = 0.5
beta = 1

n = 256
arg = np.pi / 2
r = np.linspace(0.0, 12.0, n)
z = r * np.exp(1j * arg)

# 1. Reference result
result_garrappa_ref = np.exp(z**2) * erfc(-z)

# 2. Garrappa algorithm
ml = GarrappaMittagLeffler()
result_garrappa_num = mittag_leffler(ml, z, alpha, beta)
error_garrappa_num = pointwise_error(result_garrappa_ref, result_garrappa_num)
print(f"Error: Garrappa {np.max(error_garrappa_num):.8e}")

# }}}

# {{{ evaluate (for fancy plot)

alpha = 2
beta = 1

n = 2048
x = np.linspace(-3.0, 3.0, n)
y = np.linspace(-1.5, 1.5, n)
x, y = np.meshgrid(x, y)
z = x + 1j * y
del x, y

# 1. Reference result
result_box_ref = np.cos(z)

# 2. Garrappa algorithm
ml = GarrappaMittagLeffler()
result_box_num = mittag_leffler(ml, -(z**2), alpha, beta)
error_box_num = result_box_ref - result_box_num
print(f"Error: Garrappa {np.max(np.abs(error_box_num)):.8e}")

# }}}

# {{{ plot

try:
    import matplotlib.pyplot as mp
    from matplotlib.ticker import LogLocator
except ImportError as exc:
    print("'matplotlib' package is not available for plotting")
    raise SystemExit(0) from exc

from pymittagleffler import _set_recommended_matplotlib  # noqa: E402,RUF100

_set_recommended_matplotlib()
fig = mp.figure()
ax = fig.gca()

ax.semilogy(r, error_garrappa_num, label="Garrappa (2015)")
ax.axhline(ml.eps, color="k", ls="--")
ax.set_xlabel("$|z|$")
ax.set_ylabel("Error")
ax.legend()

filename = pathlib.Path(__file__).with_suffix(".png")
fig.savefig(filename)
print(f"Saved results in {filename}")

fig.clf()
fig.set_size_inches(8, 4)
ax = fig.gca()

cs = ax.contourf(
    z.real,
    z.imag,
    np.abs(error_box_num) + 1.0e-16,
    locator=LogLocator(),
    cmap=mp.colormaps["inferno"],
)
ax.set_axis_off()

filename = filename.with_stem(f"{filename.stem}-contour")
fig.savefig(filename)
print(f"Saved results in {filename}")

# }}}
