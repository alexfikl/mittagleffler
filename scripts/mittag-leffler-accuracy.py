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
    for i in range(z.size):
        result[i] = ml.evaluate(z[i], alpha, beta)

    return result


# {{{ evaluate

alpha = 0.5
beta = 1

arg = np.pi / 2
r = np.linspace(0.0, 12.0, 256)
z = r * np.exp(1j * arg)

# 1. Reference result
result_ref = np.exp(z**2) * erfc(-z)

# 2. Garrappa algorithm
ml = GarrappaMittagLeffler()
result_garrappa = mittag_leffler(ml, z, alpha, beta)
error_garrappa = pointwise_error(result_ref, result_garrappa)
print(f"Error: Garrappa {np.max(error_garrappa):.8e}")

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

ax.semilogy(r, error_garrappa, label="Garrappa (2015)")
ax.axhline(ml.eps, color="k", ls="--")
ax.set_xlabel("$|z|$")
ax.set_ylabel("Error")
ax.legend()

filename = pathlib.Path(__file__).with_suffix(".png")
fig.savefig(filename)
print(f"Saved results in {filename}")

# }}}
