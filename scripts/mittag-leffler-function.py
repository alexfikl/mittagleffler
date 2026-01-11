# SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

r"""
This evaluates the Mittag-Leffler function at some known parameters so we can
easily visualize it.
"""

from __future__ import annotations

import pathlib

# START_MITTAG_LEFFLER_EXAMPLE
import numpy as np

from pymittagleffler import mittag_leffler

alphas = (0.5, 1.0, 2.0)
betas = (1.0, 1.5, 2.0)

arg = np.pi / 2
r = np.linspace(0.0, 12.0, 256)
z = r * np.exp(1j * arg)

result = np.empty((len(alphas), z.size), dtype=z.dtype)
for i, (alpha, beta) in enumerate(zip(alphas, betas, strict=True)):
    result[i] = mittag_leffler(z, alpha, beta)
# END_MITTAG_LEFFLER_EXAMPLE

# {{{ plot

try:
    import matplotlib.pyplot as mp
except ImportError as exc:
    print("'matplotlib' package is not available for plotting")
    raise SystemExit(0) from exc

from pymittagleffler import _set_recommended_matplotlib  # noqa: E402,RUF100

_set_recommended_matplotlib()
fig = mp.figure(figsize=(14, 8))
ax = fig.gca()

for i, (alpha, beta) in enumerate(zip(alphas, betas, strict=True)):
    ax.plot(r, result[i].real, label=rf"$\alpha = {alpha:.2f}, \beta = {beta:.2f}$")

ax.set_xlabel("$|z|$")
ax.set_ylabel(r"$E_{\alpha, \beta}(z)$ (real)")
ax.legend()

filename = pathlib.Path(__file__).with_suffix(".svg")
filename = filename.with_stem(f"{filename.stem}-real")
fig.savefig(filename)
print(f"Saved results in {filename}")

fig.clf()
ax = fig.gca()

for i, (alpha, beta) in enumerate(zip(alphas, betas, strict=True)):
    ax.plot(r, result[i].imag, label=rf"$\alpha = {alpha:.2f}, \beta = {beta:.2f}$")

ax.set_xlabel("$|z|$")
ax.set_ylabel(r"$E_{\alpha, \beta}(z)$ (imag)")
ax.legend()

filename = pathlib.Path(__file__).with_suffix(".svg")
filename = filename.with_stem(f"{filename.stem}-imag")
fig.savefig(filename)
print(f"Saved results in {filename}")

# }}}
