# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib
from typing import TYPE_CHECKING

import numpy as np
import rich.logging

from pymittagleffler import mittag_leffler

if TYPE_CHECKING:
    from collections.abc import Callable

log = logging.getLogger(pathlib.Path(__file__).stem)
log.setLevel(logging.INFO)
log.addHandler(rich.logging.RichHandler())


Array = np.ndarray[tuple[int, ...], np.dtype[np.inexact]]


# {{{ benchmark


def timeit(stmt: Callable[[], object], *, repeat: int = 32) -> float:
    import timeit as _timeit

    # time it and disregard first run
    r = np.array(_timeit.repeat(stmt=stmt, repeat=repeat + 1, number=1))
    return float(np.min(r[1:]))


def benchmark_heatmap(
    alpha: Array,
    r: Array,
    *,
    beta: float,
    theta: float,
    batch_size: int = 32,
) -> Array:
    result = np.empty((alpha.size, r.size))

    for i, a in enumerate(alpha):
        for j, rj in enumerate(r):
            z = np.full(batch_size, rj * np.exp(1j * theta))
            elapsed = timeit(lambda z=z, a=a: mittag_leffler(z, a, beta))
            result[i, j] = elapsed / batch_size * 1e6

    return result


# }}}


# {{{ main


def main(
    filename: pathlib.Path | None,
    outfile: pathlib.Path | None,
    *,
    overwrite: bool = False,
) -> int:
    if outfile is None:
        outfile = pathlib.Path("bench-heatmap")

    if not overwrite and outfile.exists():
        log.error("File already exists (use --overwrite): '%s'", outfile)
        return 1

    if filename is not None:
        data = np.load(filename, allow_pickle=True)

        alpha = data["alpha"]
        r = data["r"]
        result = data["result"]
    else:
        alpha = np.linspace(0.1, 5.0, 64)
        r = np.logspace(-3.0, 2.0, 64)
        beta = 1.25
        theta = np.pi / 2

        result = benchmark_heatmap(
            alpha,
            r,
            beta=beta,
            theta=theta,
        )

        np.savez(
            outfile.with_suffix(".npz"),
            alpha=alpha,
            r=r,
            theta=theta,
            beta=beta,
            result=result,
        )

    # {{{ plot

    try:
        import matplotlib.pyplot as mp
        from matplotlib.colors import LogNorm
        from matplotlib.ticker import (
            FuncFormatter,
            LogLocator,
            NullFormatter,
            NullLocator,
        )
    except ImportError as exc:
        print("'matplotlib' package is not available for plotting")
        raise SystemExit(0) from exc

    from pymittagleffler import _set_recommended_matplotlib

    _set_recommended_matplotlib()
    fig = mp.figure(figsize=(12, 8))
    ax = fig.gca()

    agrid, rgrid = np.meshgrid(alpha, r)
    pcm = ax.pcolormesh(
        agrid,
        rgrid,
        result.T,
        shading="auto",
        norm=LogNorm(),
        cmap="viridis",
        antialiased=False,
    )

    ax.set_yscale("log")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$|z|$")
    ax.grid(visible=False, which="both")

    vmin = float(np.nanmin(result))
    vmax = float(np.nanmax(result))

    locator = LogLocator(base=10, subs=(1.0, 2.0, 4.0, 5.0, 7.0, 15.0))
    ticks = locator.tick_values(vmin, vmax)
    ticks = ticks[(ticks >= vmin) & (ticks <= vmax)]
    ticks = np.unique(np.concatenate(([vmin], ticks, [vmax])))

    cbar = fig.colorbar(pcm, ax=ax, label=r"Time per evaluation ($\mu$s)")
    cbar.set_ticks(ticks)
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.3g}"))
    cbar.ax.yaxis.set_minor_locator(NullLocator())
    cbar.ax.yaxis.set_minor_formatter(NullFormatter())
    fig.savefig(outfile)
    log.info("Saved plot in '%s'", outfile)

    # }}}

    return 0


# }}}

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("filename", nargs="?", type=pathlib.Path, default=None)
    parser.add_argument("-o", "--outfile", type=pathlib.Path, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Only show error messages",
    )
    args = parser.parse_args()

    if not args.quiet:
        log.setLevel(logging.INFO)

    raise SystemExit(main(args.filename, args.outfile, overwrite=args.overwrite))
