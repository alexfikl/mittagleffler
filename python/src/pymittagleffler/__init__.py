# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

"""
.. autoclass:: GarrappaMittagLeffler
    :members:

    .. attribute:: eps
        :type: float

        Tolerance used by the algorithm.

    .. automethod:: evaluate

        Evaluate the Mittag-Leffler function at a scalar argument *z*. Note
        that, unlike :func:`mittag_leffler`, this function does not use any
        special cases and always evaluates the Mittag-Leffler function using
        the algorithm from Garrappa (2015, `doi:10.1137/140971191
        <https://doi.org/10.1137/140971191>`__).

.. autofunction:: mittag_leffler

    Evaluate the Mittag-Leffler function with parameters *alpha* and *beta*.

    :arg z: any scalar or :class:`numpy.ndarray` of real or complex numbers.

"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

from ._bindings import GarrappaMittagLeffler, mittag_leffler

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ("GarrappaMittagLeffler", "mittag_leffler")


class Styles:
    def __init__(self, styles: dict[str, str]) -> None:
        self.styles = styles

    def get(self, styles: str | Sequence[str]) -> tuple[str, ...]:
        if isinstance(styles, str):
            styles = [styles]

        # NOTE: this silently does nothing if the styles are incorrectly named or
        # if scienceplots does not actually exist. This should be fine..
        return tuple(self.styles[style] for style in styles if style in self.styles)


def load_scienceplots_styles() -> Styles:
    from importlib.util import find_spec

    spec = find_spec("scienceplots")
    if spec is None:
        return Styles({})

    if spec.submodule_search_locations is None:
        return Styles({})

    styles_root = pathlib.Path(spec.submodule_search_locations[0]) / "styles"
    return Styles({f.stem: str(f) for f in styles_root.rglob("*") if f.is_file()})


def _set_recommended_matplotlib() -> None:
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as mp
    except ImportError:
        return

    # start off by resetting the defaults
    mpl.rcParams.update(mpl.rcParamsDefault)

    # NOTE: the 'petroff10' style is available for version >= 3.10.0 and changes
    # the 'prop_cycle' to the 10 colors that are more accessible
    prop_cycle = mp.rcParams["axes.prop_cycle"]
    if "petroff10" in mp.style.available:
        mp.style.use("petroff10")
        prop_cycle = mp.rcParams["axes.prop_cycle"]

    # load scienceplots, if available
    scienceplots = load_scienceplots_styles()
    mp.style.use(scienceplots.get(["science", "ieee"]))

    mp.rc("figure", figsize=(10, 10), dpi=300)
    mp.rc("figure.constrained_layout", use=True)
    mp.rc("text", usetex=True)
    mp.rc(
        "legend",
        fontsize=20,
        frameon=True,
        fancybox=False,
        facecolor="white",
        edgecolor="black",
        framealpha=1.0,
        columnspacing=0.8,
    )
    mp.rc("lines", linewidth=2, markersize=10)
    mp.rc("axes", labelsize=28, titlesize=28, grid=True)
    mp.rc("axes.grid", axis="both", which="both")
    mp.rc("axes", prop_cycle=prop_cycle)
    mp.rc("xtick", labelsize=20, direction="out")
    mp.rc("ytick", labelsize=20, direction="out")
    mp.rc("xtick.major", size=6.5, width=1.5)
    mp.rc("ytick.major", size=6.5, width=1.5)
    mp.rc("xtick.minor", size=4.0)
    mp.rc("ytick.minor", size=4.0)
