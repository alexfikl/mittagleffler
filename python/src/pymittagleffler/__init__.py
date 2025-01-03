# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

"""
.. autoclass:: GarrappaMittagLeffler
    :members:

    .. attribute:: eps
        :type: float

        Tolerance used by the algorithm.

    .. automethod:: evaluate

        Evaluate the Mittag-Leffler function at a scalar argument *z*.

.. autofunction:: mittag_leffler

    Evaluate the Mittag-Leffler function with parameters *alpha* and *beta*.

    :arg z: any scalar or :class:`numpy.ndarray` of real or complex numbers.
"""

from __future__ import annotations

from ._bindings import GarrappaMittagLeffler, mittag_leffler

__all__ = ("GarrappaMittagLeffler", "mittag_leffler")


def _set_recommended_matplotlib() -> None:
    from contextlib import suppress

    try:
        import matplotlib.pyplot as mp
    except ImportError:
        return

    # start off by resetting the defaults
    import matplotlib as mpl

    # NOTE: preserve existing colors (the ones in "science" are ugly)
    mpl.rcParams.update(mpl.rcParamsDefault)
    prop_cycle = mp.rcParams["axes.prop_cycle"]
    with suppress(ImportError):
        import scienceplots  # noqa: F401

        mp.style.use(["science", "ieee"])

    # NOTE: the 'petroff10' style is available for version >= 3.10.0 and changes
    # the 'prop_cycle' to the 10 colors that are more accessible
    if "petroff10" in mp.style.available:
        mp.style.use("petroff10")
        prop_cycle = mp.rcParams["axes.prop_cycle"]

    defaults: dict[str, dict[str, object]] = {
        "figure": {
            "figsize": (8, 8),
            "dpi": 300,
            "constrained_layout.use": True,
        },
        "text": {"usetex": True},
        "legend": {"fontsize": 20},
        "lines": {"linewidth": 2, "markersize": 10},
        "axes": {
            "labelsize": 28,
            "titlesize": 28,
            "grid": True,
            "grid.axis": "both",
            "grid.which": "both",
            "prop_cycle": prop_cycle,
        },
        "xtick": {"labelsize": 20, "direction": "out"},
        "ytick": {"labelsize": 20, "direction": "out"},
        "xtick.major": {"size": 6.5, "width": 1.5},
        "ytick.major": {"size": 6.5, "width": 1.5},
        "xtick.minor": {"size": 4.0},
        "ytick.minor": {"size": 4.0},
    }

    for group, params in defaults.items():
        mp.rc(group, **params)
