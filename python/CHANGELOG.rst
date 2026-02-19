Version 0.1.9 (February 19)
---------------------------

Dependencies
^^^^^^^^^^^^

* Update to the new ``pyo3`` and ``rust-numpy`` 0.28 releases.

Maintenance
^^^^^^^^^^^

* Switch to `ty <https://github.com/astral-sh/ty>`__ for type checking.
* Small updates to build and CI (dependency groups and cleanup).

Version 0.1.8 (December 12th, 2025)
-----------------------------------

Dependencies
^^^^^^^^^^^^

* Updated the Rust edition to **2024**.
* Advertise support for Python 3.14.

Maintenance
^^^^^^^^^^^

* Update linting and formatting (for new ``ruff`` and ``mypy`` mainly).

Version 0.1.7 (November 21st, 2025)
-----------------------------------

Maintenance
^^^^^^^^^^^

* Add ``long_description`` to Python bindings, so that it shows up on PyPI.

Version 0.1.6 (November 21st, 2025)
-----------------------------------

Dependencies
^^^^^^^^^^^^

* Update to the new ``pyo3`` and ``rust-numpy`` 0.27 releases.

Version 0.1.5 (October 5th, 2025)
---------------------------------

Dependencies
^^^^^^^^^^^^

* Update to the new ``pyo3`` and ``rust-numpy`` 0.26 releases.

Version 0.1.4 (June 17th, 2025)
-------------------------------

Dependencies
^^^^^^^^^^^^

* Update to the new ``pyo3`` and ``rust-numpy`` 0.25 releases.

Maintenance
^^^^^^^^^^^

* Update linting and formatting (for new ``ruff`` and ``mypy`` mainly).
* Fix seed in tests to remove flakiness.
* Fix some small typos in README.

Version 0.1.3 (February 1st, 2025)
----------------------------------

Dependencies
^^^^^^^^^^^^

* Update to the new ``rand`` 0.9.0 release.

Maintenance
^^^^^^^^^^^

* Switch to using `just <https://just.systems/>`__ for development and on the CI.
* Add a nice image to the README.
* Update linting and formatting.

Version 0.1.2 (January 4th, 2025)
---------------------------------

Maintenance
^^^^^^^^^^^

* Switch ``README.rst`` to ``README.md`` for ``crates.io``.

Version 0.1.1 (January 4th, 2025)
---------------------------------

Maintenance
^^^^^^^^^^^

* Fix CI to produce wheels for x86_64 macOS.

Version 0.1.0 (January 4th, 2025)
---------------------------------

This is the initial release of the Python bindings for the ``mittagleffler``
Rust crate. It mainly provides a simple function to evaluate the Mittag-Leffler
function as:

.. code:: python

    from pymittagleffler import mittag_leffler

    z = np.linspace(0.0, 1.0) + 1j * np.linspace(0.0, 1.0)
    ml = mittag_leffler(z, alpha=1.0, beta=2.0)

The function accepts real and complex inputs, both scalars and :mod:`numpy` arrays
of any shape. The function is applied component wise, i.e. this is different than
the existing "Matrix" Mittag-Leffler function.
